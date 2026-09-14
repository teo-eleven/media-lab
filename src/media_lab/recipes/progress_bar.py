"""Recipe: Animated social retention progress bar for Reels / Shorts / TikTok.

Draws a smooth, real-time progress bar along the bottom or top of the video
timeline to boost audience retention and completion rates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..errors import ValidationError
from ..ffmpeg import run_ffmpeg
from ..paths import ensure_readable_source, ensure_writable_output
from ..probe import MediaInfo, probe
from ..verify import Expectations, verify_render

COLOR_MAP: dict[str, str] = {
    "yellow": "yellow@0.9",
    "red": "red@0.9",
    "white": "white@0.9",
    "cyan": "cyan@0.9",
    "green": "lime@0.9",
    "blue": "#00B0FF@0.9",
    "tiktok": "#FF0050@0.95",
    "orange": "orange@0.9",
}


@dataclass(frozen=True, slots=True)
class ProgressBarResult:
    """Outcome of adding an animated progress bar."""

    output: Path
    media: MediaInfo
    position: str
    height: int
    color: str


def add_progress_bar(
    source: Path | str,
    output: Path | str,
    config: Config,
    *,
    position: str = "bottom",
    height: int = 6,
    color: str = "yellow",
    force: bool = False,
) -> ProgressBarResult:
    """Add an animated progress bar to a video.

    Args:
        source: Input video file.
        output: Destination video file.
        config: Project configuration.
        position: 'bottom' or 'top' (default 'bottom').
        height: Bar thickness in pixels (default 6, rounded to even).
        color: Bar color name ('yellow', 'red', 'white', 'tiktok', etc.) or hex.
        force: Overwrite existing destination.

    Returns:
        ProgressBarResult with rendered metadata.
    """
    if position not in {"bottom", "top"}:
        raise ValidationError(f"position must be 'bottom' or 'top', got {position!r}")

    # Validate color format to prevent filtergraph option injection
    if not re.match(r"^#?[a-zA-Z0-9]+(@[0-9.]+)?$", color):
        raise ValidationError(f"invalid color specification: {color!r}")

    # Ensure even height for YUV420p chroma alignment
    h_bar = max(2, height if height % 2 == 0 else height + 1)
    color_val = COLOR_MAP.get(color.lower(), color)

    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    source_info = probe(resolved_source, config)
    if not source_info.has_video or source_info.height is None:
        raise ValidationError(f"source has no video stream: {resolved_source}")

    dur = source_info.duration_s
    if dur <= 0:
        raise ValidationError(f"invalid source duration: {dur}")

    # Track Y position formula: 'ih-h_bar' for bottom, '0' for top
    track_y = f"ih-{h_bar}" if position == "bottom" else "0"
    overlay_y = f"H-{h_bar}" if position == "bottom" else "0"

    # Pattern: static background track + overlay moving bar sliding from -w to 0
    # Clamped with min(0, ...) so bar remains full-width if t exceeds probed duration
    filtergraph = (
        f"[0:v]drawbox=x=0:y={track_y}:w=iw:h={h_bar}:color=black@0.4:t=fill[track];"
        f"[track]split[main][strip];"
        f"[strip]crop=iw:{h_bar}:0:{track_y},drawbox=x=0:y=0:w=iw:h={h_bar}:"
        f"color={color_val}:t=fill[bar];"
        f"[main][bar]overlay=x='min(0, -w+w*(t/{dur:.3f}))':y={overlay_y}:"
        "eval=frame,format=yuv420p[out_v]"
    )

    cmd: list[str] = [
        "-i",
        str(resolved_source),
        "-filter_complex",
        filtergraph,
        "-map",
        "[out_v]",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
    ]

    if source_info.has_audio:
        cmd.extend(["-map", "0:a", "-c:a", "copy"])
    else:
        cmd.append("-an")

    cmd.append(str(resolved_output))
    run_ffmpeg(cmd, config)

    final_info = verify_render(
        resolved_output,
        config,
        Expectations(
            duration_s=source_info.duration_s,
            width=source_info.width,
            height=source_info.height,
            requires_video=True,
            requires_audio=source_info.has_audio,
        ),
    )

    return ProgressBarResult(
        output=resolved_output,
        media=final_info,
        position=position,
        height=h_bar,
        color=color,
    )
