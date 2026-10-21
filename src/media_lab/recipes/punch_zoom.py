"""Recipe: retention punch-in zoom for video pacing and dynamic visual interest.

Applies sudden or smooth camera punch-in zooms (1.1x–1.25x) at key moments,
sentence emphasis, or periodic cadence to maximize viewer engagement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..errors import ValidationError
from ..ffmpeg import run_ffmpeg
from ..paths import ensure_readable_source, ensure_writable_output
from ..probe import MediaInfo, probe
from ..verify import Expectations, verify_render

MIN_SCALE = 1.01
MAX_SCALE = 2.50


@dataclass(frozen=True, slots=True)
class ZoomCue:
    """A single punch-in zoom interval."""

    start_s: float
    duration_s: float
    scale: float = 1.15
    cx: float = 0.5  # Normalized horizontal center [0.0, 1.0]
    cy: float = (
        0.4  # Normalized vertical center [0.0, 1.0] (0.4 is optimal for human face/headroom)
    )
    transition: str = "cut"  # "cut" or "fade"
    fade_duration_s: float = 0.2


@dataclass(frozen=True, slots=True)
class ZoomResult:
    """Outcome of applying retention punch-in zooms."""

    output: Path
    media: MediaInfo
    cues_applied: int


def generate_auto_zoom_cues(
    total_duration_s: float,
    *,
    interval_s: float = 5.0,
    zoom_duration_s: float = 2.0,
    scale: float = 1.15,
) -> list[ZoomCue]:
    """Generate rhythmic punch-in zoom cues across video duration."""
    if total_duration_s <= 0 or interval_s <= 0 or zoom_duration_s <= 0:
        return []

    cues: list[ZoomCue] = []
    current_t = interval_s

    while current_t + 0.5 < total_duration_s:
        actual_duration = min(zoom_duration_s, total_duration_s - current_t)
        if actual_duration >= 0.5:
            cues.append(
                ZoomCue(
                    start_s=round(current_t, 2),
                    duration_s=round(actual_duration, 2),
                    scale=scale,
                )
            )
        current_t += interval_s + actual_duration

    if not cues and total_duration_s >= 0.8:
        cue_dur = min(zoom_duration_s, total_duration_s * 0.5)
        cues.append(
            ZoomCue(
                start_s=round(total_duration_s * 0.25, 2),
                duration_s=round(cue_dur, 2),
                scale=scale,
            )
        )

    return cues


def punch_zoom(
    source: Path | str,
    output: Path | str,
    config: Config,
    *,
    cues: list[ZoomCue] | None = None,
    auto_interval_s: float | None = None,
    auto_zoom_duration_s: float = 2.0,
    auto_scale: float = 1.15,
    force: bool = False,
) -> ZoomResult:
    """Apply dynamic camera punch-in zooms to a video.

    Args:
        source: Input video path.
        output: Destination video path.
        config: System configuration.
        cues: Explicit list of ZoomCue intervals.
        auto_interval_s: If provided, automatically places zoom intervals every N seconds.
        auto_zoom_duration_s: Duration of auto zooms in seconds.
        auto_scale: Scaling factor for auto zooms (e.g. 1.15 for 115%).
        force: Overwrite output if it already exists.

    Returns:
        ZoomResult with verified media info and count of applied cues.
    """
    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    source_info = probe(resolved_source, config)
    if not source_info.has_video or source_info.width is None or source_info.height is None:
        raise ValidationError(f"source {resolved_source} has no valid video stream")

    duration = source_info.duration_s or 0.0

    # Determine cues
    resolved_cues: list[ZoomCue] = []
    if cues is not None:
        resolved_cues.extend(cues)
    elif auto_interval_s is not None and auto_interval_s > 0:
        resolved_cues = generate_auto_zoom_cues(
            duration,
            interval_s=auto_interval_s,
            zoom_duration_s=auto_zoom_duration_s,
            scale=auto_scale,
        )

    if not resolved_cues:
        raise ValidationError("no zoom cues specified (provide cues or auto_interval_s > 0)")

    # Validate cues
    sw = source_info.width
    sh = source_info.height

    for i, c in enumerate(resolved_cues):
        if c.start_s < 0:
            raise ValidationError(f"cue {i} start_s must be >= 0, got {c.start_s}")
        if c.duration_s <= 0:
            raise ValidationError(f"cue {i} duration_s must be > 0, got {c.duration_s}")
        if not (MIN_SCALE <= c.scale <= MAX_SCALE):
            raise ValidationError(
                f"cue {i} scale must be between {MIN_SCALE} and {MAX_SCALE}, got {c.scale}"
            )
        if not (0.0 <= c.cx <= 1.0 and 0.0 <= c.cy <= 1.0):
            raise ValidationError(f"cue {i} center must be in [0.0, 1.0], got ({c.cx}, {c.cy})")
        if c.transition not in {"cut", "fade"}:
            raise ValidationError(
                f"cue {i} transition must be 'cut' or 'fade', got {c.transition!r}"
            )

    # Build filtergraph
    # We chain overlays for each zoom cue sequentially
    filter_chains: list[str] = []
    current_stream = "0:v"

    for idx, cue in enumerate(resolved_cues):
        scaled_w = int(round(sw * cue.scale))
        # Ensure even width
        scaled_w = scaled_w - (scaled_w % 2)

        # Scaled height maintaining aspect ratio
        scaled_h = int(round(sh * cue.scale))
        scaled_h = scaled_h - (scaled_h % 2)

        # Crop window inside the scaled frame to match original sw x sh
        desired_x = int(round(cue.cx * scaled_w - sw / 2))
        desired_y = int(round(cue.cy * scaled_h - sh / 2))

        crop_x = max(0, min(scaled_w - sw, desired_x))
        crop_x = crop_x - (crop_x % 2)
        crop_y = max(0, min(scaled_h - sh, desired_y))
        crop_y = crop_y - (crop_y % 2)

        split_base = f"base_{idx}"
        split_zoom = f"z_in_{idx}"
        zoomed_crop = f"zoomed_{idx}"
        overlay_out = f"out_v_{idx}"

        filter_chains.append(f"[{current_stream}]split=2[{split_base}][{split_zoom}]")

        zoom_filters = [
            f"scale={scaled_w}:{scaled_h}",
            f"crop={sw}:{sh}:{crop_x}:{crop_y}",
        ]

        if cue.transition == "fade":
            fade_dur = min(cue.fade_duration_s, cue.duration_s / 2)
            out_start = max(cue.start_s, cue.start_s + cue.duration_s - fade_dur)
            zoom_filters.extend(
                [
                    "format=yuva420p",
                    f"fade=t=in:st={cue.start_s}:d={fade_dur}:alpha=1",
                    f"fade=t=out:st={out_start}:d={fade_dur}:alpha=1",
                ]
            )

        filter_chains.append(f"[{split_zoom}]{','.join(zoom_filters)}[{zoomed_crop}]")

        end_s = cue.start_s + cue.duration_s
        enable_expr = f"between(t,{cue.start_s},{end_s})"
        filter_chains.append(
            f"[{split_base}][{zoomed_crop}]overlay=0:0:enable='{enable_expr}'[{overlay_out}]"
        )
        current_stream = overlay_out

    filtergraph = ";".join(filter_chains)

    cmd: list[str] = [
        "-i",
        str(resolved_source),
        "-filter_complex",
        filtergraph,
        "-map",
        f"[{current_stream}]",
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
            width=sw,
            height=sh,
            requires_video=True,
            requires_audio=source_info.has_audio,
        ),
    )

    return ZoomResult(
        output=resolved_output,
        media=final_info,
        cues_applied=len(resolved_cues),
    )
