"""Recipe: Speed ramping (slow-motion / timelapse) with pitch-preserved audio.

Allows changing video and audio playback speed (0.1x – 10.0x) while preserving
audio pitch via chained FFmpeg atempo filters and precise setpts recalculation.
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

MIN_SPEED: float = 0.1
MAX_SPEED: float = 10.0


@dataclass(frozen=True, slots=True)
class SpeedResult:
    """Outcome of speed modification."""

    output: Path
    media: MediaInfo
    speed_factor: float
    original_duration_s: float
    new_duration_s: float


def generate_atempo_chain(speed: float) -> str:
    """Decompose an arbitrary speed factor > 0 into valid chained atempo filters.

    FFmpeg's atempo filter strictly requires values in [0.5, 2.0]. Chaining
    multiple atempo instances multiplies their speeds together.
    """
    if speed <= 0:
        raise ValidationError(f"speed must be > 0, got {speed}")

    factors: list[float] = []
    current = float(speed)

    while current > 2.0:
        factors.append(2.0)
        current /= 2.0

    while current < 0.5:
        factors.append(0.5)
        current /= 0.5

    factors.append(round(current, 6))
    return ",".join(f"atempo={f}" for f in factors)


def change_speed(
    source: Path | str,
    output: Path | str,
    config: Config,
    *,
    speed: float = 1.0,
    force: bool = False,
) -> SpeedResult:
    """Change playback speed of a video or audio file.

    Args:
        source: Input media file.
        output: Destination media file.
        config: Project configuration.
        speed: Speed factor (e.g. 0.5 for 2x slow-mo, 2.0 for 2x timelapse).
        force: Overwrite existing destination.

    Returns:
        SpeedResult with duration metrics.
    """
    if not (MIN_SPEED <= speed <= MAX_SPEED):
        raise ValidationError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}, got {speed}")

    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    source_info = probe(resolved_source, config)
    expected_duration = source_info.duration_s / speed

    cmd: list[str] = ["-i", str(resolved_source)]

    if source_info.has_video:
        pts_factor = 1.0 / speed
        v_filter = f"setpts={pts_factor:.6f}*(PTS-STARTPTS)"

        if source_info.has_audio:
            a_filter = generate_atempo_chain(speed)
            cmd.extend(
                [
                    "-filter_complex",
                    f"[0:v]{v_filter}[out_v];[0:a]{a_filter}[out_a]",
                    "-map",
                    "[out_v]",
                    "-map",
                    "[out_a]",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "256k",
                ]
            )
        else:
            cmd.extend(
                [
                    "-vf",
                    v_filter,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-an",
                ]
            )
    else:
        # Audio-only file
        if not source_info.has_audio:
            raise ValidationError(f"source has neither video nor audio: {resolved_source}")
        a_filter = generate_atempo_chain(speed)
        out_suffix = resolved_output.suffix.lower()
        if out_suffix == ".mp3":
            cmd.extend(["-af", a_filter, "-c:a", "libmp3lame", "-b:a", "256k"])
        elif out_suffix in {".wav", ".aiff"}:
            cmd.extend(["-af", a_filter, "-c:a", "pcm_s16le"])
        else:
            cmd.extend(["-af", a_filter, "-c:a", "aac", "-b:a", "256k"])

    cmd.append(str(resolved_output))
    run_ffmpeg(cmd, config)

    final_info = verify_render(
        resolved_output,
        config,
        Expectations(
            duration_s=round(expected_duration, 2),
            requires_video=source_info.has_video,
            requires_audio=source_info.has_audio,
        ),
    )

    return SpeedResult(
        output=resolved_output,
        media=final_info,
        speed_factor=speed,
        original_duration_s=round(source_info.duration_s, 2),
        new_duration_s=round(final_info.duration_s, 2),
    )
