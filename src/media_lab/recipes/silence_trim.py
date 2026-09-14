"""Automatic silence and pause trimming recipe (jump-cut audio/video).

Detects dead time, long pauses, and silence intervals using ffmpeg's silencedetect
and extracts only active segments, producing tight pacing for Reels, TikTok, and podcasts.
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
from ..validation import check_range
from ..verify import Expectations, verify_render

DEFAULT_MIN_SILENCE_S = 0.4
MIN_SILENCE_S = 0.1
MAX_SILENCE_S = 5.0

DEFAULT_NOISE_THRESHOLD_DB = -38.0
MIN_NOISE_THRESHOLD_DB = -60.0
MAX_NOISE_THRESHOLD_DB = -15.0

SILENCE_START_PATTERN = re.compile(r"silence_start:\s*([0-9\.]+)")
SILENCE_END_PATTERN = re.compile(r"silence_end:\s*([0-9\.]+)")


@dataclass(frozen=True, slots=True)
class SilenceInterval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class SilenceTrimResult:
    """Outcome of silence trimming."""

    output: Path
    media: MediaInfo
    original_duration_s: float
    new_duration_s: float
    removed_duration_s: float
    cuts_count: int


def detect_silence_intervals(
    source: Path,
    config: Config,
    *,
    min_silence_s: float = DEFAULT_MIN_SILENCE_S,
    noise_db: float = DEFAULT_NOISE_THRESHOLD_DB,
) -> list[SilenceInterval]:
    """Detect silence segments in an audio or video file."""
    cmd = [
        "-i",
        str(source),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence_s}",
        "-f",
        "null",
        "-",
    ]
    res = run_ffmpeg(cmd, config)
    stderr = res.stderr

    intervals: list[SilenceInterval] = []
    starts = [float(m.group(1)) for m in SILENCE_START_PATTERN.finditer(stderr)]
    ends = [float(m.group(1)) for m in SILENCE_END_PATTERN.finditer(stderr)]

    # Pair starts with ends
    for start, end in zip(starts, ends, strict=False):
        if end > start:
            intervals.append(SilenceInterval(start=start, end=end))

    return intervals


def calculate_keep_intervals(
    total_duration: float,
    silence_intervals: list[SilenceInterval],
    *,
    padding_s: float = 0.08,
) -> list[tuple[float, float]]:
    """Compute active non-silent intervals with a subtle padding around speech."""
    if not silence_intervals:
        return [(0.0, total_duration)]

    keep: list[tuple[float, float]] = []
    current_pos = 0.0

    for s in silence_intervals:
        # End active segment slightly into the silence
        segment_end = min(total_duration, s.start + padding_s)
        if segment_end > current_pos + 0.05:
            keep.append((round(current_pos, 3), round(segment_end, 3)))
        # Start next active segment slightly before silence ends
        current_pos = max(0.0, s.end - padding_s)

    if current_pos < total_duration - 0.05:
        keep.append((round(current_pos, 3), round(total_duration, 3)))

    return keep


def trim_silence(
    source: Path | str,
    output: Path | str,
    config: Config,
    *,
    min_silence_s: float = DEFAULT_MIN_SILENCE_S,
    noise_db: float = DEFAULT_NOISE_THRESHOLD_DB,
    padding_s: float = 0.08,
    force: bool = False,
) -> SilenceTrimResult:
    """Trim silence intervals and long pauses from video or audio files.

    Args:
        source: Input video or audio.
        output: Output file with pauses removed.
        config: Project configuration.
        min_silence_s: Minimum duration of a pause to be removed.
        noise_db: Noise floor threshold in dB (default -38.0).
        padding_s: Speech buffer padding in seconds (default 0.08s).
        force: Overwrite existing destination.
    """
    check_range(min_silence_s, MIN_SILENCE_S, MAX_SILENCE_S, "min_silence_s")
    check_range(noise_db, MIN_NOISE_THRESHOLD_DB, MAX_NOISE_THRESHOLD_DB, "noise_db")

    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    source_info = probe(resolved_source, config)
    if not source_info.has_audio:
        raise ValidationError(f"source has no audio stream to detect silence: {resolved_source}")

    total_duration = source_info.duration_s
    silence_intervals = detect_silence_intervals(
        resolved_source, config, min_silence_s=min_silence_s, noise_db=noise_db
    )

    keep_intervals = calculate_keep_intervals(
        total_duration, silence_intervals, padding_s=padding_s
    )

    # If no pauses found or only one full segment, copy directly
    if len(keep_intervals) <= 1 and (
        not silence_intervals or keep_intervals[0][1] - keep_intervals[0][0] >= total_duration - 0.1
    ):
        import shutil

        shutil.copy2(resolved_source, resolved_output)
        final_info = probe(resolved_output, config)
        return SilenceTrimResult(
            output=resolved_output,
            media=final_info,
            original_duration_s=round(total_duration, 2),
            new_duration_s=round(total_duration, 2),
            removed_duration_s=0.0,
            cuts_count=0,
        )

    # Build select / aselect expression: between(t, start, end) + ...
    select_clauses = [f"between(t,{start:.3f},{end:.3f})" for start, end in keep_intervals]
    select_expr = "+".join(select_clauses)

    args = ["-i", str(resolved_source)]

    if source_info.has_video:
        filtergraph = (
            f"[0:v]select='{select_expr}',setpts=N/FRAME_RATE/TB[v];"
            f"[0:a]aselect='{select_expr}',asetpts=N/SR/TB[a]"
        )
        args.extend(
            [
                "-filter_complex",
                filtergraph,
                "-map",
                "[v]",
                "-map",
                "[a]",
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
        filtergraph = f"aselect='{select_expr}',asetpts=N/SR/TB"
        args.extend(["-af", filtergraph, "-c:a", "aac", "-b:a", "256k"])

    args.append(str(resolved_output))
    run_ffmpeg(args, config)

    final_info = verify_render(
        resolved_output,
        config,
        Expectations(requires_audio=True, requires_video=source_info.has_video),
    )
    removed_s = max(0.0, total_duration - final_info.duration_s)

    return SilenceTrimResult(
        output=resolved_output,
        media=final_info,
        original_duration_s=round(total_duration, 2),
        new_duration_s=round(final_info.duration_s, 2),
        removed_duration_s=round(removed_s, 2),
        cuts_count=len(silence_intervals),
    )
