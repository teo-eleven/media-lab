"""Recipe: Auto-Hook — Viral Energy Analyzer and Slicing Engine.

Inspired by Opus Clip and modern short-form viral repurposing algorithms.
Scans long-form video or audio (podcasts, interviews, streams) for peak acoustic energy
and vocal dynamic presence, automatically extracting the optimal 30–60s clip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
from ..errors import ValidationError
from ..ffmpeg import run_ffmpeg
from ..paths import ensure_readable_source, ensure_writable_output, work_directory
from ..probe import MediaInfo, probe
from ..verify import Expectations, verify_render


@dataclass(frozen=True, slots=True)
class HookSegment:
    """Detected peak viral hook interval."""

    start_s: float
    end_s: float
    duration_s: float
    score: float


@dataclass(frozen=True, slots=True)
class HookResult:
    """Outcome of extracting a viral hook clip."""

    output: Path
    media: MediaInfo
    hook: HookSegment


def detect_viral_hook(
    source: Path | str,
    config: Config,
    *,
    window_duration_s: float = 30.0,
    step_s: float = 0.5,
) -> HookSegment:
    """Analyze audio energy across sliding windows and locate the peak hook section.

    Args:
        source: Input media file.
        config: System configuration.
        window_duration_s: Desired target window length in seconds.
        step_s: Sliding step interval in seconds.

    Returns:
        HookSegment with start, end, duration, and energy score.
    """
    resolved_source = ensure_readable_source(source)
    source_info = probe(resolved_source, config)

    if not source_info.has_audio:
        raise ValidationError(
            f"Source has no audio track to detect vocal energy: {resolved_source}"
        )

    total_dur = source_info.duration_s
    if total_dur < window_duration_s:
        raise ValidationError(
            f"Source duration ({total_dur:.2f}s) is shorter than window ({window_duration_s:.2f}s)"
        )

    # Fast audio extraction to 8000Hz mono raw PCM for instant numpy analysis
    work = work_directory(config, "auto_hook")
    raw_pcm = work / f"{resolved_source.stem}_analysis.pcm"

    cmd = [
        "-i",
        str(resolved_source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "8000",
        "-f",
        "s16le",
        "-y",
        str(raw_pcm),
    ]
    run_ffmpeg(cmd, config)

    if not raw_pcm.is_file():
        raise ValidationError("Failed to extract audio track for viral energy analysis.")

    samples = np.fromfile(raw_pcm, dtype=np.int16).astype(np.float32)
    sample_rate = 8000
    window_samples = int(window_duration_s * sample_rate)
    step_samples = max(1, int(step_s * sample_rate))

    if len(samples) < window_samples:
        return HookSegment(
            start_s=0.0,
            end_s=total_dur,
            duration_s=total_dur,
            score=1.0,
        )

    best_score = -1.0
    best_start_idx = 0

    # Sliding window RMS calculation
    for start in range(0, len(samples) - window_samples + 1, step_samples):
        chunk = samples[start : start + window_samples]
        rms = float(np.sqrt(np.mean(chunk**2)))
        # Bonus for dynamic variance (avoid continuous static noise/hum)
        std = float(np.std(chunk))
        score = rms * (std + 1.0)

        if score > best_score:
            best_score = score
            best_start_idx = start

    start_s = round(best_start_idx / sample_rate, 2)
    end_s = round(min(total_dur, start_s + window_duration_s), 2)
    actual_dur = round(end_s - start_s, 2)

    return HookSegment(
        start_s=start_s,
        end_s=end_s,
        duration_s=actual_dur,
        score=best_score,
    )


def extract_viral_hook(
    source: Path | str,
    output: Path | str,
    config: Config,
    *,
    target_duration_s: float = 30.0,
    fade_s: float = 0.15,
    force: bool = False,
) -> HookResult:
    """Detect and extract the highest-energy viral clip from a longer media file.

    Args:
        source: Long-form input video or audio.
        output: Destination sliced clip path.
        config: System configuration.
        target_duration_s: Desired length of the sliced hook clip.
        fade_s: Audio fade-in and fade-out duration to avoid pops.
        force: Overwrite existing output.

    Returns:
        HookResult with the final path and detected hook metadata.
    """
    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    hook = detect_viral_hook(resolved_source, config, window_duration_s=target_duration_s)
    source_info = probe(resolved_source, config)

    # Audio fade filter
    fade_out_st = max(0.0, hook.duration_s - fade_s)
    af_filter = f"afade=t=in:ss=0:d={fade_s},afade=t=out:st={fade_out_st:.2f}:d={fade_s}"

    cmd = [
        "-ss",
        f"{hook.start_s:.2f}",
        "-t",
        f"{hook.duration_s:.2f}",
        "-i",
        str(resolved_source),
    ]

    if source_info.has_video:
        cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast"])

    if source_info.has_audio:
        cmd.extend(["-af", af_filter, "-c:a", "aac"])

    cmd.extend(["-movflags", "+faststart", "-y", str(resolved_output)])

    run_ffmpeg(cmd, config)

    rendered_info = verify_render(
        resolved_output,
        config,
        expected=Expectations(
            duration_s=hook.duration_s,
            duration_tolerance_s=0.5,
            width=source_info.width if source_info.has_video else None,
            height=source_info.height if source_info.has_video else None,
            requires_video=source_info.has_video,
            requires_audio=source_info.has_audio,
        ),
    )

    return HookResult(
        output=resolved_output,
        media=rendered_info,
        hook=hook,
    )
