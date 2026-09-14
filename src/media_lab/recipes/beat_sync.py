"""Recipe: Audio rhythm, BPM and transient beat detection for musical video editing.

Extracts audio transients and rhythmic energy peaks to enable cutting on the beat,
triggering retention zoom punch-ins, and synchronizing visual SFX with music drops.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
from ..errors import MediaLabError, ValidationError
from ..paths import ensure_readable_source
from ..probe import probe


@dataclass(frozen=True, slots=True)
class BeatSyncResult:
    """Outcome of audio beat and rhythm analysis."""

    source: Path
    beats_s: tuple[float, ...]
    estimated_bpm: float | None
    duration_s: float
    total_beats: int


def detect_beats(
    source: Path | str,
    config: Config,
    *,
    hop_length_ms: float = 25.0,
    sensitivity: float = 1.2,
    min_beat_interval_s: float = 0.25,
) -> BeatSyncResult:
    """Analyze audio track and detect rhythm beats and onset transients.

    Args:
        source: Audio or video file with sound.
        config: Project configuration.
        hop_length_ms: Frame analysis window in milliseconds (default 25ms).
        sensitivity: Standard deviation multiplier for energy peak threshold (default 1.2).
        min_beat_interval_s: Minimum interval between detected beats (default 0.25s = max 240 BPM).

    Returns:
        BeatSyncResult containing list of beat timestamps in seconds and estimated BPM.
    """
    resolved_source = ensure_readable_source(source)
    source_info = probe(resolved_source, config)
    if not source_info.has_audio:
        raise ValidationError(f"source {resolved_source} has no audio track for beat detection")

    dur = source_info.duration_s
    sr = 22050  # 22.05kHz mono is optimal for transient envelope analysis
    hop_samples = int(sr * (hop_length_ms / 1000.0))

    cmd = [
        str(config.ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(resolved_source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "f32le",
        "-",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            timeout=120,
        )
    except subprocess.CalledProcessError as err:
        stderr_text = err.stderr.decode("utf-8", errors="replace")
        raise MediaLabError(f"ffmpeg audio extraction failed: {stderr_text}") from err

    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if len(samples) == 0:
        return BeatSyncResult(
            source=resolved_source,
            beats_s=(),
            estimated_bpm=None,
            duration_s=round(dur, 2),
            total_beats=0,
        )

    # 1. Compute RMS energy in sliding windows
    num_frames = len(samples) // hop_samples
    if num_frames < 2:
        return BeatSyncResult(
            source=resolved_source,
            beats_s=(),
            estimated_bpm=None,
            duration_s=round(dur, 2),
            total_beats=0,
        )

    frames = samples[: num_frames * hop_samples].reshape(num_frames, hop_samples)
    rms_energy = np.sqrt(np.mean(frames**2, axis=1) + 1e-9)

    # 2. Compute onset energy flux (positive derivative)
    flux = np.diff(rms_energy, prepend=rms_energy[0])
    flux = np.maximum(0.0, flux)

    # 3. Peak picking with adaptive threshold
    mean_flux = float(np.mean(flux))
    std_flux = float(np.std(flux))
    threshold = mean_flux + sensitivity * std_flux

    min_frame_distance = max(1, int(min_beat_interval_s / (hop_length_ms / 1000.0)))

    detected_indices: list[int] = []
    last_idx = -min_frame_distance

    for i in range(1, len(flux) - 1):
        if (
            flux[i] > threshold
            and flux[i] > flux[i - 1]
            and flux[i] >= flux[i + 1]
            and (i - last_idx >= min_frame_distance)
        ):
            detected_indices.append(i)
            last_idx = i

    time_per_frame = hop_length_ms / 1000.0
    beats_s = tuple(round(idx * time_per_frame, 3) for idx in detected_indices)

    # 4. Estimate tempo (BPM) from median inter-beat interval
    bpm: float | None = None
    if len(beats_s) >= 4:
        diffs = np.diff(beats_s)
        # Filter realistic musical intervals (0.3s - 1.2s -> 50 - 200 BPM)
        musical_diffs = diffs[(diffs >= 0.3) & (diffs <= 1.2)]
        if len(musical_diffs) > 0:
            median_interval = float(np.median(musical_diffs))
            if median_interval > 0:
                raw_bpm = 60.0 / median_interval
                # Normalize to standard 70 - 150 BPM range
                while raw_bpm < 70:
                    raw_bpm *= 2
                while raw_bpm > 160:
                    raw_bpm /= 2
                bpm = round(raw_bpm, 1)

    return BeatSyncResult(
        source=resolved_source,
        beats_s=beats_s,
        estimated_bpm=bpm,
        duration_s=round(dur, 2),
        total_beats=len(beats_s),
    )
