"""Studio vocal mastering and audio enhancement recipe.

Cleans and masters spoken voice or music using a tuned broadcast filtergraph:
rumble high-pass, warmth and presence parametric EQ, de-esser sibilance reduction,
broadcast dynamic leveling compression, and EBU R128 loudness normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..errors import ValidationError
from ..ffmpeg import run_ffmpeg
from ..paths import ensure_readable_source, ensure_writable_output
from ..probe import MediaInfo, probe
from ..validation import check_range
from ..verify import Expectations, verify_render

DEFAULT_TARGET_LUFS = -14.0
MIN_TARGET_LUFS = -30.0
MAX_TARGET_LUFS = -8.0

DEFAULT_VOICE_PROFILE = "podcast"
VOICE_PROFILES = ("podcast", "warm", "crisp", "clean", "radio")


@dataclass(frozen=True, slots=True)
class AudioEnhanceResult:
    """Outcome of audio enhancement."""

    output: Path
    media: MediaInfo
    profile: str
    target_lufs: float


def _build_enhance_filtergraph(
    profile: str,
    target_lufs: float,
    *,
    de_ess: bool = True,
    noise_gate: bool = True,
) -> str:
    """Construct an audio filtergraph for studio vocal mastering."""
    filters: list[str] = [
        # 1. High-pass filter at 80Hz: remove mic rumble, table bumps, air conditioning hum
        "highpass=f=80",
        # 2. Low-pass filter at 15kHz: remove high-frequency hiss
        "lowpass=f=15000",
    ]

    # 3. Parametric EQ based on voice profile
    if profile == "podcast":
        # Balanced broadcast: warm chest body + crisp presence
        filters.extend(
            [
                "equalizer=f=220:t=q:w=1.2:g=1.5",
                "equalizer=f=2800:t=q:w=2.0:g=-1.5",
                "equalizer=f=4500:t=q:w=1.4:g=2.5",
            ]
        )
    elif profile == "warm":
        # Rich, warm, radio host sound
        filters.extend(
            [
                "equalizer=f=200:t=q:w=1.0:g=2.5",
                "equalizer=f=500:t=q:w=1.5:g=1.0",
                "equalizer=f=3500:t=q:w=1.5:g=1.5",
            ]
        )
    elif profile == "crisp":
        # Highly intelligible for social media, TikTok, Reels
        filters.extend(
            [
                "equalizer=f=250:t=q:w=1.5:g=-1.0",
                "equalizer=f=3800:t=q:w=1.4:g=3.5",
                "equalizer=f=8000:t=q:w=1.8:g=2.0",
            ]
        )
    elif profile == "radio":
        # Tight punchy commercial voice
        filters.extend(
            [
                "equalizer=f=150:t=q:w=1.2:g=2.0",
                "equalizer=f=1000:t=q:w=2.0:g=-1.0",
                "equalizer=f=5000:t=q:w=1.2:g=3.0",
            ]
        )
    # "clean" keeps flat frequency curve

    # 4. De-Esser: tame harsh 's', 'sh', 'ch' frequencies
    if de_ess:
        filters.append("equalizer=f=6800:t=q:w=2.2:g=-3.5")

    # 5. Noise gate / expander: suppress ambient background noise when speaker is quiet
    if noise_gate:
        filters.append("agate=threshold=0.015:ratio=2.5:attack=15:release=200:range=0.08")

    # 6. Dynamic broadcast leveling compression (compand)
    filters.append("compand=attacks=0.02:decays=0.25:points=-80/-80|-40/-28|-20/-14|0/-8:gain=0")

    # 7. Final broadcast loudness normalization to target LUFS
    filters.append(f"loudnorm=I={target_lufs}:TP=-1.5:LRA=10")

    return ",".join(filters)


def enhance_audio(
    source: Path | str,
    output: Path | str,
    config: Config,
    *,
    profile: str = DEFAULT_VOICE_PROFILE,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    de_ess: bool = True,
    noise_gate: bool = True,
    force: bool = False,
) -> AudioEnhanceResult:
    """Master and enhance vocal audio in a video or standalone audio file.

    Args:
        source: Input audio or video file.
        output: Destination path for enhanced media.
        config: Project configuration.
        profile: EQ character ('podcast', 'warm', 'crisp', 'clean', 'radio').
        target_lufs: Target integrated loudness (default -14.0 LUFS).
        de_ess: Attenuate harsh sibilance frequencies.
        noise_gate: Gate background room noise during pauses.
        force: Allow overwriting existing output.
    """
    if profile not in VOICE_PROFILES:
        raise ValidationError(
            f"unsupported voice profile: {profile!r}, must be one of {VOICE_PROFILES}"
        )
    check_range(target_lufs, MIN_TARGET_LUFS, MAX_TARGET_LUFS, "target_lufs")

    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    source_info = probe(resolved_source, config)
    if not source_info.has_audio:
        raise ValidationError(f"source has no audio stream to enhance: {resolved_source}")

    filtergraph = _build_enhance_filtergraph(
        profile,
        target_lufs,
        de_ess=de_ess,
        noise_gate=noise_gate,
    )

    args = ["-i", str(resolved_source), "-af", filtergraph]

    if source_info.has_video:
        # Copy video stream bit-for-bit without re-encoding
        args.extend(["-c:v", "copy", "-c:a", "aac", "-b:a", "256k"])
    else:
        # Standalone audio export
        if resolved_output.suffix.lower() == ".wav":
            args.extend(["-c:a", "pcm_s16le"])
        elif resolved_output.suffix.lower() == ".mp3":
            args.extend(["-c:a", "libmp3lame", "-q:a", "2"])
        else:
            args.extend(["-c:a", "aac", "-b:a", "256k"])

    args.append(str(resolved_output))
    run_ffmpeg(args, config)

    final_info = verify_render(
        resolved_output,
        config,
        Expectations(duration_s=source_info.duration_s),
    )

    return AudioEnhanceResult(
        output=resolved_output,
        media=final_info,
        profile=profile,
        target_lufs=target_lufs,
    )
