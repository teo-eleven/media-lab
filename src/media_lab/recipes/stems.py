"""Separate audio into stems (vocals, no-vocals, etc.) and clean speech using Demucs.

Runs Demucs as a child process through `ml_runner` to keep heavy PyTorch
dependencies outside media_lab's process boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..errors import ValidationError
from ..ffmpeg import run_ffmpeg
from ..ml_runner import MlRunner
from ..paths import ensure_readable_source, ensure_writable_output, work_directory
from ..probe import probe
from ..validation import check_choice
from ..verify import Expectations, verify_render

_DEMUCS_INFER = Path(__file__).resolve().parent.parent / "ml" / "demucs_infer.py"

TWO_STEMS_CHOICES: Sequence[str] = ("vocals", "drums", "bass", "other", "none")
AUDIO_FORMAT_CHOICES: Sequence[str] = ("wav", "mp3", "flac")
DEVICE_CHOICES: Sequence[str] = ("auto", "mps", "cpu", "cuda")


@dataclass(frozen=True, slots=True)
class StemsResult:
    """Outcome of audio stem separation."""

    stems: dict[str, Path]
    output_dir: Path
    cleaned_video: Path | None = None


def separate_stems(
    source: Path | str,
    output_dir: Path | str,
    config: Config,
    ml_runner: MlRunner,
    *,
    two_stems: str = "vocals",
    model: str = "htdemucs",
    audio_format: str = "wav",
    device: str = "auto",
    shifts: int = 0,
    clean_speech: bool = False,
    output_video: Path | str | None = None,
    force: bool = False,
) -> StemsResult:
    """Separate audio tracks into stems and optionally produce a video with clean speech.

    Args:
        source: Path to input audio or video file.
        output_dir: Directory where separated stems will be stored.
        config: Configuration object.
        ml_runner: Subprocess runner for ML scripts.
        two_stems: Separate selected stem and remainder ('vocals', 'drums', etc.).
        model: Pretrained Demucs model (default: 'htdemucs').
        audio_format: Audio format for output stems ('wav', 'mp3', 'flac').
        device: Device to run Demucs on ('auto', 'mps', 'cpu', 'cuda').
        shifts: Random shifts for equivariant stabilization (default: 0).
        clean_speech: If True, muxes isolated vocals back onto source video.
        output_video: Path for the cleaned video if clean_speech is True.
        force: Allow overwriting existing stem files or output video.
    """
    check_choice(two_stems, TWO_STEMS_CHOICES, "two_stems")
    check_choice(audio_format, AUDIO_FORMAT_CHOICES, "audio_format")
    check_choice(device, DEVICE_CHOICES, "device")

    resolved_source = ensure_readable_source(source)
    source_info = probe(resolved_source, config)
    if not source_info.has_audio:
        raise ValidationError(f"source has no audio stream to separate: {resolved_source}")

    resolved_out_dir = Path(output_dir)
    if not resolved_out_dir.is_absolute():
        resolved_out_dir = (config.root / resolved_out_dir).resolve()
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    work = work_directory(config, "stems")
    audio_to_separate = resolved_source
    # If source has video, extract audio to WAV for faster and reliable decoding
    if source_info.has_video:
        extracted_wav = work / f"{resolved_source.stem}_extracted.wav"
        run_ffmpeg(
            [
                "-i",
                str(resolved_source),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                "-ac",
                "2",
                str(extracted_wav),
            ],
            config,
        )
        audio_to_separate = extracted_wav

    args = [
        str(audio_to_separate),
        str(resolved_out_dir),
        "--model",
        model,
        "--two-stems",
        two_stems,
        "--device",
        device,
        "--shifts",
        str(shifts),
        "--format",
        audio_format,
    ]

    ml_runner.run(_DEMUCS_INFER, args)

    stems: dict[str, Path] = {}
    for item in resolved_out_dir.glob(f"*.{audio_format}"):
        stems[item.stem] = item

    cleaned_video_path: Path | None = None
    if clean_speech:
        if not source_info.has_video:
            raise ValidationError(
                f"clean_speech requires a video source, but {resolved_source} has no video"
            )

        vocals_file = stems.get("vocals")
        if not vocals_file or not vocals_file.is_file():
            raise ValidationError(f"vocals stem missing in {resolved_out_dir} for clean_speech")

        target_video = (
            output_video
            if output_video is not None
            else resolved_out_dir / f"{resolved_source.stem}_clean{resolved_source.suffix}"
        )
        resolved_output_video = ensure_writable_output(target_video, config, force=force)

        # Mux clean vocals with original video stream
        run_ffmpeg(
            [
                "-i",
                str(resolved_source),
                "-i",
                str(vocals_file),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-t",
                f"{source_info.duration_s:.3f}",
                str(resolved_output_video),
            ],
            config,
        )
        verify_render(
            resolved_output_video,
            config,
            Expectations(duration_s=source_info.duration_s, requires_audio=True),
        )
        cleaned_video_path = resolved_output_video

    return StemsResult(stems=stems, output_dir=resolved_out_dir, cleaned_video=cleaned_video_path)
