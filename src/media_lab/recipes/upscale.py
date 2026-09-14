"""Upscale video or PNG frame sequences using Real-ESRGAN.

Runs Real-ESRGAN as a child process through `ml_runner`. Preserves alpha transparency
by upscaling RGB with RRDBNet and alpha with Lanczos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config, require_realesrgan
from ..errors import MediaLabError
from ..ffmpeg import run_ffmpeg
from ..ml_runner import MlRunner
from ..paths import (
    ensure_readable_directory,
    ensure_readable_source,
    ensure_writable_directory,
    ensure_writable_output,
    work_directory,
)
from ..probe import MediaInfo, probe
from ..validation import check_choice
from ..verify import Expectations, verify_render

SCALE_CHOICES = (2, 4)
STILL_FPS_FALLBACK = 30.0

_REALESRGAN_INFER = Path(__file__).resolve().parent.parent / "ml" / "realesrgan_infer.py"


@dataclass(frozen=True, slots=True)
class UpscaleResult:
    """Outcome of an upscale pass."""

    media: MediaInfo | None
    output: Path
    frames: int
    scale: int


def _clear_frames(directory: Path) -> None:
    for png in directory.glob("f-*.png"):
        png.unlink()


def upscale(
    source: Path | str,
    output: Path | str,
    config: Config,
    ml_runner: MlRunner,
    *,
    scale: int = 2,
    tile: int = 512,
    fps: float | None = None,
    force: bool = False,
) -> UpscaleResult:
    """Upscale a video file or a directory of PNG frames with Real-ESRGAN.

    Args:
        source: Path to an input video (.mov, .mp4) or a directory of PNG frames.
        output: Destination video (.mov, .mp4) or destination directory for frames.
        config: Project configuration.
        ml_runner: Subprocess ML runner.
        scale: Upscale factor (2 or 4).
        tile: Tile size (512 recommended for GPU/MPS memory).
        fps: Target framerate when writing video from directory, or override.
        force: Allow overwriting existing output.
    """
    check_choice(scale, SCALE_CHOICES, "scale")
    weight = require_realesrgan(config, scale=scale)

    source_raw = Path(source)
    if source_raw.is_dir():
        source_path = ensure_readable_directory(source_raw)
        is_source_dir = True
    else:
        source_path = ensure_readable_source(source_raw)
        is_source_dir = False

    output_raw = Path(output)
    is_output_video = output_raw.suffix.lower() in (".mov", ".mp4", ".mkv", ".webm")

    if is_source_dir and not is_output_video:
        frames = sorted(source_path.glob("f-*.png"))
        if not frames:
            frames = sorted(source_path.glob("*.png"))
        if not frames:
            raise MediaLabError(f"no PNG frames found in directory: {source_path}")

        output_path = ensure_writable_directory(output_raw, config, force=force)
        ml_runner.run(
            _REALESRGAN_INFER,
            [
                str(source_path),
                str(output_path),
                str(weight),
                "--scale",
                str(scale),
                "--tile",
                str(tile),
            ],
        )
        out_frames = list(output_path.glob("*.png"))
        return UpscaleResult(
            media=None,
            output=output_path,
            frames=len(out_frames),
            scale=scale,
        )

    resolved_output = ensure_writable_output(output_raw, config, force=force)
    stem = resolved_output.stem

    if is_source_dir:
        frames_in = source_path
        frames = sorted(source_path.glob("f-*.png"))
        if not frames:
            frames = sorted(source_path.glob("*.png"))
        if not frames:
            raise MediaLabError(f"no PNG frames found in directory: {source_path}")
        target_fps = fps or STILL_FPS_FALLBACK
        duration_s = len(frames) / target_fps
        has_alpha = True
    else:
        resolved_source = ensure_readable_source(source_path)
        info = probe(resolved_source, config)
        if not info.has_video:
            raise MediaLabError(f"source has no video stream: {resolved_source}")
        target_fps = fps or (info.fps if info.fps > 0 else STILL_FPS_FALLBACK)
        duration_s = info.duration_s
        has_alpha = info.has_alpha

        frames_in = work_directory(config, f"{stem}-source-frames")
        _clear_frames(frames_in)
        run_ffmpeg(["-i", str(resolved_source), str(frames_in / "f-%04d.png")], config)
        frames = list(frames_in.glob("f-*.png"))

    frames_up = work_directory(config, f"{stem}-up-frames")
    _clear_frames(frames_up)

    ml_runner.run(
        _REALESRGAN_INFER,
        [
            str(frames_in),
            str(frames_up),
            str(weight),
            "--scale",
            str(scale),
            "--tile",
            str(tile),
        ],
    )

    is_mov = resolved_output.suffix.lower() == ".mov"
    if is_mov or has_alpha:
        run_ffmpeg(
            [
                "-framerate",
                f"{target_fps:g}",
                "-i",
                str(frames_up / "f-%04d.png"),
                "-c:v",
                "prores_ks",
                "-profile:v",
                "4",
                "-pix_fmt",
                "yuva444p10le",
                "-an",
                str(resolved_output),
            ],
            config,
        )
    else:
        run_ffmpeg(
            [
                "-framerate",
                f"{target_fps:g}",
                "-i",
                str(frames_up / "f-%04d.png"),
                "-c:v",
                "libx264",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(resolved_output),
            ],
            config,
        )

    media = verify_render(
        resolved_output,
        config,
        Expectations(
            duration_s=duration_s,
            requires_alpha=has_alpha or is_mov,
            requires_audio=False,
        ),
    )
    return UpscaleResult(
        media=media,
        output=resolved_output,
        frames=len(frames),
        scale=scale,
    )
