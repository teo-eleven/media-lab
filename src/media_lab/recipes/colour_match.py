"""Recipe for colour matching and scene relighting of subject frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..colour_transfer import RelightParams, apply_relight, transfer_colour
from ..config import Config
from ..errors import MediaLabError
from ..ffmpeg import run_ffmpeg
from ..paths import (
    ensure_readable_directory,
    ensure_readable_source,
    ensure_writable_directory,
    ensure_writable_output,
    work_directory,
)
from ..probe import MediaInfo, probe
from ..verify import Expectations, verify_render

STILL_FPS_FALLBACK = 30.0


@dataclass(frozen=True, slots=True)
class ColourMatchResult:
    """Outcome of colour match pass."""

    media: MediaInfo | None
    output: Path
    frames: int


def _clear_frames(directory: Path) -> None:
    for png in directory.glob("f-*.png"):
        png.unlink()


def colour_match(
    source: Path | str,
    output: Path | str,
    config: Config,
    *,
    bg_ref: Path | str | None = None,
    transfer_strength: float = 1.0,
    params: RelightParams | None = None,
    fps: float | None = None,
    force: bool = False,
) -> ColourMatchResult:
    """Match subject colour to background reference and apply scene relighting."""
    relight_settings = params or RelightParams()

    source_raw = Path(source)
    if source_raw.is_dir():
        source_path = ensure_readable_directory(source_raw)
        is_source_dir = True
    else:
        source_path = ensure_readable_source(source_raw)
        is_source_dir = False

    output_raw = Path(output)
    is_output_video = output_raw.suffix.lower() in (".mov", ".mp4", ".mkv", ".webm")
    stem = output_raw.stem

    bg_img: Image.Image | None = None
    if bg_ref is not None:
        bg_path = ensure_readable_source(bg_ref)
        bg_img = Image.open(bg_path).convert("RGB")

    if is_source_dir:
        frames_in = source_path
        frame_paths = sorted(frames_in.glob("f-*.png")) or sorted(frames_in.glob("*.png"))
        if not frame_paths:
            raise MediaLabError(f"no PNG frames found in source directory: {source_path}")
        target_fps = fps or STILL_FPS_FALLBACK
        duration_s = len(frame_paths) / target_fps
        has_alpha = True
    else:
        info = probe(source_path, config)
        if not info.has_video:
            raise MediaLabError(f"source has no video stream: {source_path}")
        target_fps = fps or (info.fps if info.fps > 0 else STILL_FPS_FALLBACK)
        duration_s = info.duration_s
        has_alpha = info.has_alpha

        frames_in = work_directory(config, f"{stem}-colour-src")
        _clear_frames(frames_in)
        run_ffmpeg(["-i", str(source_path), str(frames_in / "f-%04d.png")], config)
        frame_paths = sorted(frames_in.glob("f-*.png"))

    if not is_output_video:
        output_path = ensure_writable_directory(output_raw, config, force=force)
        out_frames_dir = output_path
    else:
        output_path = ensure_writable_output(output_raw, config, force=force)
        out_frames_dir = work_directory(config, f"{stem}-colour-frames")
        _clear_frames(out_frames_dir)

    for i, fp in enumerate(frame_paths):
        im = Image.open(fp).convert("RGBA")
        if bg_img is not None and transfer_strength > 0.0:
            im = transfer_colour(im, bg_img, strength=transfer_strength)
        im = apply_relight(im, relight_settings)

        out_name = fp.name if not is_output_video else f"f-{i + 1:04d}.png"
        im.save(out_frames_dir / out_name)

    if not is_output_video:
        return ColourMatchResult(
            media=None,
            output=output_path,
            frames=len(frame_paths),
        )

    is_mov = output_path.suffix.lower() == ".mov"
    if is_mov or has_alpha:
        run_ffmpeg(
            [
                "-framerate",
                f"{target_fps:g}",
                "-i",
                str(out_frames_dir / "f-%04d.png"),
                "-c:v",
                "prores_ks",
                "-profile:v",
                "4",
                "-pix_fmt",
                "yuva444p10le",
                "-an",
                str(output_path),
            ],
            config,
        )
    else:
        run_ffmpeg(
            [
                "-framerate",
                f"{target_fps:g}",
                "-i",
                str(out_frames_dir / "f-%04d.png"),
                "-c:v",
                "libx264",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(output_path),
            ],
            config,
        )

    media = verify_render(
        output_path,
        config,
        Expectations(
            duration_s=duration_s,
            requires_alpha=has_alpha or is_mov,
            requires_audio=False,
        ),
    )
    return ColourMatchResult(
        media=media,
        output=output_path,
        frames=len(frame_paths),
    )
