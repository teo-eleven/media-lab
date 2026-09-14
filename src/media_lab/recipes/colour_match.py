"""Recipe for colour matching and scene relighting of subject frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..colour_transfer import RelightParams, apply_relight, transfer_colour
from ..config import Config
from ..errors import MediaLabError, PathSafetyError
from ..ffmpeg import run_ffmpeg
from ..paths import ensure_readable_source, ensure_writable_output, work_directory
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

    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = (config.root / source_path).resolve()
    if not source_path.exists():
        raise MediaLabError(f"source path does not exist: {source_path}")

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = (config.root / output_path).resolve()

    bg_img: Image.Image | None = None
    if bg_ref is not None:
        bg_path = Path(bg_ref)
        if not bg_path.is_absolute():
            bg_path = (config.root / bg_path).resolve()
        if not bg_path.exists():
            raise MediaLabError(f"background reference path does not exist: {bg_path}")
        bg_img = Image.open(bg_path).convert("RGB")

    is_source_dir = source_path.is_dir()
    is_output_video = output_path.suffix.lower() in (".mov", ".mp4", ".mkv", ".webm")
    stem = output_path.stem

    if is_source_dir:
        frames_in = source_path
        frame_paths = sorted(frames_in.glob("f-*.png")) or sorted(frames_in.glob("*.png"))
        if not frame_paths:
            raise MediaLabError(f"no PNG frames found in source directory: {source_path}")
        target_fps = fps or STILL_FPS_FALLBACK
        duration_s = len(frame_paths) / target_fps
        has_alpha = True
    else:
        resolved_source = ensure_readable_source(source_path)
        info = probe(resolved_source, config)
        if not info.has_video:
            raise MediaLabError(f"source has no video stream: {resolved_source}")
        target_fps = fps or (info.fps if info.fps > 0 else STILL_FPS_FALLBACK)
        duration_s = info.duration_s
        has_alpha = info.has_alpha

        frames_in = work_directory(config, f"{stem}-colour-src")
        _clear_frames(frames_in)
        run_ffmpeg(["-i", str(resolved_source), str(frames_in / "f-%04d.png")], config)
        frame_paths = sorted(frames_in.glob("f-*.png"))

    if not is_output_video:
        if output_path.exists() and any(output_path.iterdir()) and not force:
            raise PathSafetyError(f"output directory is not empty: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        out_frames_dir = output_path
    else:
        ensure_writable_output(output_path, config, force=force)
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
