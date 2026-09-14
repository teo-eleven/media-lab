"""Subject grounding recipe: place cutout silhouettes with foot-locking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..config import Config
from ..errors import MediaLabError, PathSafetyError, ValidationError
from ..ffmpeg import run_ffmpeg
from ..grounding import compute_grounding_transforms, detect_foot_point, render_contact_shadow
from ..paths import ensure_readable_source, ensure_writable_output, work_directory
from ..probe import MediaInfo, probe
from ..verify import Expectations, verify_render

DEFAULT_CANVAS = (2160, 3840)
DEFAULT_GROUND_Y = 3560
STILL_FPS_FALLBACK = 30.0


@dataclass(frozen=True, slots=True)
class GroundResult:
    """Outcome of placing and grounding subject on canvas."""

    media: MediaInfo | None
    output: Path
    frames: int
    canvas: tuple[int, int]
    ground_y: int


def _clear_frames(directory: Path) -> None:
    for png in directory.glob("f-*.png"):
        png.unlink()


def ground_subject(
    source: Path | str,
    output: Path | str,
    config: Config,
    *,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
    ground_y: int = DEFAULT_GROUND_Y,
    dx: int = 0,
    base_scale: float = 1.0,
    sun_dir: tuple[float, float] = (-0.2, -1.0),
    shadow_opacity: float = 1.0,
    enable_shadow: bool = True,
    zoom_normalise: bool = True,
    fps: float | None = None,
    force: bool = False,
) -> GroundResult:
    """Ground and position subject frames onto canvas with foot-lock and contact shadow.

    Args:
        source: Input directory of PNG frames or video file (.mov, .mp4).
        output: Destination directory or destination video file (.mov).
        config: Project configuration.
        canvas: (width, height) of output canvas.
        ground_y: Vertical pixel line where feet touch the ground.
        dx: Horizontal offset from canvas center.
        base_scale: Base scaling factor for subject.
        sun_dir: (dx, dy) vector for directional shadow falloff.
        shadow_opacity: 0.0 to 1.0 opacity multiplier for contact shadow.
        enable_shadow: Whether to bake the 3-layer contact shadow.
        zoom_normalise: Stabilise frame-to-frame distance deltas via median height.
        fps: Framerate when muxing video, or override.
        force: Allow overwriting destination.
    """
    if canvas[0] <= 0 or canvas[1] <= 0:
        raise ValidationError(f"canvas dimensions must be positive, got {canvas}")
    if ground_y <= 0 or ground_y > canvas[1]:
        raise ValidationError(
            f"ground_y must be between 1 and canvas height {canvas[1]}, got {ground_y}"
        )
    if base_scale <= 0.0:
        raise ValidationError(f"base_scale must be positive, got {base_scale}")

    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = (config.root / source_path).resolve()
    if not source_path.exists():
        raise MediaLabError(f"source path does not exist: {source_path}")

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = (config.root / output_path).resolve()

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
    else:
        resolved_source = ensure_readable_source(source_path)
        info = probe(resolved_source, config)
        if not info.has_video:
            raise MediaLabError(f"source has no video stream: {resolved_source}")
        target_fps = fps or (info.fps if info.fps > 0 else STILL_FPS_FALLBACK)
        duration_s = info.duration_s

        frames_in = work_directory(config, f"{stem}-ground-src")
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
        out_frames_dir = work_directory(config, f"{stem}-ground-frames")
        _clear_frames(out_frames_dir)

    # Pass 1: detect foot contact points for all frames
    foot_points = []
    opened_images: list[Image.Image] = []
    for fp in frame_paths:
        im = Image.open(fp).convert("RGBA")
        opened_images.append(im)
        foot_points.append(detect_foot_point(im))

    # Pass 2: compute transforms (pinning + zoom normalisation)
    transforms = compute_grounding_transforms(
        foot_points,
        canvas_size=canvas,
        ground_y=ground_y,
        dx=dx,
        base_scale=base_scale,
        zoom_normalise=zoom_normalise,
    )

    # Pre-render shadow if enabled
    shadow_img = None
    if enable_shadow and transforms:
        shadow_img = render_contact_shadow(
            canvas,
            transforms[0].ground_x,
            transforms[0].ground_y,
            sun_dir=sun_dir,
            opacity=shadow_opacity,
        )

    # Pass 3: render frames onto canvas
    for i, (im, t) in enumerate(zip(opened_images, transforms, strict=True)):
        orig_w, orig_h = im.size
        new_w = max(1, int(orig_w * t.scale))
        new_h = max(1, int(orig_h * t.scale))
        sub = im.resize((new_w, new_h), Image.Resampling.LANCZOS)

        canvas_img = Image.new("RGBA", canvas, (0, 0, 0, 0))
        if shadow_img is not None:
            canvas_img = Image.alpha_composite(canvas_img, shadow_img)

        canvas_img.alpha_composite(sub, (t.pos_x, t.pos_y))
        dest_file = out_frames_dir / f"f-{i + 1:04d}.png"
        canvas_img.save(dest_file)

    if not is_output_video:
        return GroundResult(
            media=None,
            output=output_path,
            frames=len(frame_paths),
            canvas=canvas,
            ground_y=ground_y,
        )

    # Mux to ProRes 4444 with alpha
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

    media = verify_render(
        output_path,
        config,
        Expectations(
            width=canvas[0],
            height=canvas[1],
            duration_s=duration_s,
            requires_alpha=True,
            requires_audio=False,
        ),
    )
    return GroundResult(
        media=media,
        output=output_path,
        frames=len(frame_paths),
        canvas=canvas,
        ground_y=ground_y,
    )
