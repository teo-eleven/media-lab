"""Reproduce the punto v23 render through the native Phase-1 & Phase-2 tools.

Pipeline: matte-video -> (Real-ESRGAN upscale) -> colour-match -> subject-ground -> compose-spec,
followed by a proxy preview against the reference render.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..colour_transfer import RelightParams
from ..config import Config
from ..errors import MediaLabError
from ..ml_runner import MlRunner
from ..paths import ensure_writable_output
from .colour_match import colour_match
from .compose_spec import ComposeResult, compose
from .matte_video import V23_ALPHA_GAIN, V23_ALPHA_LIFT, MatteResult, matte_video
from .proxy_preview import ProxyResult, proxy_preview
from .subject_ground import ground_subject
from .upscale import upscale

SOURCE = "in/punto-source.mp4"
BACKGROUND = "in/backgrounds/nyc-wallst.mp4"
REFERENCE = "out/punto-final_2160x3840_30fps_h264-crf17.mp4"
BG_START_S = 1.0
CANVAS = (2160, 3840)
FPS = 30
FRAMES = 193
OCCLUSION = {"height": 130, "feather": 70, "y": 3710}

# NYC Wall St scene relighting profile from v23
PUNTO_RELIGHT = RelightParams(
    ambient=(0.99, 1.00, 1.01),
    bright=0.96,
    gamma=1.03,
    key_dir=(-0.2, -1.0),
    key_amt=0.04,
    bounce=(0.55, 0.56, 0.57),
    bounce_amt=0.09,
    sat=0.82,
    contrast=0.93,
    blur=0.55,
)
PUNTO_GROUND_Y = 3560
PUNTO_DX = -180
PUNTO_BASE_SCALE = 0.98
PUNTO_SUN_DIR = (0.03, 1.0)
PUNTO_SHADOW_OPACITY = 135.0 / 255.0


@dataclass(frozen=True, slots=True)
class PuntoResult:
    output: Path
    proxy_mode: bool
    matte: MatteResult
    composed: ComposeResult
    preview: ProxyResult


def run_punto(
    config: Config,
    ml_runner: MlRunner,
    output: Path | str,
    *,
    proxy: bool = False,
    force: bool = False,
) -> PuntoResult:
    """Rebuild the punto v23 clip using the native modular recipes.

    Raises:
        MediaLabError, MlEnvError, SpecError, VerificationError.
    """
    root = config.root
    resolved_output = ensure_writable_output(output, config, force=force)
    if not (root / SOURCE).is_file():
        raise MediaLabError(f"missing {SOURCE} (the punto source is local and gitignored)")

    # 1. Matting with RVM
    model = "mobilenetv3" if proxy else "resnet50"
    matte_mov = config.work_dir / "punto-matte.mov"
    matte = matte_video(
        root / SOURCE,
        matte_mov,
        config,
        ml_runner,
        model=model,
        alpha_lift=V23_ALPHA_LIFT,
        alpha_gain=V23_ALPHA_GAIN,
        force=True,
    )
    matte_frames_dir = config.work_dir / "punto-matte-rvm"

    # 2. Upscale (skipped in proxy mode)
    if proxy:
        frames_to_relight = matte_frames_dir
        scale_factor = PUNTO_BASE_SCALE * 0.5
    else:
        up_dir = config.work_dir / "punto-upscaled"
        upscale(matte_frames_dir, up_dir, config, ml_runner, scale=2, force=True)
        frames_to_relight = up_dir
        scale_factor = PUNTO_BASE_SCALE

    # 3. Scene colour matching and relighting
    relit_dir = config.work_dir / "punto-relit"
    colour_match(frames_to_relight, relit_dir, config, params=PUNTO_RELIGHT, force=True)

    # 4. Grounding and placement onto canvas
    placed_dir = config.work_dir / "punto-edit/isnet/placed"
    effective_ground_y = PUNTO_GROUND_Y if CANVAS[1] >= PUNTO_GROUND_Y else int(CANVAS[1] * 0.9)
    ground_subject(
        relit_dir,
        placed_dir,
        config,
        canvas=CANVAS,
        ground_y=effective_ground_y,
        dx=PUNTO_DX,
        base_scale=scale_factor,
        sun_dir=PUNTO_SUN_DIR,
        shadow_opacity=PUNTO_SHADOW_OPACITY,
        zoom_normalise=True,
        force=True,
    )

    # 5. Composite according to spec
    spec_path = config.work_dir / "punto-v23.yaml"
    spec_path.write_text(yaml.safe_dump(_spec_dict(config)), encoding="utf-8")
    composed = compose(spec_path, resolved_output, config, force=force)

    # 6. Proxy preview and comparison sheet
    reference = root / REFERENCE
    preview = proxy_preview(
        resolved_output,
        config,
        compare=reference if reference.is_file() else None,
        force=force,
    )
    return PuntoResult(
        output=resolved_output,
        proxy_mode=proxy,
        matte=matte,
        composed=composed,
        preview=preview,
    )


def _spec_dict(config: Config) -> dict[str, object]:
    placed_dir = config.work_dir / "punto-edit/isnet/placed"
    return {
        "background": {
            "path": BACKGROUND,
            "width": CANVAS[0],
            "height": CANVAS[1],
            "fps": FPS,
            "frames": FRAMES,
            "start_s": BG_START_S,
        },
        "subject": {"frames": str(placed_dir / "f-%04d.png"), "fps": FPS},
        "occlusion": dict(OCCLUSION),
        "grade": {"profile": "v23"},
        "output": {"crf": 18, "preset": "medium"},
    }
