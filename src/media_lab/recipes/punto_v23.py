"""Reproduce the punto v23 render through the three Phase-1 tools.

matte-video -> (Real-ESRGAN upscale) -> place_composite -> compose-spec, then a
contact sheet against the reference render. The two middle steps are the
verbatim v23 scripts in `docs/video-agent/pipeline/`; they hardcode
`work/punto-edit/...` paths that do not chain as-is, so this runner stages
frames between them (audit M2/M3). `--proxy` skips the ~24-minute upscale.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import Config
from ..errors import MediaLabError
from ..ml_runner import MlRunner
from ..paths import ensure_writable_output
from .compose_spec import ComposeResult, compose
from .matte_video import V23_ALPHA_GAIN, V23_ALPHA_LIFT, MatteResult, matte_video
from .proxy_preview import ProxyResult, proxy_preview

SOURCE = "in/punto-source.mp4"
BACKGROUND = "in/backgrounds/nyc-wallst.mp4"
REFERENCE = "out/punto-final_2160x3840_30fps_h264-crf17.mp4"
BG_START_S = 1.0
CANVAS = (2160, 3840)
FPS = 30
FRAMES = 193
OCCLUSION = {"height": 130, "feather": 70, "y": 3710}

_PIPELINE = Path("docs") / "video-agent" / "pipeline"

# The exact dirs the legacy scripts read/write, relative to the repo root.
_CUT_DIR = "work/punto-edit/isnet/cut"
_UP_DIR = "work/punto-edit/isnet/up"
_RVM_UP_DIR = "work/punto-edit/rvm_up"
_PLACED_DIR = "work/punto-edit/isnet/placed"


@dataclass(frozen=True, slots=True)
class PuntoResult:
    output: Path
    proxy_mode: bool
    matte: MatteResult
    composed: ComposeResult
    preview: ProxyResult


def _stage_frames(src: Path, dst: Path) -> int:
    """Replace dst/f-*.png with hard links (or copies) of src/f-*.png."""
    dst.mkdir(parents=True, exist_ok=True)
    for stale in dst.glob("f-*.png"):
        stale.unlink()
    frames = sorted(src.glob("f-*.png"))
    if not frames:
        raise MediaLabError(f"no f-*.png frames to stage from {src}")
    for frame in frames:
        target = dst / frame.name
        try:
            target.hardlink_to(frame)
        except OSError:
            shutil.copy2(frame, target)
    return len(frames)


def run_punto(
    config: Config,
    ml_runner: MlRunner,
    output: Path | str,
    *,
    proxy: bool = False,
    force: bool = False,
) -> PuntoResult:
    """Rebuild the punto v23 clip. Raises MediaLabError / MlEnvError /
    SpecError / VerificationError."""
    root = config.root
    resolved_output = ensure_writable_output(output, config, force=force)
    if not (root / SOURCE).is_file():
        raise MediaLabError(f"missing {SOURCE} (the punto source is local and gitignored)")

    model = "mobilenetv3" if proxy else "resnet50"
    matte_mov = config.work_dir / "punto-matte.mov"
    matte = matte_video(
        root / SOURCE, matte_mov, config, ml_runner,
        model=model, alpha_lift=V23_ALPHA_LIFT, alpha_gain=V23_ALPHA_GAIN, force=True,
    )
    # matte_video leaves its RGBA frames in work/punto-matte-rvm/
    _stage_frames(config.work_dir / "punto-matte-rvm", root / _CUT_DIR)

    if proxy:
        # place_composite falls back to the 720x1280 cut frames -> ~half-scale
        # subject. Acceptable for a rough preview; noted in the result.
        if (root / _RVM_UP_DIR).is_dir():
            shutil.rmtree(root / _RVM_UP_DIR)
    else:
        ml_runner.run(root / _PIPELINE / "upscale_realesrgan.py", cwd=root)
        _stage_frames(root / _UP_DIR, root / _RVM_UP_DIR)

    ml_runner.run(root / _PIPELINE / "place_composite.py", cwd=root)

    spec_path = config.work_dir / "punto-v23.yaml"
    spec_path.write_text(yaml.safe_dump(_spec_dict()), encoding="utf-8")
    composed = compose(spec_path, resolved_output, config, force=force)

    reference = root / REFERENCE
    preview = proxy_preview(
        resolved_output, config,
        compare=reference if reference.is_file() else None, force=force,
    )
    return PuntoResult(
        output=resolved_output, proxy_mode=proxy,
        matte=matte, composed=composed, preview=preview,
    )


def _spec_dict() -> dict[str, object]:
    return {
        "background": {
            "path": BACKGROUND, "width": CANVAS[0], "height": CANVAS[1],
            "fps": FPS, "frames": FRAMES, "start_s": BG_START_S,
        },
        "subject": {"frames": f"{_PLACED_DIR}/f-%04d.png", "fps": FPS},
        "occlusion": dict(OCCLUSION),
        "grade": {"profile": "v23"},
        "output": {"crf": 18, "preset": "medium"},
    }
