"""Matte a person out of a video with RVM (Robust Video Matting).

RVM runs as a child process through `ml_runner` - GPL-3 code and `torch`
never enter this process. Output is a ProRes 4444 `.mov` with alpha, plus an
alpha-stability score: the mean frame-to-frame alpha change over the
foreground, on a 0-255 scale, lower = less flicker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import RVM_WEIGHT_FILES, Config, require_ml
from ..errors import MediaLabError, VerificationError
from ..ffmpeg import measure_alpha_spread, run_ffmpeg
from ..ml_runner import MlRunner
from ..paths import ensure_readable_source, ensure_writable_output, work_directory
from ..probe import MediaInfo, probe
from ..validation import check_choice
from ..verify import Expectations, verify_render

MODEL_CHOICES = ("resnet50", "mobilenetv3")
OUTPUT_SUFFIX = ".mov"
# A matte whose alpha never varies means RVM separated nothing.
MIN_ALPHA_SPREAD = 1
STILL_FPS_FALLBACK = 30.0
# v23's matte_rvm.py stretched alpha contrast to cut edge spill on the punto
# clip's park background. Not a sensible default for arbitrary footage, so the
# recipe defaults to identity; the punto runner passes these explicitly.
V23_ALPHA_LIFT = 18.0
V23_ALPHA_GAIN = 1.28

_RVM_INFER = Path(__file__).resolve().parent.parent / "ml" / "rvm_infer.py"


@dataclass(frozen=True, slots=True)
class MatteResult:
    """A finished matte plus what RVM reported about it."""

    media: MediaInfo
    model: str
    frames: int
    alpha_spread: int
    stability_score: float


def _clear_frames(directory: Path) -> None:
    """A shorter earlier run must not leave stale trailing frames behind."""
    for png in directory.glob("f-*.png"):
        png.unlink()


def matte_video(
    source: Path | str,
    output: Path | str,
    config: Config,
    ml_runner: MlRunner,
    *,
    model: str = "resnet50",
    alpha_lift: float = 0.0,
    alpha_gain: float = 1.0,
    force: bool = False,
) -> MatteResult:
    """Run RVM over `source` and write a ProRes 4444 `.mov` cutout to `output`.

    Args:
        source: Input clip. Never modified.
        output: Destination `.mov`. Refused if it exists unless `force`.
        config: Resolved project configuration.
        ml_runner: The channel to the ML child process.
        model: `resnet50` (default, sharper) or `mobilenetv3` (faster).
        alpha_lift / alpha_gain: post-process the matte as
            `clip((alpha - lift) * gain, 0, 255)`. Defaults are identity;
            pass `V23_ALPHA_LIFT` / `V23_ALPHA_GAIN` to match the punto v23 run.
        force: Overwrite an existing `output`.

    Raises:
        ValidationError: `model` is not one of MODEL_CHOICES.
        MlEnvError: the RVM checkout/weights are missing, or the child failed.
        PathSafetyError: `source` or `output` violate the path contract.
        MediaLabError: `output` is not a `.mov`, or `source` has no video.
        VerificationError: the render does not match, or the matte is uniform.
    """
    check_choice(model, MODEL_CHOICES, "model")
    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)
    if resolved_output.suffix.lower() != OUTPUT_SUFFIX:
        raise MediaLabError(
            f"matte output must be a {OUTPUT_SUFFIX} file, got {resolved_output.suffix!r}"
        )
    require_ml(config, model=model)

    info = probe(resolved_source, config)
    if not info.has_video:
        raise MediaLabError(f"source has no video stream: {resolved_source}")
    fps = info.fps if info.fps > 0 else STILL_FPS_FALLBACK

    stem = resolved_output.stem
    frames_in = work_directory(config, f"{stem}-frames")
    rvm_out = work_directory(config, f"{stem}-rvm")
    _clear_frames(frames_in)
    _clear_frames(rvm_out)

    run_ffmpeg(["-i", str(resolved_source), str(frames_in / "f-%04d.png")], config)

    weight = config.weights_dir / RVM_WEIGHT_FILES[model]
    ml_runner.run(
        _RVM_INFER,
        [
            str(frames_in), str(rvm_out), model, str(weight), str(config.rvm_repo),
            "--alpha-lift", str(alpha_lift), "--alpha-gain", str(alpha_gain),
        ],
    )

    run_ffmpeg(
        [
            "-framerate", f"{fps:g}", "-i", str(rvm_out / "f-%04d.png"),
            "-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le",
            "-an", str(resolved_output),
        ],
        config,
    )

    media = verify_render(
        resolved_output,
        config,
        Expectations(duration_s=info.duration_s, requires_alpha=True, requires_audio=False),
    )
    alpha_spread = measure_alpha_spread(resolved_output, config)
    if alpha_spread < MIN_ALPHA_SPREAD:
        raise VerificationError(
            str(resolved_output),
            ("alpha channel is uniform: RVM separated nothing from the background",),
        )

    stats = json.loads((rvm_out / "stats.json").read_text(encoding="utf-8"))
    return MatteResult(
        media=media,
        model=model,
        frames=len(stats.get("frames", [])),
        alpha_spread=alpha_spread,
        stability_score=float(stats.get("stability_score", 0.0)),
    )
