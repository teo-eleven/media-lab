"""RVM matting recipe. RVM itself is mocked by default; one opt-in slow test runs it."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from media_lab.config import Config, load_config
from media_lab.errors import MediaLabError, MlEnvError, PathSafetyError, ValidationError
from media_lab.ml_runner import MlResult, MlRunner
from media_lab.recipes.matte_video import V23_ALPHA_GAIN, V23_ALPHA_LIFT, matte_video


def _stub_rvm_env(config: Config) -> None:
    """The minimal RVM checkout + weight files require_ml() looks for."""
    (config.rvm_repo / "model").mkdir(parents=True, exist_ok=True)
    (config.rvm_repo / "model" / "__init__.py").write_text("x\n", encoding="utf-8")
    config.weights_dir.mkdir(parents=True, exist_ok=True)
    for name in ("rvm_resnet50.pth", "rvm_mobilenetv3.pth"):
        (config.weights_dir / name).write_bytes(b"stub")


@pytest.fixture
def matte_config(config: Config) -> Config:
    _stub_rvm_env(config)
    return config


@pytest.fixture
def ml_runner(matte_config: Config) -> MlRunner:
    return MlRunner.from_config(matte_config)


def _fake_child(
    *, opaque: bool = False, score: float = 4.2, captured: dict[str, Any] | None = None
) -> Callable[..., MlResult]:
    """Build a stand-in for MlRunner.run that writes RGBA frames + stats.json."""

    def run(
        self: MlRunner, script: Path, args: Sequence[str] = (), *, timeout_s: int = 0
    ) -> MlResult:
        argv = list(args)
        if captured is not None:
            captured["script"] = Path(script)
            captured["args"] = argv
        frames_in, out_dir = Path(argv[0]), Path(argv[1])
        per_frame = []
        for i, src in enumerate(sorted(frames_in.glob("f-*.png")), 1):
            rgb = Image.open(src).convert("RGB")
            w, h = rgb.size
            alpha = Image.new("L", (w, h), 255)
            if not opaque:
                alpha = Image.new("L", (w, h), 0)
                drift = i % 3
                ImageDraw.Draw(alpha).ellipse(
                    [w // 4 + drift, h // 4, 3 * w // 4 + drift, 3 * h // 4], fill=255
                )
            rgba = rgb.convert("RGBA")
            rgba.putalpha(alpha)
            rgba.save(out_dir / f"f-{i:04d}.png")
            band = np.asarray(alpha, dtype=float)
            per_frame.append(
                {"frame": i, "alpha_mean": float(band.mean()), "alpha_var": float(band.var())}
            )
        (out_dir / "stats.json").write_text(
            json.dumps(
                {"model": argv[2], "device": "cpu", "frames": per_frame,
                 "stability_score": score}
            ),
            encoding="utf-8",
        )
        return MlResult(
            command=(str(self.python), str(script)), stdout="", stderr="", duration_s=0.0
        )

    return run


def test_writes_a_prores_matte_with_alpha_and_a_score(
    silent_video: Path, matte_config: Config, ml_runner: MlRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MlRunner, "run", _fake_child(score=5.5))

    result = matte_video(
        silent_video, matte_config.work_dir / "m.mov", matte_config, ml_runner
    )

    assert (matte_config.work_dir / "m.mov").is_file()
    assert result.media.has_alpha is True
    assert result.media.has_audio is False
    assert result.model == "resnet50"
    assert result.frames > 0
    assert result.alpha_spread >= 1
    assert result.stability_score == pytest.approx(5.5)


def test_passes_the_selected_model_and_weight_to_the_child(
    silent_video: Path, matte_config: Config, ml_runner: MlRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(MlRunner, "run", _fake_child(captured=captured))

    matte_video(
        silent_video, matte_config.work_dir / "m.mov", matte_config, ml_runner,
        model="mobilenetv3",
    )

    assert captured["args"][2] == "mobilenetv3"
    assert captured["args"][3].endswith("rvm_mobilenetv3.pth")
    assert captured["script"].name == "rvm_infer.py"


def test_forwards_alpha_post_processing_knobs(
    silent_video: Path, matte_config: Config, ml_runner: MlRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(MlRunner, "run", _fake_child(captured=captured))

    matte_video(
        silent_video, matte_config.work_dir / "m.mov", matte_config, ml_runner,
        alpha_lift=V23_ALPHA_LIFT, alpha_gain=V23_ALPHA_GAIN,
    )

    assert "--alpha-lift" in captured["args"]
    assert captured["args"][captured["args"].index("--alpha-lift") + 1] == str(V23_ALPHA_LIFT)


def test_rejects_an_unknown_model(
    silent_video: Path, matte_config: Config, ml_runner: MlRunner
) -> None:
    with pytest.raises(ValidationError, match="model must be one of"):
        matte_video(
            silent_video, matte_config.work_dir / "m.mov", matte_config, ml_runner, model="birefnet"
        )


def test_rejects_a_non_mov_output(
    silent_video: Path, matte_config: Config, ml_runner: MlRunner
) -> None:
    with pytest.raises(MediaLabError, match="must be a .mov"):
        matte_video(silent_video, matte_config.out_dir / "m.mp4", matte_config, ml_runner)


def test_reports_a_missing_rvm_environment(
    silent_video: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No _stub_rvm_env: require_ml must fail before any ffmpeg work."""
    monkeypatch.setattr(MlRunner, "run", _fake_child())
    with pytest.raises(MlEnvError, match="RVM checkout not found"):
        matte_video(silent_video, config.work_dir / "m.mov", config, MlRunner.from_config(config))


def test_uniform_alpha_is_rejected(
    silent_video: Path, matte_config: Config, ml_runner: MlRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MlRunner, "run", _fake_child(opaque=True))
    with pytest.raises(MediaLabError, match="uniform"):
        matte_video(silent_video, matte_config.work_dir / "m.mov", matte_config, ml_runner)


def test_existing_output_needs_force(
    silent_video: Path, matte_config: Config, ml_runner: MlRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MlRunner, "run", _fake_child())
    target = matte_config.work_dir / "m.mov"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"old")

    with pytest.raises(PathSafetyError, match="already exists"):
        matte_video(silent_video, target, matte_config, ml_runner)

    result = matte_video(silent_video, target, matte_config, ml_runner, force=True)
    assert result.frames > 0


def test_stale_frames_from_an_earlier_run_are_cleared(
    silent_video: Path, matte_config: Config, ml_runner: MlRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = matte_config.work_dir / "m-rvm"
    stale.mkdir(parents=True, exist_ok=True)
    (stale / "f-9999.png").write_bytes(b"stale")
    monkeypatch.setattr(MlRunner, "run", _fake_child())

    matte_video(silent_video, matte_config.work_dir / "m.mov", matte_config, ml_runner)

    assert not (stale / "f-9999.png").exists()


@pytest.mark.slow
def test_real_rvm_on_a_short_clip(tmp_path: Path) -> None:
    config = load_config()
    if not config.rvm_ready:
        pytest.skip("RVM checkout/weights not set up (run scripts/fetch-rvm.sh)")
    source = Path("in/punto-source.mp4")
    if not source.is_file():
        pytest.skip("in/punto-source.mp4 not present")

    short = config.work_dir / "matte-smoke-src.mp4"
    subprocess.run(
        [str(config.ffmpeg), "-y", "-loglevel", "error", "-i", str(source),
         "-frames:v", "10", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(short)],
        check=True,
    )
    out = config.work_dir / "matte-smoke.mov"
    result = matte_video(
        short, out, config, MlRunner.from_config(config), model="mobilenetv3", force=True
    )

    assert result.frames == 10
    assert result.alpha_spread > 50
    assert result.stability_score >= 0
