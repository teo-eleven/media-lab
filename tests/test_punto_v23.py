"""The punto v23 reproduction runner.

The `--proxy` chain is exercised with the ML subprocess mocked (canned RGBA
frames for both rvm_infer and place_composite); one @slow test runs the real
thing when the RVM checkout + the local source clips are present.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from media_lab.config import Config, load_config
from media_lab.ml_runner import MlResult, MlRunner
from media_lab.recipes import punto_v23
from media_lab.recipes.punto_v23 import run_punto

SMALL_W = 256
SMALL_H = 456
SMALL_FRAMES = 8
FPS = 25


def _ffmpeg(config: Config, args: list[str]) -> None:
    subprocess.run(
        [str(config.ffmpeg), "-y", "-loglevel", "error", *args], check=True, capture_output=True
    )


def _write_rgba(path: Path, w: int, h: int, tint: int) -> None:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = (tint, 80, 200, 255)
    Image.fromarray(arr, "RGBA").save(path)


def _mock_ml(self: MlRunner, script: Path, args: Sequence[str] = (), **kw: object) -> MlResult:
    name = Path(script).name
    if name == "rvm_infer.py":
        frames_in, out_dir = Path(list(args)[0]), Path(list(args)[1])
        n = len(sorted(frames_in.glob("f-*.png")))
        stats = []
        for i in range(1, n + 1):
            _write_rgba(out_dir / f"f-{i:04d}.png", SMALL_W, SMALL_H, tint=200)
            stats.append({"frame": i, "alpha_mean": 90.0, "alpha_var": 100.0})
        (out_dir / "stats.json").write_text(
            json.dumps({"model": "x", "device": "cpu", "frames": stats,
                        "stability_score": 3.3})
        )
    elif name == "place_composite.py":
        root = Path(kw["cwd"])  # type: ignore[arg-type]
        cut = sorted((root / "work/punto-edit/isnet/cut").glob("f-*.png"))
        placed = root / "work/punto-edit/isnet/placed"
        placed.mkdir(parents=True, exist_ok=True)
        for i in range(1, len(cut) + 1):
            _write_rgba(placed / f"f-{i:04d}.png", SMALL_W, SMALL_H, tint=120)
    return MlResult(command=(str(self.python), str(script)), stdout="", stderr="", duration_s=0.0)


def _stub_rvm_env(config: Config) -> None:
    (config.rvm_repo / "model").mkdir(parents=True, exist_ok=True)
    (config.rvm_repo / "model" / "__init__.py").write_text("x\n", encoding="utf-8")
    config.weights_dir.mkdir(parents=True, exist_ok=True)
    for name in ("rvm_resnet50.pth", "rvm_mobilenetv3.pth"):
        (config.weights_dir / name).write_bytes(b"stub")


@pytest.fixture
def punto_scene(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the runner to a renderable size and lay down synthetic inputs."""
    _stub_rvm_env(config)
    monkeypatch.setattr(punto_v23, "CANVAS", (SMALL_W, SMALL_H))
    monkeypatch.setattr(punto_v23, "FPS", FPS)
    monkeypatch.setattr(punto_v23, "FRAMES", SMALL_FRAMES)
    monkeypatch.setattr(punto_v23, "OCCLUSION", {"height": 30, "feather": 10, "y": 400})
    monkeypatch.setattr(MlRunner, "run", _mock_ml)

    (config.in_dir / "punto-source.mp4").parent.mkdir(parents=True, exist_ok=True)
    _ffmpeg(
        config,
        ["-f", "lavfi", "-i", f"testsrc=size={SMALL_W}x{SMALL_H}:rate={FPS}:duration=1",
         "-frames:v", str(SMALL_FRAMES), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(config.in_dir / "punto-source.mp4")],
    )
    (config.in_dir / "backgrounds").mkdir(parents=True, exist_ok=True)
    _ffmpeg(
        config,
        ["-f", "lavfi", "-i", f"color=c=gray:size={SMALL_W}x{SMALL_H}:rate={FPS}:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(config.in_dir / "backgrounds" / "nyc-wallst.mp4")],
    )


def test_proxy_chain_runs_and_renders(config: Config, punto_scene: None) -> None:
    result = run_punto(
        config, MlRunner.from_config(config), config.work_dir / "punto.mp4", proxy=True
    )

    assert result.proxy_mode is True
    assert result.output.is_file()
    assert (result.composed.media.width, result.composed.media.height) == (SMALL_W, SMALL_H)
    assert result.composed.media.duration_s == pytest.approx(SMALL_FRAMES / FPS, abs=0.2)
    assert result.matte.model == "mobilenetv3"
    assert result.matte.stability_score == pytest.approx(3.3)
    assert result.preview.sheet.is_file()
    assert (config.work_dir / "punto-v23.yaml").is_file()


def test_missing_source_is_reported(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MlRunner, "run", _mock_ml)
    with pytest.raises(Exception, match="punto source"):
        run_punto(config, MlRunner.from_config(config), config.work_dir / "punto.mp4", proxy=True)


def test_existing_output_needs_force(config: Config, punto_scene: None) -> None:
    target = config.work_dir / "punto.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"old")
    with pytest.raises(Exception, match="already exists"):
        run_punto(config, MlRunner.from_config(config), target, proxy=True)


@pytest.mark.slow
def test_real_proxy_acceptance(tmp_path: Path) -> None:
    config = load_config()
    if not config.rvm_ready:
        pytest.skip("RVM not set up")
    for rel in (punto_v23.SOURCE, punto_v23.BACKGROUND):
        if not (config.root / rel).is_file():
            pytest.skip(f"missing {rel}")

    result = run_punto(
        config, MlRunner.from_config(config),
        config.work_dir / "punto-acceptance-proxy.mp4", proxy=True, force=True,
    )
    assert (result.composed.media.width, result.composed.media.height) == (2160, 3840)
    assert result.composed.media.duration_s == pytest.approx(6.433, abs=0.1)
    assert result.preview.sheet.is_file()
