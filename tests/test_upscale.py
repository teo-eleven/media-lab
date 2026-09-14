"""Tests for Real-ESRGAN upscale recipe and CLI."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from media_lab.config import Config
from media_lab.errors import MediaLabError, MlEnvError, PathSafetyError, ValidationError
from media_lab.ml_runner import MlResult, MlRunner
from media_lab.recipes.upscale import upscale


def _stub_realesrgan_weights(config: Config) -> None:
    config.weights_dir.mkdir(parents=True, exist_ok=True)
    for name in ("RealESRGAN_x2plus.pth", "RealESRGAN_x4plus.pth"):
        (config.weights_dir / name).write_bytes(b"stub")


@pytest.fixture
def upscale_config(config: Config) -> Config:
    _stub_realesrgan_weights(config)
    return config


@pytest.fixture
def ml_runner(upscale_config: Config) -> MlRunner:
    return MlRunner.from_config(upscale_config)


def _fake_upscale_child(
    captured: dict[str, Any] | None = None,
) -> Callable[..., MlResult]:
    def run(
        self: MlRunner,
        script: Path | str,
        args: Sequence[str] = (),
        *,
        timeout_s: int = 0,
        cwd: Path | None = None,
    ) -> MlResult:
        argv = list(args)
        if captured is not None:
            captured["script"] = Path(script)
            captured["args"] = argv

        frames_in, frames_out = Path(argv[0]), Path(argv[1])
        frames_out.mkdir(parents=True, exist_ok=True)

        scale = 2
        if "--scale" in argv:
            scale = int(argv[argv.index("--scale") + 1])

        found = sorted(frames_in.glob("f-*.png")) or sorted(frames_in.glob("*.png"))
        for src in found:
            im = Image.open(src).convert("RGBA")
            w, h = im.size
            up = im.resize((w * scale, h * scale), Image.Resampling.NEAREST)
            up.save(frames_out / src.name)

        return MlResult(
            command=tuple([str(script), *argv]),
            stdout="DONE",
            stderr="",
            duration_s=0.1,
        )

    return run


def test_upscale_video_happy_path(
    upscale_config: Config,
    ml_runner: MlRunner,
    silent_video: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = silent_video
    out = upscale_config.out_dir / "upscaled.mp4"

    captured: dict[str, Any] = {}
    monkeypatch.setattr(MlRunner, "run", _fake_upscale_child(captured))

    res = upscale(source, out, upscale_config, ml_runner, scale=2)

    assert res.output == out
    assert res.scale == 2
    assert res.frames > 0
    assert res.media is not None
    assert res.media.width == 1280
    assert res.media.height == 720
    assert "--scale" in captured["args"]
    idx = captured["args"].index("--scale")
    assert captured["args"][idx + 1] == "2"


def test_upscale_directory_to_directory(
    upscale_config: Config,
    ml_runner: MlRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    in_dir = upscale_config.work_dir / "raw_frames"
    in_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        im = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        im.save(in_dir / f"f-{i:04d}.png")

    out_dir = upscale_config.work_dir / "up_frames"

    monkeypatch.setattr(MlRunner, "run", _fake_upscale_child())

    res = upscale(in_dir, out_dir, upscale_config, ml_runner, scale=2)

    assert res.output == out_dir
    assert res.frames == 3
    assert res.media is None
    assert (out_dir / "f-0000.png").is_file()
    up_im = Image.open(out_dir / "f-0000.png")
    assert up_im.size == (64, 64)


def test_upscale_rejects_invalid_scale(
    upscale_config: Config,
    ml_runner: MlRunner,
    silent_video: Path,
) -> None:
    with pytest.raises(ValidationError, match="scale"):
        upscale(
            silent_video,
            upscale_config.out_dir / "out.mp4",
            upscale_config,
            ml_runner,
            scale=3,
        )


def test_upscale_guards_existing_output_without_force(
    upscale_config: Config,
    ml_runner: MlRunner,
    silent_video: Path,
) -> None:
    upscale_config.out_dir.mkdir(parents=True, exist_ok=True)
    out = upscale_config.out_dir / "exists.mp4"
    out.write_bytes(b"dummy")

    with pytest.raises(PathSafetyError):
        upscale(silent_video, out, upscale_config, ml_runner, scale=2, force=False)


def test_upscale_fails_when_weights_missing(
    config: Config,
    silent_video: Path,
) -> None:
    runner = MlRunner.from_config(config)
    with pytest.raises(MlEnvError):
        upscale(silent_video, config.out_dir / "out.mp4", config, runner, scale=2)




def test_upscale_fails_on_missing_source(
    upscale_config: Config,
    ml_runner: MlRunner,
) -> None:
    with pytest.raises(MediaLabError, match="does not exist"):
        upscale(
            upscale_config.in_dir / "nonexistent.mp4",
            upscale_config.out_dir / "out.mp4",
            upscale_config,
            ml_runner,
        )


def test_upscale_fails_on_empty_directory(
    upscale_config: Config,
    ml_runner: MlRunner,
) -> None:
    empty_dir = upscale_config.work_dir / "empty"
    empty_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(MediaLabError, match="no PNG frames found"):
        upscale(
            empty_dir,
            upscale_config.out_dir / "out.mp4",
            upscale_config,
            ml_runner,
        )
