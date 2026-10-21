"""The compose-spec recipe: YAML -> one ffmpeg pass -> verified render."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from media_lab.config import Config
from media_lab.errors import PathSafetyError, SpecError
from media_lab.recipes.compose_spec import compose

CANVAS_W = 256
CANVAS_H = 456
FRAMES = 15
FPS = 30


def _ffmpeg(config: Config, args: list[str]) -> None:
    subprocess.run(
        [str(config.ffmpeg), "-y", "-loglevel", "error", *args], check=True, capture_output=True
    )


@pytest.fixture
def scene(config: Config) -> dict[str, str]:
    """A tiny background clip + a matching pre-placed RGBA frame sequence."""
    bg = config.in_dir / "bg.mp4"
    _ffmpeg(
        config,
        [
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={CANVAS_W}x{CANVAS_H}:rate={FPS}:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(bg),
        ],
    )
    placed = config.in_dir / "placed"
    placed.mkdir()
    _ffmpeg(
        config,
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=red:size={CANVAS_W}x{CANVAS_H}:rate={FPS}:duration={FRAMES / FPS}",
            "-vf",
            "format=rgba,colorchannelmixer=aa=0.5",
            "-frames:v",
            str(FRAMES),
            str(placed / "f-%04d.png"),
        ],
    )
    return {"bg": "in/bg.mp4", "frames": "in/placed/f-%04d.png"}


def _spec_file(config: Config, scene: dict[str, str], **grade: object) -> Path:
    raw: dict[str, object] = {
        "background": {
            "path": scene["bg"],
            "width": CANVAS_W,
            "height": CANVAS_H,
            "fps": FPS,
            "frames": FRAMES,
            "start_s": 0.0,
        },
        "subject": {"frames": scene["frames"], "fps": FPS},
        "occlusion": {"height": 30, "feather": 10, "y": 400},
        "output": {"crf": 28, "preset": "ultrafast"},
    }
    if grade:
        raw["grade"] = grade
    path = config.in_dir / "spec.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_renders_the_spec_and_keeps_the_filtergraph(config: Config, scene: dict[str, str]) -> None:
    result = compose(_spec_file(config, scene), config.out_dir / "shot.mp4", config)

    assert (result.media.width, result.media.height) == (CANVAS_W, CANVAS_H)
    assert result.media.duration_s == pytest.approx(FRAMES / FPS, abs=0.2)
    assert result.media.has_audio is False
    assert result.filtergraph_path.is_file()
    assert result.filtergraph_path.read_text(encoding="utf-8").strip().endswith("[out]")


def test_grade_none_still_renders(config: Config, scene: dict[str, str]) -> None:
    result = compose(_spec_file(config, scene, profile="none"), config.out_dir / "shot.mp4", config)
    assert (result.media.width, result.media.height) == (CANVAS_W, CANVAS_H)


def test_rejects_a_subject_frame_that_is_not_canvas_size(
    config: Config, scene: dict[str, str]
) -> None:
    wrong = config.in_dir / "wrong"
    wrong.mkdir()
    _ffmpeg(
        config,
        [
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:size=100x100:rate={FPS}:duration=0.5",
            "-vf",
            "format=rgba",
            "-frames:v",
            str(FRAMES),
            str(wrong / "f-%04d.png"),
        ],
    )
    raw = yaml.safe_load(_spec_file(config, scene).read_text())
    raw["subject"]["frames"] = "in/wrong/f-%04d.png"
    spec = config.in_dir / "spec.yaml"
    spec.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(SpecError, match="compose-spec overlays pre-placed full-canvas frames"):
        compose(spec, config.out_dir / "shot.mp4", config)


def test_existing_output_needs_force(config: Config, scene: dict[str, str]) -> None:
    target = config.out_dir / "shot.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"old")
    spec = _spec_file(config, scene)

    with pytest.raises(PathSafetyError, match="already exists"):
        compose(spec, target, config)

    result = compose(spec, target, config, force=True)
    assert result.media.width == CANVAS_W


def test_reports_a_missing_background(config: Config, scene: dict[str, str]) -> None:
    raw = yaml.safe_load(_spec_file(config, scene).read_text())
    raw["background"]["path"] = "in/absent.mp4"
    spec = config.in_dir / "spec.yaml"
    spec.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(PathSafetyError, match="does not exist"):
        compose(spec, config.out_dir / "shot.mp4", config)


def test_reports_missing_subject_frames(config: Config, scene: dict[str, str]) -> None:
    raw = yaml.safe_load(_spec_file(config, scene).read_text())
    raw["subject"]["frames"] = "in/nope/f-%04d.png"
    spec = config.in_dir / "spec.yaml"
    spec.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(PathSafetyError, match="does not exist"):
        compose(spec, config.out_dir / "shot.mp4", config)
