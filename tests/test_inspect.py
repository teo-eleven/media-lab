"""Tests for deep media inspection module and CLI."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from media_lab.cli import main
from media_lab.config import Config
from media_lab.errors import MediaLabError
from media_lab.inspect import inspect_media


def _generate_synthetic_video(path: Path, config: Config, duration: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(config.ffmpeg),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size=320x240:rate=24",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )
    return path


def _generate_synthetic_image(path: Path, width: int = 200, height: int = 200) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGBA", (width, height), (50, 100, 150, 255))
    im.save(path)
    return path


def test_inspect_video_happy_path(config: Config) -> None:
    video_path = _generate_synthetic_video(config.in_dir / "sample.mp4", config)
    report = inspect_media(video_path, config)

    assert report.media_type == "video"
    assert report.width == 320
    assert report.height == 240
    assert report.aspect_ratio == "320:240"
    assert report.duration_s >= 0.9
    assert report.fps == 24.0
    assert report.audio is not None
    assert report.audio.has_speech is True

    # Test serialization
    as_dict = report.to_dict()
    assert as_dict["media_type"] == "video"
    as_json = report.to_json()
    parsed = json.loads(as_json)
    assert parsed["width"] == 320


def test_inspect_image_happy_path(config: Config) -> None:
    img_path = _generate_synthetic_image(config.in_dir / "sample.png", 200, 200)
    report = inspect_media(img_path, config)

    assert report.media_type == "image"
    assert report.width == 200
    assert report.height == 200
    assert report.aspect_ratio == "1:1"
    assert report.visual is not None
    assert report.visual.has_alpha is True
    assert len(report.visual.dominant_colors) > 0


def test_inspect_missing_file_fails(config: Config) -> None:
    with pytest.raises(MediaLabError, match="does not exist"):
        inspect_media(config.in_dir / "nonexistent.mp4", config)


def test_cli_inspect(
    config: Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("media_lab.cli.load_config", lambda: config)
    img_path = _generate_synthetic_image(config.in_dir / "cli_img.png", 100, 100)

    # Standard human-readable output
    rc = main(["inspect", str(img_path)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "media inspection:" in captured
    assert "dimensions:" in captured

    # Machine-readable JSON output
    rc = main(["inspect", str(img_path), "--json"])
    assert rc == 0
    captured_json = capsys.readouterr().out
    data = json.loads(captured_json)
    assert data["width"] == 100
    assert data["media_type"] == "image"
