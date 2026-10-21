"""Tests for the Smart Viral Thumbnail Generator."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from media_lab.config import Config
from media_lab.ml_runner import MlRunner
from media_lab.recipes.thumbnail_generator import (
    ThumbnailResult,
    create_glow_outline,
    generate_viral_thumbnail,
)


@pytest.fixture
def synthetic_video(tmp_path: Path, config: Config) -> Path:
    out = tmp_path / "thumb_source.mp4"
    cmd = [
        str(config.ffmpeg),
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=640x360:rate=15",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def test_create_glow_outline(tmp_path: Path) -> None:
    # Create simple RGBA image with a white square in center
    rgba_img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    for x in range(70, 130):
        for y in range(70, 130):
            rgba_img.putpixel((x, y), (255, 255, 255, 255))

    glow_layer = create_glow_outline(rgba_img, color="yellow", stroke_width=8, glow=True)
    assert glow_layer.size == (200, 200)
    assert glow_layer.mode == "RGBA"

    # Check that pixels outside the original square have yellow tint
    has_glow_pixel = False
    for x in range(60, 70):
        px = glow_layer.getpixel((x, 100))
        assert isinstance(px, tuple)
        r, g, b, a = px[:4]
        if a > 0 and r > 200 and g > 150:
            has_glow_pixel = True
            break
    assert has_glow_pixel


def test_generate_viral_thumbnail_execution(
    synthetic_video: Path, config: Config, tmp_path: Path
) -> None:
    ml_runner = MlRunner.from_config(config)
    output = tmp_path / "viral_thumb.jpg"

    res = generate_viral_thumbnail(
        synthetic_video,
        output,
        config,
        ml_runner,
        title="MUST WATCH",
        style="viral",
        stroke_color="yellow",
        stroke_width=6,
        glow=True,
        target_aspect="16:9",
        force=True,
    )

    assert isinstance(res, ThumbnailResult)
    assert res.output.is_file()
    assert res.width == 1280
    assert res.height == 720
    assert res.title == "MUST WATCH"

    with Image.open(res.output) as img:
        assert img.size == (1280, 720)


def test_generate_vertical_thumbnail_execution(
    synthetic_video: Path, config: Config, tmp_path: Path
) -> None:
    ml_runner = MlRunner.from_config(config)
    output = tmp_path / "vertical_thumb.jpg"

    res = generate_viral_thumbnail(
        synthetic_video,
        output,
        config,
        ml_runner,
        title="TIKTOK VIRAL",
        style="viral",
        target_aspect="9:16",
        force=True,
    )

    assert res.output.is_file()
    assert res.width == 1080
    assert res.height == 1920
