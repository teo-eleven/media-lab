from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from media_lab.config import Config
from media_lab.errors import ValidationError
from media_lab.recipes.brand_kit import (
    BrandResult,
    apply_brand_watermark,
)


@pytest.fixture
def synthetic_video(tmp_path: Path, config: Config) -> Path:
    out = tmp_path / "brand_source.mp4"
    cmd = [
        str(config.ffmpeg),
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=640x360:rate=12",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


@pytest.fixture
def synthetic_logo(tmp_path: Path) -> Path:
    logo_path = tmp_path / "logo.png"
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 200))
    img.save(logo_path)
    return logo_path


def test_apply_brand_watermark_top_right(
    synthetic_video: Path, synthetic_logo: Path, config: Config, tmp_path: Path
) -> None:
    output = tmp_path / "branded_tr.mp4"
    result = apply_brand_watermark(
        synthetic_video,
        synthetic_logo,
        output,
        config,
        position="top-right",
        opacity=0.8,
        scale_pct=15.0,
        force=True,
    )

    assert isinstance(result, BrandResult)
    assert result.output.is_file()
    assert result.position == "top-right"
    assert result.opacity == 0.8
    assert result.media.width == 640
    assert result.media.height == 360


def test_apply_brand_watermark_invalid_position_raises(
    synthetic_video: Path, synthetic_logo: Path, config: Config, tmp_path: Path
) -> None:
    output = tmp_path / "branded_invalid.mp4"
    with pytest.raises(ValidationError, match="Invalid brand position"):
        apply_brand_watermark(
            synthetic_video,
            synthetic_logo,
            output,
            config,
            position="invalid-corner",
        )


def test_apply_brand_watermark_invalid_opacity_raises(
    synthetic_video: Path, synthetic_logo: Path, config: Config, tmp_path: Path
) -> None:
    output = tmp_path / "branded_invalid_op.mp4"
    with pytest.raises(ValidationError, match="opacity"):
        apply_brand_watermark(
            synthetic_video,
            synthetic_logo,
            output,
            config,
            opacity=1.5,
        )
