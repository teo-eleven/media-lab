"""Tests for Photo & Graphics recipes: typography, inpainting, and face_retouch."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from media_lab.cli import main
from media_lab.config import Config
from media_lab.errors import ValidationError
from media_lab.recipes.face_retouch import (
    apply_depth_blur,
    detect_skin_mask,
    retouch_portrait,
    smooth_skin_layer,
)
from media_lab.recipes.inpainting import (
    create_bbox_mask,
    inpaint_image,
)
from media_lab.recipes.typography import (
    TypographyStyle,
    apply_typography,
    render_typography_overlay,
)


def _generate_synthetic_image(
    path: Path, width: int = 400, height: int = 300, color: tuple[int, int, int] = (180, 180, 180)
) -> Path:
    """Generate simple RGB image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=color)
    img.save(path)
    return path


def _generate_portrait_image(path: Path) -> Path:
    """Generate synthetic portrait with skin-toned center and contrasting background."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[:, :] = [50, 100, 150]  # Blue background (BGR: 150, 100, 50)

    # Center face region with skin tones (BGR: approx [120, 150, 210] -> RGB [210, 150, 120])
    # In BGR:
    arr[70:230, 80:220] = [120, 150, 210]
    # Add simulated blemish / noise
    arr[140:160, 140:160] = [20, 20, 20]

    cv2.imwrite(str(path), arr)
    return path


def _generate_synthetic_video(path: Path, config: Config, duration: float = 2.0) -> Path:
    """Generate short video for typography overlay test."""
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
            f"testsrc=duration={duration}:size=320x240:rate=25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


# ---------------------------------------------------------------------------
# Typography Tests
# ---------------------------------------------------------------------------


def test_render_typography_overlay() -> None:
    style = TypographyStyle(position="bottom", font_size=32, badge=True, shadow=True)
    overlay, lines_count = render_typography_overlay(
        600, 400, "Breaking News\nMedia Lab Live", style
    )
    assert overlay.size == (600, 400)
    assert lines_count == 2
    assert overlay.mode == "RGBA"


def test_apply_typography_image(config: Config) -> None:
    src = _generate_synthetic_image(config.in_dir / "photo.png", width=400, height=300)
    out = config.out_dir / "titled.png"

    res = apply_typography(
        src,
        out,
        "Hello Media Lab",
        config,
        style=TypographyStyle(position="center", badge=True),
    )
    assert res.output.is_file()
    assert res.is_video is False
    assert res.width == 400
    assert res.height == 300
    assert res.lines_rendered >= 1


def test_apply_typography_video(config: Config) -> None:
    src = _generate_synthetic_video(config.in_dir / "clip.mp4", config, duration=1.5)
    out = config.out_dir / "video_titled.mp4"

    res = apply_typography(
        src,
        out,
        "Title on Video",
        config,
        style=TypographyStyle(position="top", badge=True),
        start_s=0.2,
        duration_s=1.0,
        fade_s=0.2,
    )
    assert res.output.is_file()
    assert res.is_video is True
    assert res.width == 320
    assert res.height == 240


def test_typography_validation_errors(config: Config) -> None:
    src = _generate_synthetic_image(config.in_dir / "photo.png")
    out = config.out_dir / "out.png"

    with pytest.raises(ValidationError, match="cannot be empty"):
        apply_typography(src, out, "   ", config)

    with pytest.raises(ValidationError, match="unsupported position"):
        apply_typography(
            src, out, "Valid Text", config, style=TypographyStyle(position="underground")
        )


# ---------------------------------------------------------------------------
# Inpainting Tests
# ---------------------------------------------------------------------------


def test_create_bbox_mask() -> None:
    mask = create_bbox_mask(100, 100, (10, 20, 30, 40))
    assert mask.shape == (100, 100)
    assert np.count_nonzero(mask) == 30 * 40
    assert mask[25, 15] == 255
    assert mask[5, 5] == 0


def test_inpaint_image_bbox(config: Config) -> None:
    src = _generate_portrait_image(config.in_dir / "blemish.png")
    out = config.out_dir / "inpainted_bbox.png"

    # Erase the artificial black square at 140:160, 140:160
    res = inpaint_image(
        src,
        out,
        config,
        bbox=(140, 140, 20, 20),
        method="telea",
        inpaint_radius=4,
    )
    assert res.output.is_file()
    assert res.erased_pixels == 400
    assert res.width == 300
    assert res.height == 300

    # Verify that the black patch was replaced with non-black pixels
    res_bgr = cv2.imread(str(res.output))
    assert res_bgr is not None
    center_color = res_bgr[150, 150]
    assert np.mean(center_color) > 50.0  # Much brighter than the original dark spot [20, 20, 20]


def test_inpaint_image_mask(config: Config) -> None:
    src = _generate_portrait_image(config.in_dir / "blemish2.png")
    mask_file = config.in_dir / "erase_mask.png"
    mask_arr = np.zeros((300, 300), dtype=np.uint8)
    mask_arr[140:160, 140:160] = 255
    Image.fromarray(mask_arr).save(mask_file)

    out = config.out_dir / "inpainted_mask.png"
    res = inpaint_image(src, out, config, mask_path=mask_file, method="ns")
    assert res.output.is_file()
    assert res.erased_pixels == 400


def test_inpaint_validation_errors(config: Config) -> None:
    src = _generate_synthetic_image(config.in_dir / "clean.png")
    out = config.out_dir / "out.png"

    with pytest.raises(ValidationError, match="must provide either bbox or mask_path"):
        inpaint_image(src, out, config)

    with pytest.raises(ValidationError, match="unsupported inpainting method"):
        inpaint_image(src, out, config, bbox=(10, 10, 20, 20), method="magic")


# ---------------------------------------------------------------------------
# Face Retouch Tests
# ---------------------------------------------------------------------------


def test_detect_skin_mask() -> None:
    # Skin tone BGR [120, 150, 210]
    skin_patch = np.full((50, 50, 3), [120, 150, 210], dtype=np.uint8)
    mask = detect_skin_mask(skin_patch)
    assert np.all(mask == 255)

    # Blue background BGR [255, 0, 0]
    blue_patch = np.full((50, 50, 3), [255, 0, 0], dtype=np.uint8)
    mask_blue = detect_skin_mask(blue_patch)
    assert np.all(mask_blue == 0)


def test_smooth_skin_layer() -> None:
    noisy_skin = np.full((60, 60, 3), [120, 150, 210], dtype=np.uint8)
    # Add high frequency noise
    noisy_skin[20:40, 20:40] = [135, 165, 225]
    std_before = float(np.std(noisy_skin))

    smoothed = smooth_skin_layer(noisy_skin, strength=0.8)
    std_after = float(np.std(smoothed))
    assert std_after < std_before  # Variance reduced via bilateral smoothing


def test_apply_depth_blur() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Checkerboard background
    img[::10, :] = 255
    blurred = apply_depth_blur(img, blur_sigma=10.0)
    assert blurred.shape == (100, 100, 3)


def test_retouch_portrait_recipe(config: Config) -> None:
    src = _generate_portrait_image(config.in_dir / "portrait.png")
    out = config.out_dir / "retouched.png"

    res = retouch_portrait(
        src,
        out,
        config,
        smooth_skin=True,
        skin_strength=0.6,
        depth_blur=True,
        blur_sigma=10.0,
        radiance=0.2,
    )
    assert res.output.is_file()
    assert res.skin_smoothed is True
    assert res.depth_blur_applied is True
    assert res.width == 300
    assert res.height == 300


def test_retouch_validation_errors(config: Config) -> None:
    src = _generate_synthetic_image(config.in_dir / "portrait.png")
    out = config.out_dir / "out.png"

    with pytest.raises(ValidationError, match="skin_strength must be between"):
        retouch_portrait(src, out, config, skin_strength=2.5)

    with pytest.raises(ValidationError, match="blur_sigma must be between"):
        retouch_portrait(src, out, config, blur_sigma=-5.0)


# ---------------------------------------------------------------------------
# CLI Integration Tests
# ---------------------------------------------------------------------------


def test_cli_photo_advanced(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("media_lab.cli.load_config", lambda: config)

    src_img = _generate_portrait_image(config.in_dir / "cli_portrait.png")

    # 1. CLI text-overlay
    out_typo = config.out_dir / "cli_typo.png"
    rc = main(
        [
            "text-overlay",
            str(src_img),
            "-o",
            str(out_typo),
            "--text",
            "Badge Title",
            "--position",
            "bottom",
            "--force",
        ]
    )
    assert rc == 0
    assert out_typo.is_file()

    # 2. CLI inpaint
    out_inpaint = config.out_dir / "cli_inpaint.png"
    rc = main(
        ["inpaint", str(src_img), "-o", str(out_inpaint), "--bbox", "140,140,20,20", "--force"]
    )
    assert rc == 0
    assert out_inpaint.is_file()

    # 3. CLI retouch
    out_retouch = config.out_dir / "cli_retouch.png"
    rc = main(
        [
            "retouch",
            str(src_img),
            "-o",
            str(out_retouch),
            "--skin-strength",
            "0.5",
            "--bokeh",
            "8.0",
            "--radiance",
            "0.1",
            "--force",
        ]
    )
    assert rc == 0
    assert out_retouch.is_file()
