"""Tests for scale-from-plate estimation."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from media_lab.config import Config
from media_lab.errors import MediaLabError, ValidationError
from media_lab.recipes.scale_plate import estimate_plate_scale
from media_lab.scale_plate import estimate_scale_from_reference, measure_silhouette_height


def test_estimate_scale_from_reference_math() -> None:
    est = estimate_scale_from_reference(
        subject_height=1000,
        ref_height=500,
        ref_ground_y=3200,
        height_ratio=1.0,
    )
    assert est.subject_height_px == 1000
    assert est.target_height_px == 500
    assert est.base_scale == 0.5
    assert est.ground_y == 3200

    est_ratio = estimate_scale_from_reference(
        subject_height=1000,
        ref_height=500,
        ref_ground_y=3200,
        height_ratio=1.1,
    )
    assert est_ratio.target_height_px == 550
    assert est_ratio.base_scale == 0.55


def test_estimate_scale_validation() -> None:
    with pytest.raises(ValidationError, match="subject_height"):
        estimate_scale_from_reference(0, 500, 3200)

    with pytest.raises(ValidationError, match="ref_height"):
        estimate_scale_from_reference(1000, -5, 3200)

    with pytest.raises(ValidationError, match="ref_ground_y"):
        estimate_scale_from_reference(1000, 500, 0)

    with pytest.raises(ValidationError, match="height_ratio"):
        estimate_scale_from_reference(1000, 500, 3200, height_ratio=0.0)


def test_measure_silhouette_height_image(config: Config) -> None:
    config.work_dir.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((200, 100, 4), dtype=np.uint8)
    arr[30:150, 40:60] = [255, 255, 255, 255]
    img_path = config.work_dir / "sil.png"
    Image.fromarray(arr, "RGBA").save(img_path)

    h = measure_silhouette_height(img_path)
    assert h == 119  # y=30 to y=149 inclusive


def test_measure_silhouette_height_empty_fails(config: Config) -> None:
    config.work_dir.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((100, 100, 4), dtype=np.uint8)
    img_path = config.work_dir / "empty.png"
    Image.fromarray(arr, "RGBA").save(img_path)

    with pytest.raises(MediaLabError, match="no opaque subject"):
        measure_silhouette_height(img_path)


def test_estimate_plate_scale_recipe_directory(config: Config) -> None:
    frames_dir = config.work_dir / "test_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        arr = np.zeros((200, 100, 4), dtype=np.uint8)
        arr[50:150, 40:60] = [200, 100, 50, 255]
        Image.fromarray(arr, "RGBA").save(frames_dir / f"f-{i:04d}.png")

    result = estimate_plate_scale(
        frames_dir,
        config,
        ref_height=200,
        ref_ground_y=1800,
        height_ratio=1.0,
    )
    assert result.subject_height_px == 99
    assert result.target_height_px == 200
    assert result.ground_y == 1800
    assert pytest.approx(result.base_scale, 0.01) == 2.02
