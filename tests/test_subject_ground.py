"""Tests for subject grounding: foot detection, shadow rendering, and placement."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from media_lab.config import Config
from media_lab.errors import MediaLabError, PathSafetyError, ValidationError
from media_lab.grounding import (
    FootPoint,
    compute_grounding_transforms,
    detect_foot_point,
    render_contact_shadow,
)
from media_lab.recipes.subject_ground import ground_subject


def test_detect_foot_point_finds_contact() -> None:
    im = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    arr = np.zeros((100, 100, 4), dtype=np.uint8)
    # Silhouette between y=20..80 and x=40..60
    arr[20:81, 40:61] = [255, 255, 255, 255]
    im = Image.fromarray(arr, "RGBA")

    fp = detect_foot_point(im)
    assert fp is not None
    assert fp.fy == 80
    assert fp.fx == 50
    assert fp.height == 60


def test_detect_foot_point_empty_image_returns_none() -> None:
    im = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    assert detect_foot_point(im) is None


def test_render_contact_shadow() -> None:
    canvas_size = (300, 300)
    shadow = render_contact_shadow(canvas_size, 150, 200, opacity=1.0)
    assert shadow.size == canvas_size
    arr = np.asarray(shadow)
    # The shadow alpha should be non-zero around (150, 200)
    assert arr[200, 150, 3] > 0
    # Far corner should be transparent
    assert arr[10, 10, 3] == 0

    zero_shadow = render_contact_shadow(canvas_size, 150, 200, opacity=0.0)
    assert np.asarray(zero_shadow)[..., 3].max() == 0


def test_compute_grounding_transforms() -> None:
    points = [
        FootPoint(fx=50, fy=100, height=80),
        FootPoint(fx=52, fy=102, height=82),
        None,
    ]
    transforms = compute_grounding_transforms(
        points,
        canvas_size=(500, 1000),
        ground_y=800,
        dx=10,
        base_scale=1.0,
        zoom_normalise=True,
    )
    assert len(transforms) == 3
    assert transforms[0].ground_x == 260
    assert transforms[0].ground_y == 800
    assert 0.90 <= transforms[0].scale <= 1.10


def test_ground_subject_directory_to_directory(config: Config) -> None:
    in_dir = config.work_dir / "cutouts"
    in_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        arr = np.zeros((120, 80, 4), dtype=np.uint8)
        arr[20:100, 30:50] = [200, 100, 50, 255]
        Image.fromarray(arr, "RGBA").save(in_dir / f"f-{i:04d}.png")

    out_dir = config.work_dir / "placed"

    res = ground_subject(
        in_dir,
        out_dir,
        config,
        canvas=(200, 400),
        ground_y=350,
        enable_shadow=True,
    )

    assert res.output == out_dir
    assert res.frames == 2
    assert res.canvas == (200, 400)
    assert res.ground_y == 350
    assert (out_dir / "f-0001.png").is_file()

    placed_im = Image.open(out_dir / "f-0001.png")
    assert placed_im.size == (200, 400)


def test_ground_subject_validates_inputs(config: Config) -> None:
    with pytest.raises(ValidationError, match="canvas"):
        ground_subject(config.in_dir, config.out_dir, config, canvas=(0, 100))

    with pytest.raises(ValidationError, match="ground_y"):
        ground_subject(config.in_dir, config.out_dir, config, canvas=(100, 100), ground_y=150)

    with pytest.raises(ValidationError, match="base_scale"):
        ground_subject(
            config.in_dir, config.out_dir, config, canvas=(100, 100), ground_y=50, base_scale=-1.0
        )


def test_ground_subject_guards_existing_output_without_force(config: Config) -> None:
    in_dir = config.work_dir / "cut"
    in_dir.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGBA", (50, 50), (255, 0, 0, 255))
    im.save(in_dir / "f-0000.png")

    out_dir = config.work_dir / "occupied"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stale.txt").write_text("old")

    with pytest.raises(PathSafetyError):
        ground_subject(in_dir, out_dir, config, canvas=(100, 100), ground_y=80, force=False)


def test_ground_subject_fails_on_missing_source(config: Config) -> None:
    with pytest.raises(MediaLabError, match="does not exist"):
        ground_subject(
            config.in_dir / "nonexistent",
            config.out_dir / "placed",
            config,
            canvas=(100, 100),
            ground_y=80,
        )
