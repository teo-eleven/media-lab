"""Tests for colour transfer and scene relighting."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from media_lab.colour_transfer import RelightParams, apply_relight, transfer_colour
from media_lab.config import Config
from media_lab.errors import MediaLabError, PathSafetyError
from media_lab.recipes.colour_match import colour_match


def test_transfer_colour_preserves_alpha_and_shifts_mood() -> None:
    # Red subject on transparent canvas
    src_arr = np.zeros((100, 100, 4), dtype=np.uint8)
    src_arr[20:80, 20:80] = [240, 30, 30, 255]
    src_im = Image.fromarray(src_arr, "RGBA")

    # Cool blue target
    tgt_im = Image.new("RGB", (100, 100), (30, 80, 220))

    transferred = transfer_colour(src_im, tgt_im, strength=1.0)
    res_arr = np.asarray(transferred)

    # Alpha is preserved
    assert np.array_equal(res_arr[..., 3], src_arr[..., 3])
    # Blue channel should increase, red should decrease
    assert res_arr[50, 50, 2] > src_arr[50, 50, 2]
    assert res_arr[50, 50, 0] < src_arr[50, 50, 0]


def test_transfer_colour_zero_strength_is_identity() -> None:
    src_im = Image.new("RGBA", (50, 50), (120, 80, 40, 200))
    tgt_im = Image.new("RGB", (50, 50), (200, 200, 200))
    out = transfer_colour(src_im, tgt_im, strength=0.0)
    assert np.array_equal(np.asarray(out), np.asarray(src_im))


def test_transfer_colour_empty_alpha() -> None:
    src_im = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    tgt_im = Image.new("RGB", (50, 50), (255, 255, 255))
    out = transfer_colour(src_im, tgt_im, strength=1.0)
    assert np.asarray(out)[..., 3].max() == 0


def test_apply_relight_effects() -> None:
    im = Image.new("RGBA", (80, 80), (100, 100, 100, 255))
    params = RelightParams(
        bright=1.2,
        gamma=0.9,
        key_dir=(0.5, 0.0),
        key_amt=0.1,
        bounce=(0.8, 0.5, 0.2),
        bounce_amt=0.2,
        sat=1.1,
        contrast=1.05,
    )
    relit = apply_relight(im, params)
    arr = np.asarray(relit)

    # Alpha preserved
    assert np.all(arr[..., 3] == 255)
    # Brightness should be higher
    assert arr[40, 40, 0] > 100


def test_colour_match_recipe_directory_to_directory(config: Config) -> None:
    config.work_dir.mkdir(parents=True, exist_ok=True)
    in_dir = config.work_dir / "src_cutouts"
    in_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        im = Image.new("RGBA", (60, 60), (150, 100, 50, 255))
        im.save(in_dir / f"f-{i:04d}.png")

    bg_path = config.work_dir / "bg.png"
    Image.new("RGB", (100, 100), (40, 140, 200)).save(bg_path)

    out_dir = config.work_dir / "matched_cutouts"

    res = colour_match(
        in_dir,
        out_dir,
        config,
        bg_ref=bg_path,
        transfer_strength=0.8,
        params=RelightParams(bright=1.05),
    )

    assert res.output == out_dir
    assert res.frames == 2
    assert (out_dir / "f-0000.png").is_file()


def test_colour_match_guards_existing_output(config: Config) -> None:
    config.work_dir.mkdir(parents=True, exist_ok=True)
    in_dir = config.work_dir / "in"
    in_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (10, 10), (1, 1, 1, 1)).save(in_dir / "f-0000.png")

    out_dir = config.work_dir / "out_occupied"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "file.txt").write_text("exists")

    with pytest.raises(PathSafetyError):
        colour_match(in_dir, out_dir, config, force=False)


def test_colour_match_missing_source_fails(config: Config) -> None:
    with pytest.raises(MediaLabError, match="does not exist"):
        colour_match(config.in_dir / "nonexistent", config.out_dir / "out", config)
