"""Tests for photo recipe, framing, background replacement, looks, and CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from media_lab.cli import main
from media_lab.config import Config
from media_lab.errors import MediaLabError, PathSafetyError, ValidationError
from media_lab.recipes.photo import edit_photo, process_photo_batch


def _create_synthetic_image(
    path: Path,
    width: int = 400,
    height: int = 300,
    mode: str = "RGB",
    color: tuple[int, ...] = (100, 150, 200),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "RGBA":
        im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        # Add an opaque rectangle in the center (representing a subject)
        subj = Image.new("RGBA", (width // 2, height // 2), color + (255,))
        im.paste(subj, (width // 4, height // 4))
    else:
        im = Image.new("RGB", (width, height), color)
    im.save(path)
    return path


def test_photo_framing_aspects(config: Config) -> None:
    src = _create_synthetic_image(config.in_dir / "orig.jpg", 800, 600)

    # 1:1
    out_1x1 = config.out_dir / "out_1x1.jpg"
    res1 = edit_photo(src, out_1x1, config, aspect="1:1")
    assert res1.width == 1080 and res1.height == 1080
    assert out_1x1.is_file()

    # 4:5
    out_4x5 = config.out_dir / "out_4x5.jpg"
    res2 = edit_photo(src, out_4x5, config, aspect="4:5")
    assert res2.width == 1080 and res2.height == 1350

    # 9:16
    out_9x16 = config.out_dir / "out_9x16.jpg"
    res3 = edit_photo(src, out_9x16, config, aspect="9:16")
    assert res3.width == 1080 and res3.height == 1920

    # original
    out_orig = config.out_dir / "out_orig.jpg"
    res4 = edit_photo(src, out_orig, config, aspect="original")
    assert res4.width == 800 and res4.height == 600


def test_photo_with_backdrop_and_colour_match(config: Config) -> None:
    subject = _create_synthetic_image(config.in_dir / "subject.png", 300, 600, mode="RGBA")
    bg = _create_synthetic_image(config.in_dir / "bg.jpg", 1920, 1080, color=(50, 80, 120))
    out = config.out_dir / "composed_photo.png"

    res = edit_photo(
        subject,
        out,
        config,
        aspect="4:5",
        bg=bg,
        colour_match=True,
        shadow=True,
    )
    assert res.width == 1080 and res.height == 1350
    assert out.is_file()


def test_photo_looks_and_sharpen(config: Config) -> None:
    src = _create_synthetic_image(config.in_dir / "look_src.png", 400, 400)

    for look in ("warm", "cool", "vintage", "noir", "vibrant", "punchy", "cinematic"):
        out = config.out_dir / f"look_{look}.png"
        res = edit_photo(src, out, config, look=look, sharpen=True)
        assert res.output_path.is_file()


def test_photo_batch_processing(config: Config) -> None:
    batch_in = config.in_dir / "batch_input"
    batch_in.mkdir(parents=True, exist_ok=True)
    _create_synthetic_image(batch_in / "p1.jpg", 300, 300)
    _create_synthetic_image(batch_in / "p2.png", 400, 500)
    _create_synthetic_image(batch_in / "p3.webp", 600, 400)

    batch_out = config.out_dir / "batch_output"
    res = process_photo_batch(batch_in, batch_out, config, aspect="1:1", look="vibrant")

    assert len(res.results) == 3
    for r in res.results:
        assert r.width == 1080 and r.height == 1080
        assert r.output_path.is_file()


def test_photo_error_cases(config: Config) -> None:
    src = _create_synthetic_image(config.in_dir / "err_src.jpg", 300, 300)

    with pytest.raises(MediaLabError, match="does not exist"):
        edit_photo(config.in_dir / "nonexistent.jpg", config.out_dir / "out.jpg", config)

    with pytest.raises(ValidationError, match="aspect must be one of"):
        edit_photo(src, config.out_dir / "out.jpg", config, aspect="21:9")

    with pytest.raises(ValidationError, match="look must be one of"):
        edit_photo(src, config.out_dir / "out.jpg", config, look="neon_cyberpunk")

    empty_dir = config.in_dir / "empty_photos"
    empty_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(MediaLabError, match="no images found"):
        process_photo_batch(empty_dir, config.out_dir / "empty_out", config)


def test_photo_overwrite_protection(config: Config) -> None:
    src = _create_synthetic_image(config.in_dir / "over_src.jpg", 300, 300)
    out = config.out_dir / "protected_photo.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"existing")

    with pytest.raises(PathSafetyError, match="output already exists"):
        edit_photo(src, out, config, force=False)

    res = edit_photo(src, out, config, force=True)
    assert res.output_path == out


def test_cli_photo(
    config: Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("media_lab.cli.load_config", lambda: config)
    src = _create_synthetic_image(config.in_dir / "cli_photo.jpg", 500, 500)
    out = config.out_dir / "cli_photo_1x1.jpg"

    rc = main(["photo", str(src), "-o", str(out), "--aspect", "1:1", "--look", "warm"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "photo written" in captured
    assert "1080x1080" in captured
