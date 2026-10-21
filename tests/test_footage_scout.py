"""Tests for footage scout recipe and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from media_lab import cli
from media_lab.config import Config
from media_lab.recipes.footage_scout import (
    ScoutedAsset,
    _calculate_aspect_ratio_string,
    _calculate_orientation,
    classify_media_category,
    format_scouted_table,
    scout_library,
)
from tests.conftest import _run_ffmpeg


def _create_synthetic_assets(base_dir: Path) -> dict[str, Path]:
    bg_dir = base_dir / "backgrounds"
    sfx_dir = base_dir / "sfx"
    broll_dir = base_dir / "broll"
    bg_dir.mkdir(parents=True, exist_ok=True)
    sfx_dir.mkdir(parents=True, exist_ok=True)
    broll_dir.mkdir(parents=True, exist_ok=True)

    # 1. Background video (horizontal 16:9, 2s)
    bg_video = bg_dir / "cyberpunk_city.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x360:rate=25:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(bg_video),
        ]
    )

    # 2. Background still image (horizontal)
    bg_still = bg_dir / "studio_neon.png"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=640x360:d=1",
            "-frames:v",
            "1",
            str(bg_still),
        ]
    )

    # 3. SFX whoosh (audio, 1s)
    sfx_whoosh = sfx_dir / "whoosh_fast.wav"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=300:duration=1",
            "-c:a",
            "pcm_s16le",
            str(sfx_whoosh),
        ]
    )

    # 4. SFX impact (audio, 0.5s)
    sfx_impact = sfx_dir / "heavy_impact.wav"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=100:duration=0.5",
            "-c:a",
            "pcm_s16le",
            str(sfx_impact),
        ]
    )

    # 5. Vertical B-Roll (vertical 9:16, 3s)
    broll_vert = broll_dir / "gadget_review_cutaway.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=360x640:rate=25:duration=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(broll_vert),
        ]
    )

    return {
        "bg_video": bg_video,
        "bg_still": bg_still,
        "sfx_whoosh": sfx_whoosh,
        "sfx_impact": sfx_impact,
        "broll_vert": broll_vert,
    }


def test_classify_media_category() -> None:
    assert classify_media_category(Path("in/sfx/fast_whoosh.wav"), "audio") == "sfx_whoosh"
    assert classify_media_category(Path("in/sfx/punch_impact.wav"), "audio") == "sfx_impact"
    assert classify_media_category(Path("in/sfx/soft_click.wav"), "audio") == "sfx_click"
    assert classify_media_category(Path("in/sfx/bubble_pop.wav"), "audio") == "sfx_pop"
    assert classify_media_category(Path("in/sfx/ding_notification.wav"), "audio") == "sfx_ding"
    assert classify_media_category(Path("in/sfx/tension_riser.wav"), "audio") == "sfx_riser"
    assert classify_media_category(Path("in/sfx/generic_track.mp3"), "audio") == "sfx"
    assert classify_media_category(Path("in/music/theme.mp3"), "audio") == "audio"

    assert classify_media_category(Path("in/backgrounds/city.mp4"), "video") == "plate_video"
    assert classify_media_category(Path("in/plate_nature.mov"), "video") == "plate_video"
    assert classify_media_category(Path("in/broll/cutaway.mp4"), "video") == "broll"
    assert classify_media_category(Path("in/backgrounds/wall.png"), "image") == "plate_still"
    assert classify_media_category(Path("in/photos/avatar.jpg"), "image") == "image"
    assert classify_media_category(Path("in/unknown.bin"), "other") == "unknown"


def test_aspect_ratio_and_orientation_helpers() -> None:
    assert _calculate_aspect_ratio_string(0, 0) is None
    assert _calculate_aspect_ratio_string(1080, 1920) == "9:16"
    assert _calculate_aspect_ratio_string(1920, 1080) == "16:9"
    assert _calculate_aspect_ratio_string(1000, 1000) == "1:1"
    assert _calculate_aspect_ratio_string(1080, 1350) == "4:5"
    assert _calculate_aspect_ratio_string(2560, 1080) == "21:9"
    assert _calculate_aspect_ratio_string(500, 300) == "500:300"

    assert _calculate_orientation(0, 0) is None
    assert _calculate_orientation(1080, 1920) == "vertical"
    assert _calculate_orientation(1920, 1080) == "horizontal"
    assert _calculate_orientation(1000, 1000) == "square"


def test_scout_library_finds_and_filters_assets(config: Config) -> None:
    _create_synthetic_assets(config.in_dir)

    all_assets = scout_library([config.in_dir], config)
    assert len(all_assets) == 5

    names = {a.name for a in all_assets}
    assert "cyberpunk_city.mp4" in names
    assert "studio_neon.png" in names
    assert "whoosh_fast.wav" in names
    assert "heavy_impact.wav" in names
    assert "gadget_review_cutaway.mp4" in names

    # Filter by vertical orientation
    vertical_assets = scout_library([config.in_dir], config, orientation="vertical")
    assert len(vertical_assets) == 1
    assert vertical_assets[0].name == "gadget_review_cutaway.mp4"
    assert vertical_assets[0].aspect_ratio == "9:16"

    # Filter by horizontal orientation
    horiz_assets = scout_library([config.in_dir], config, orientation="horizontal")
    assert len(horiz_assets) == 2

    # Filter by SFX category / media_type
    sfx_assets = scout_library([config.in_dir], config, media_type="audio")
    assert len(sfx_assets) == 2
    assert {a.category for a in sfx_assets} == {"sfx_whoosh", "sfx_impact"}

    # Filter by query
    cyber_assets = scout_library([config.in_dir], config, query="cyberpunk")
    assert len(cyber_assets) == 1
    assert cyber_assets[0].category == "plate_video"

    # Filter by duration range
    mid_assets = scout_library([config.in_dir], config, min_duration=0.8, max_duration=2.5)
    mid_names = [a.name for a in mid_assets]
    assert "whoosh_fast.wav" in mid_names
    assert "cyberpunk_city.mp4" in mid_names
    assert "heavy_impact.wav" not in mid_names
    assert "gadget_review_cutaway.mp4" not in mid_names

    # Non-existent directory handling
    empty = scout_library([config.root / "does_not_exist"], config)
    assert empty == []


def test_format_scouted_table() -> None:
    assert format_scouted_table([]) == "No matching assets found in library."

    dummy = ScoutedAsset(
        path=Path("/tmp/clip.mp4"),
        name="clip.mp4",
        category="broll",
        media_type="video",
        duration_s=5.2,
        width=1920,
        height=1080,
        fps=30.0,
        aspect_ratio="16:9",
        orientation="horizontal",
        file_size_bytes=1024 * 1024 * 3,
    )
    table = format_scouted_table([dummy])
    assert "Category" in table
    assert "clip.mp4" in table
    assert "1920x1080" in table
    assert "3.00MB" in table


def test_scout_cli_output(
    config: Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("media_lab.cli.load_config", lambda: config)
    _create_synthetic_assets(config.in_dir)

    # Test JSON output
    code = cli.main(["scout", "--dir", str(config.in_dir), "--json"])
    assert code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) == 5

    # Test vertical filter via CLI
    code_vert = cli.main(["scout", "--dir", str(config.in_dir), "--vertical"])
    assert code_vert == 0
    out_vert = capsys.readouterr().out
    assert "gadget_review_cutaway.mp4" in out_vert
    assert "cyberpunk_city.mp4" not in out_vert

    # Test human table output
    code_table = cli.main(["scout", "--dir", str(config.in_dir), "--type", "audio"])
    assert code_table == 0
    out_table = capsys.readouterr().out
    assert "whoosh_fast.wav" in out_table
    assert "Found 2 matching assets." in out_table
