"""Tests for Video & Pacing recipes: smart_reframe, punch_zoom, and broll."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from media_lab.cli import main
from media_lab.config import Config
from media_lab.errors import ValidationError
from media_lab.recipes.broll import BrollCut, insert_broll
from media_lab.recipes.punch_zoom import (
    ZoomCue,
    generate_auto_zoom_cues,
    punch_zoom,
)
from media_lab.recipes.smart_reframe import (
    calculate_crop_box,
    detect_frame_subject_center,
    smart_reframe,
)


def _generate_synthetic_clip(
    path: Path, config: Config, duration: float = 3.0, width: int = 640, height: int = 360
) -> Path:
    """Generate 16:9 synthetic test video with audio."""
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
            f"testsrc=duration={duration}:size={width}x{height}:rate=25",
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
            "-b:a",
            "128k",
            str(path),
        ],
        check=True,
    )
    return path


def _generate_synthetic_image(path: Path, width: int = 640, height: int = 360) -> Path:
    """Generate still test image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Smart Reframe Tests
# ---------------------------------------------------------------------------


def test_detect_frame_subject_center() -> None:
    # Uniform black frame -> defaults to center 0.5
    black = np.zeros((100, 200, 3), dtype=np.uint8)
    assert detect_frame_subject_center(black) == 0.5

    # Frame with simulated skin patch on the right side
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    # RGB values that map to Cb in [77, 127], Cr in [133, 173] (e.g. skin tone R=210, G=150, B=120)
    img[:, 150:180] = [210, 150, 120]
    center = detect_frame_subject_center(img)
    assert center > 0.6  # Subject centroid skewed right


def test_calculate_crop_box() -> None:
    # 16:9 to 9:16 (640x360 -> target aspect 9/16 = 0.5625)
    # Target crop: h=360, w=202
    cx, cy, cw, ch = calculate_crop_box(640, 360, 9 / 16, subject_center_x=0.5)
    assert ch == 360
    assert cw == 202
    assert cy == 0
    assert cx == 218

    # Subject center skewed to left (e.g. 0.1)
    cx_left, _, _, _ = calculate_crop_box(640, 360, 9 / 16, subject_center_x=0.1)
    assert cx_left == 0


def test_smart_reframe_modes(config: Config) -> None:
    src = _generate_synthetic_clip(config.in_dir / "wide.mp4", config, duration=1.5)

    # 1. Smart mode
    out_smart = config.out_dir / "reframe_smart.mp4"
    res_smart = smart_reframe(
        src,
        out_smart,
        config,
        target_aspect="9:16",
        mode="smart",
        target_width=360,
        target_height=640,
    )
    assert res_smart.output.is_file()
    assert res_smart.target_aspect == "9:16"
    assert res_smart.media.width == 360
    assert res_smart.media.height == 640
    assert res_smart.media.has_video is True
    assert res_smart.media.has_audio is True

    # 2. Center mode (1:1 square)
    out_center = config.out_dir / "reframe_center.mp4"
    res_center = smart_reframe(
        src,
        out_center,
        config,
        target_aspect="1:1",
        mode="center",
        target_width=360,
        target_height=360,
    )
    assert res_center.output.is_file()
    assert res_center.target_aspect == "1:1"
    assert res_center.media.width == 360
    assert res_center.media.height == 360

    # 3. Split mode (blurred backdrop fill)
    out_split = config.out_dir / "reframe_split.mp4"
    res_split = smart_reframe(
        src,
        out_split,
        config,
        target_aspect="9:16",
        mode="split",
        target_width=360,
        target_height=640,
    )
    assert res_split.output.is_file()
    assert res_split.media.width == 360
    assert res_split.media.height == 640


def test_smart_reframe_validation_errors(config: Config) -> None:
    src = _generate_synthetic_clip(config.in_dir / "wide.mp4", config, duration=1.0)
    out = config.out_dir / "out.mp4"

    with pytest.raises(ValidationError, match="unsupported target_aspect"):
        smart_reframe(src, out, config, target_aspect="invalid_aspect")

    with pytest.raises(ValidationError, match="unsupported reframe mode"):
        smart_reframe(src, out, config, mode="magic")


# ---------------------------------------------------------------------------
# Punch Zoom Tests
# ---------------------------------------------------------------------------


def test_generate_auto_zoom_cues() -> None:
    cues = generate_auto_zoom_cues(15.0, interval_s=4.0, zoom_duration_s=2.0, scale=1.15)
    assert len(cues) == 2
    assert cues[0].start_s == 4.0
    assert cues[0].duration_s == 2.0
    assert cues[0].scale == 1.15
    assert cues[1].start_s == 10.0


def test_punch_zoom_recipes(config: Config) -> None:
    src = _generate_synthetic_clip(config.in_dir / "wide.mp4", config, duration=2.0)

    # 1. Explicit cut cues
    out_cues = config.out_dir / "zoomed_cut.mp4"
    cues = [ZoomCue(start_s=0.5, duration_s=0.8, scale=1.15, cx=0.5, cy=0.4, transition="cut")]
    res_cues = punch_zoom(src, out_cues, config, cues=cues)
    assert res_cues.output.is_file()
    assert res_cues.cues_applied == 1
    assert res_cues.media.width == 640
    assert res_cues.media.height == 360

    # 2. Fade transition zoom
    out_fade = config.out_dir / "zoomed_fade.mp4"
    fade_cues = [ZoomCue(start_s=0.4, duration_s=1.0, scale=1.20, transition="fade")]
    res_fade = punch_zoom(src, out_fade, config, cues=fade_cues)
    assert res_fade.output.is_file()
    assert res_fade.cues_applied == 1

    # 3. Auto interval zoom
    out_auto = config.out_dir / "zoomed_auto.mp4"
    res_auto = punch_zoom(
        src, out_auto, config, auto_interval_s=0.5, auto_zoom_duration_s=0.5, auto_scale=1.12
    )
    assert res_auto.output.is_file()
    assert res_auto.cues_applied >= 1


def test_punch_zoom_validation_errors(config: Config) -> None:
    src = _generate_synthetic_clip(config.in_dir / "wide.mp4", config, duration=1.0)
    out = config.out_dir / "out.mp4"

    with pytest.raises(ValidationError, match="no zoom cues specified"):
        punch_zoom(src, out, config, cues=[])

    with pytest.raises(ValidationError, match="scale must be between"):
        punch_zoom(src, out, config, cues=[ZoomCue(start_s=0.1, duration_s=0.5, scale=5.0)])


# ---------------------------------------------------------------------------
# B-Roll Tests
# ---------------------------------------------------------------------------


def test_broll_cutaway(config: Config) -> None:
    src = _generate_synthetic_clip(config.in_dir / "main.mp4", config, duration=2.5)
    broll_video = _generate_synthetic_clip(
        config.in_dir / "broll_vid.mp4", config, duration=1.0, width=320, height=240
    )
    broll_img = _generate_synthetic_image(config.in_dir / "broll_still.png")

    # 1. Video cutaway with cut transition and mute
    out_cut = config.out_dir / "broll_cut.mp4"
    res_cut = insert_broll(
        src,
        out_cut,
        [BrollCut(path=broll_video, start_s=0.5, duration_s=1.0, transition="cut", volume=0.0)],
        config,
    )
    assert res_cut.output.is_file()
    assert res_cut.cuts_count == 1
    assert res_cut.media.has_audio is True
    assert res_cut.media.width == 640

    # 2. Image cutaway with fade transition
    out_fade = config.out_dir / "broll_image_fade.mp4"
    res_fade = insert_broll(
        src,
        out_fade,
        [
            BrollCut(
                path=broll_img, start_s=0.8, duration_s=1.2, transition="fade", fade_duration_s=0.2
            )
        ],
        config,
    )
    assert res_fade.output.is_file()
    assert res_fade.cuts_count == 1

    # 3. Cutaway with audio mixing underneath
    out_mix = config.out_dir / "broll_audio_mix.mp4"
    res_mix = insert_broll(
        src,
        out_mix,
        [BrollCut(path=broll_video, start_s=0.5, duration_s=1.0, volume=0.5)],
        config,
    )
    assert res_mix.output.is_file()
    assert res_mix.cuts_count == 1


def test_broll_validation_errors(config: Config) -> None:
    src = _generate_synthetic_clip(config.in_dir / "main.mp4", config, duration=1.0)
    out = config.out_dir / "out.mp4"

    with pytest.raises(ValidationError, match="at least one BrollCut"):
        insert_broll(src, out, [], config)

    with pytest.raises(ValidationError, match="transition must be 'cut' or 'fade'"):
        insert_broll(
            src, out, [BrollCut(path=src, start_s=0.0, duration_s=0.5, transition="slide")], config
        )


# ---------------------------------------------------------------------------
# CLI Integration Tests
# ---------------------------------------------------------------------------


def test_cli_video_pacing(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("media_lab.cli.load_config", lambda: config)
    src = _generate_synthetic_clip(config.in_dir / "talk.mp4", config, duration=2.0)
    broll_img = _generate_synthetic_image(config.in_dir / "broll.jpg")

    # CLI smart-reframe
    out_reframe = config.out_dir / "cli_reframe.mp4"
    rc = main(
        [
            "smart-reframe",
            str(src),
            "-o",
            str(out_reframe),
            "--aspect",
            "9:16",
            "--mode",
            "center",
            "--force",
        ]
    )
    assert rc == 0
    assert out_reframe.is_file()

    # CLI punch-zoom
    out_zoom = config.out_dir / "cli_zoom.mp4"
    rc = main(
        [
            "punch-zoom",
            str(src),
            "-o",
            str(out_zoom),
            "--auto-interval",
            "0.5",
            "--duration",
            "0.5",
            "--scale",
            "1.15",
            "--force",
        ]
    )
    assert rc == 0
    assert out_zoom.is_file()

    # CLI broll
    out_broll = config.out_dir / "cli_broll.mp4"
    rc = main(
        [
            "broll",
            str(src),
            "-o",
            str(out_broll),
            "--clip",
            str(broll_img),
            "--start",
            "0.5",
            "--duration",
            "1.0",
            "--transition",
            "fade",
            "--force",
        ]
    )
    assert rc == 0
    assert out_broll.is_file()
