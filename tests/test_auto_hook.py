from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from media_lab.config import Config
from media_lab.errors import ValidationError
from media_lab.recipes.auto_hook import (
    HookSegment,
    detect_viral_hook,
    extract_viral_hook,
)


@pytest.fixture
def long_synthetic_video(tmp_path: Path, config: Config) -> Path:
    # Creates a 6-second clip: 0-2s silent, 2-4s loud (high energy), 4-6s silent
    out = tmp_path / "podcast_sample.mp4"
    cmd = [
        str(config.ffmpeg),
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=6:size=640x360:rate=12",
        "-f",
        "lavfi",
        "-i",
        (
            "sine=frequency=1000:duration=6,"
            "volume='enable=between(t,2,4):volume=2.0',"
            "volume='enable=not(between(t,2,4)):volume=0.01'"
        ),
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


def test_detect_viral_hook_locates_energy_peak(long_synthetic_video: Path, config: Config) -> None:
    hook = detect_viral_hook(
        long_synthetic_video,
        config,
        window_duration_s=2.0,
    )

    assert isinstance(hook, HookSegment)
    assert hook.duration_s == pytest.approx(2.0, abs=0.2)
    # The loud section was between 2.0 and 4.0s
    assert 1.5 <= hook.start_s <= 2.5
    assert hook.score > 0.0


def test_extract_viral_hook_renders_subclip(
    long_synthetic_video: Path, config: Config, tmp_path: Path
) -> None:
    output = tmp_path / "hook_out.mp4"
    result = extract_viral_hook(
        long_synthetic_video,
        output,
        config,
        target_duration_s=2.0,
        force=True,
    )

    assert result.output.is_file()
    assert result.media.duration_s == pytest.approx(2.0, abs=0.4)
    assert result.media.has_video
    assert result.media.has_audio


def test_auto_hook_raises_on_silent_or_too_short(tmp_path: Path, config: Config) -> None:
    short_clip = tmp_path / "tiny.mp4"
    cmd = [
        str(config.ffmpeg),
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=0.5:size=320x240:rate=10",
        str(short_clip),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    with pytest.raises(ValidationError):
        extract_viral_hook(short_clip, tmp_path / "out.mp4", config, target_duration_s=2.0)
