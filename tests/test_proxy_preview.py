"""proxy-preview: fast proxy + contact sheet, optional side-by-side."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from media_lab.config import Config
from media_lab.errors import PathSafetyError
from media_lab.probe import probe
from media_lab.recipes.proxy_preview import proxy_preview


@pytest.fixture
def other_clip(config: Config) -> Path:
    """A second, visually different clip for --compare."""
    path = config.in_dir / "other.mp4"
    subprocess.run(
        [
            str(config.ffmpeg),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:size=480x270:rate=25:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_writes_a_proxy_and_a_sheet(silent_video: Path, config: Config) -> None:
    result = proxy_preview(silent_video, config, height=540, frames=6)

    assert result.proxy.is_file()
    assert probe(result.proxy, config).height == 540
    assert probe(result.proxy, config).has_audio is False
    assert result.sheet.is_file()
    assert probe(result.sheet, config).width > 0
    assert result.comparison is None


def test_compare_writes_a_side_by_side(
    silent_video: Path, other_clip: Path, config: Config
) -> None:
    result = proxy_preview(silent_video, config, frames=4, compare=other_clip)

    assert result.comparison is not None
    assert result.comparison.is_file()
    # the side-by-side is wider than the single sheet
    assert probe(result.comparison, config).width > probe(result.sheet, config).width


def test_rejects_too_few_frames(silent_video: Path, config: Config) -> None:
    with pytest.raises(ValueError, match="frames must be at least 2"):
        proxy_preview(silent_video, config, frames=1)


def test_force_guards_the_proxy(silent_video: Path, config: Config) -> None:
    stale = config.work_dir / f"{silent_video.stem}-proxy.mp4"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"old")

    with pytest.raises(PathSafetyError, match="already exists"):
        proxy_preview(silent_video, config)

    result = proxy_preview(silent_video, config, force=True)
    assert result.proxy.is_file()


def test_rejects_a_missing_source(config: Config) -> None:
    with pytest.raises(PathSafetyError, match="does not exist"):
        proxy_preview(config.in_dir / "nope.mp4", config)
