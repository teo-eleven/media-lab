from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from media_lab.config import Config
from media_lab.ml_runner import MlRunner
from media_lab.recipes.reelify import reelify


@pytest.fixture
def raw_talking_clip(tmp_path: Path, config: Config) -> Path:
    out = tmp_path / "raw_clip.mp4"
    cmd = [
        str(config.ffmpeg),
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=640x360:rate=15",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
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


def test_reelify_pipeline_execution(raw_talking_clip: Path, config: Config, tmp_path: Path) -> None:
    ml_runner = MlRunner.from_config(config)
    output = tmp_path / "reelified.mp4"

    result = reelify(
        raw_talking_clip,
        output,
        config,
        ml_runner,
        burn_subtitles=False,
        enable_punch_zoom=False,
        enable_progress_bar=True,
        progress_bar_color="yellow",
        force=True,
    )

    assert result.output.is_file()
    assert result.media.width == 1080
    assert result.media.height == 1920
    assert "smart_reframe_9:16" in result.stages_executed
    assert "progress_bar" in result.stages_executed
