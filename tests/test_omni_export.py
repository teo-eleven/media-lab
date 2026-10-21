from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from media_lab.config import Config
from media_lab.recipes.omni_export import omni_export


@pytest.fixture
def sample_clip(tmp_path: Path, config: Config) -> Path:
    out = tmp_path / "master_sample.mp4"
    cmd = [
        str(config.ffmpeg),
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=640x360:rate=12",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def test_omni_export_generates_all_aspects(
    sample_clip: Path, config: Config, tmp_path: Path
) -> None:
    deliverables_dir = tmp_path / "campaign_assets"
    result = omni_export(
        sample_clip,
        deliverables_dir,
        config,
        aspects=("9:16", "1:1"),
        mode="center",
        force=True,
    )

    assert result.output_dir == deliverables_dir
    assert len(result.deliverables) == 2
    for deliv in result.deliverables:
        assert deliv.path.is_file()
        assert deliv.media.has_video
