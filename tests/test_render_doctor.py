from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from media_lab.config import Config
from media_lab.recipes.render_doctor import diagnose_render


@pytest.fixture
def synthetic_video(tmp_path: Path, config: Config) -> Path:
    out = tmp_path / "synthetic_audit.mp4"
    cmd = [
        str(config.ffmpeg),
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=640x360:rate=24",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=2",
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


def test_doctor_render_passes_on_valid_video(synthetic_video: Path, config: Config) -> None:
    report = diagnose_render(synthetic_video, config, sample_frames=4, generate_contact_sheet=True)
    assert report.passed or report.has_warnings
    assert report.media_info.width == 640
    assert report.media_info.height == 360
    assert report.contact_sheet is not None
    assert report.contact_sheet.is_file()


def test_doctor_render_detects_aspect_mismatch(synthetic_video: Path, config: Config) -> None:
    report = diagnose_render(
        synthetic_video,
        config,
        sample_frames=2,
        expected_aspect="9:16",
        generate_contact_sheet=False,
    )
    aspect_checks = [c for c in report.checks if c.name == "Aspect Ratio Target"]
    assert len(aspect_checks) == 1
    assert aspect_checks[0].status == "WARNING"
    assert report.has_warnings
