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


def test_doctor_render_auto_fix_odd_dimensions(tmp_path: Path, config: Config) -> None:
    from media_lab.recipes.render_doctor import auto_fix_render

    odd_video = tmp_path / "odd_video.mp4"
    cmd = [
        str(config.ffmpeg),
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=639x359:rate=10",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv444p",
        str(odd_video),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    initial_report = diagnose_render(odd_video, config, sample_frames=0)
    assert not initial_report.passed

    fixed_video = tmp_path / "fixed_even.mp4"
    fix_result = auto_fix_render(odd_video, fixed_video, initial_report, config, force=True)

    assert fix_result.output.is_file()
    assert fix_result.media.width % 2 == 0
    assert fix_result.media.height % 2 == 0

    re_report = diagnose_render(fixed_video, config, sample_frames=0)
    assert re_report.passed or re_report.has_warnings


def test_doctor_report_to_dict(synthetic_video: Path, config: Config) -> None:
    report = diagnose_render(synthetic_video, config, sample_frames=2, generate_contact_sheet=False)
    data = report.to_dict()
    assert "overall_status" in data
    assert "checks" in data
    assert len(data["checks"]) > 0
