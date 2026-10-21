"""Tests for NLE Timeline Export (FCPXML and CMX 3600 EDL)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from media_lab.config import Config
from media_lab.recipes.timeline_export import (
    TimelineClip,
    TimelineData,
    export_timeline,
    generate_cmx3600_edl,
    generate_fcpxml,
    seconds_to_timecode,
)


@pytest.fixture
def sample_video(tmp_path: Path, config: Config) -> Path:
    out = tmp_path / "sample_video.mp4"
    cmd = [
        str(config.ffmpeg),
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=3:size=1920x1080:rate=25",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=3",
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


def test_seconds_to_timecode() -> None:
    assert seconds_to_timecode(0.0, fps=25.0) == "00:00:00:00"
    assert seconds_to_timecode(1.0, fps=25.0) == "00:00:01:00"
    assert (
        seconds_to_timecode(1.5, fps=25.0) == "00:00:01:12"
        or seconds_to_timecode(1.5, fps=25.0) == "00:00:01:13"
    )
    assert seconds_to_timecode(3661.04, fps=25.0) == "01:01:01:01"


def test_generate_cmx3600_edl() -> None:
    clips = (
        TimelineClip(
            name="clip1.mp4",
            source_path=Path("/tmp/clip1.mp4"),
            start_s=0.0,
            duration_s=4.0,
            timeline_start_s=0.0,
        ),
        TimelineClip(
            name="clip2.mp4",
            source_path=Path("/tmp/clip2.mp4"),
            start_s=1.0,
            duration_s=5.0,
            timeline_start_s=4.0,
        ),
    )
    data = TimelineData(
        project_name="Commercial Spot",
        fps=25.0,
        width=1920,
        height=1080,
        clips=clips,
    )
    edl_text = generate_cmx3600_edl(data)
    assert "TITLE: COMMERCIAL_SPOT" in edl_text
    assert "FCM: NON-DROP FRAME" in edl_text
    assert "001" in edl_text
    assert "FROM CLIP NAME: clip1.mp4" in edl_text
    assert "002" in edl_text
    assert "FROM CLIP NAME: clip2.mp4" in edl_text


def test_generate_fcpxml() -> None:
    clips = (
        TimelineClip(
            name="hero.mp4",
            source_path=Path("/tmp/hero.mp4"),
            start_s=0.0,
            duration_s=10.0,
            timeline_start_s=0.0,
            has_audio=True,
        ),
    )
    data = TimelineData(
        project_name="Reel Project",
        fps=25.0,
        width=1080,
        height=1920,
        clips=clips,
    )
    xml_text = generate_fcpxml(data)
    assert "<!DOCTYPE fcpxml>" in xml_text
    assert '<fcpxml version="1.10">' in xml_text
    assert 'width="1080" height="1920"' in xml_text
    assert "asset-clip" in xml_text
    assert 'name="hero.mp4"' in xml_text


def test_export_timeline_fcpxml_file(sample_video: Path, config: Config, tmp_path: Path) -> None:
    out_xml = tmp_path / "timeline.fcpxml"
    res = export_timeline(sample_video, out_xml, config, format="fcpxml")

    assert res.output.is_file()
    assert res.format == "fcpxml"
    assert res.clip_count == 1
    content = out_xml.read_text(encoding="utf-8")
    assert "<fcpxml" in content
    assert sample_video.name in content


def test_export_timeline_edl_file(sample_video: Path, config: Config, tmp_path: Path) -> None:
    out_edl = tmp_path / "timeline.edl"
    res = export_timeline(sample_video, out_edl, config, format="edl")

    assert res.output.is_file()
    assert res.format == "edl"
    content = out_edl.read_text(encoding="utf-8")
    assert "TITLE: SAMPLE_VIDEO" in content
    assert sample_video.name in content
