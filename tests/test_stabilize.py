"""Tests for 2-pass camera motion stabilization recipe."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from media_lab.config import Config
from media_lab.errors import PathSafetyError
from media_lab.probe import probe
from media_lab.recipes.stabilize import stabilize_video


def _make_shaky_test_video(path: Path, config: Config, *, with_audio: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(config.ffmpeg),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1.0:size=320x240:rate=24",
    ]
    if with_audio:
        cmd.extend(
            [
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1.0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
            ]
        )
    else:
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
            ]
        )
    cmd.append(str(path))
    subprocess.run(cmd, check=True)
    return path


def test_stabilize_video_with_audio(config: Config) -> None:
    src = _make_shaky_test_video(config.in_dir / "shaky.mp4", config, with_audio=True)
    out = config.out_dir / "stabilized.mp4"

    res = stabilize_video(src, out, config, smoothing=10, force=True)
    assert res.output == out
    assert out.is_file()

    info = probe(out, config)
    assert info.has_video
    assert info.has_audio
    assert info.duration_s > 0.8


def test_stabilize_video_silent_input(config: Config) -> None:
    src = _make_shaky_test_video(config.in_dir / "silent_shaky.mp4", config, with_audio=False)
    out = config.out_dir / "stabilized_silent.mp4"

    res = stabilize_video(src, out, config, smoothing=10, force=True)
    assert res.output == out
    assert out.is_file()

    info = probe(out, config)
    assert info.has_video
    assert not info.has_audio
    assert info.duration_s > 0.8


def test_stabilize_video_output_exists_guard(config: Config) -> None:
    src = _make_shaky_test_video(config.in_dir / "shaky2.mp4", config, with_audio=True)
    out = config.out_dir / "stabilized2.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"existing")

    with pytest.raises(PathSafetyError):
        stabilize_video(src, out, config, force=False)
