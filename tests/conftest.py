"""Shared fixtures. Test media is synthesised with ffmpeg, never checked in."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from media_lab.config import Config

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = REPO_ROOT / "bin"
CLIP_SECONDS = 2
CLIP_WIDTH = 640
CLIP_HEIGHT = 360
CLIP_FPS = 25


def _run_ffmpeg(args: list[str]) -> None:
    command = [str(BIN_DIR / "ffmpeg"), "-y", "-loglevel", "error", *args]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"fixture ffmpeg failed: {completed.stderr.strip()}")


@pytest.fixture(scope="session")
def bin_dir() -> Path:
    if not (BIN_DIR / "ffmpeg").is_file():
        pytest.skip("ffmpeg missing; run ./scripts/fetch-ffmpeg.sh")
    return BIN_DIR


@pytest.fixture
def config(tmp_path: Path, bin_dir: Path) -> Config:
    """A Config rooted in a temp directory, using the real ffmpeg binaries."""
    (tmp_path / "in").mkdir()
    return Config(
        root=tmp_path,
        ffmpeg_dir=bin_dir,
        in_dir=tmp_path / "in",
        out_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        kino_timeout_s=300,
        hyperframes_command=REPO_ROOT / "node_modules" / ".bin" / "hyperframes",
        rvm_repo=tmp_path / "tools" / "RobustVideoMatting",
        weights_dir=tmp_path / "weights",
    )


@pytest.fixture
def sample_video(config: Config) -> Path:
    """A short clip with both a video and an audio stream."""
    path = config.in_dir / "sample.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={CLIP_WIDTH}x{CLIP_HEIGHT}:rate={CLIP_FPS}:duration={CLIP_SECONDS}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={CLIP_SECONDS}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ]
    )
    return path


@pytest.fixture
def silent_video(config: Config) -> Path:
    """A short clip with no audio stream at all."""
    path = config.in_dir / "silent.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={CLIP_WIDTH}x{CLIP_HEIGHT}:rate={CLIP_FPS}:duration={CLIP_SECONDS}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
    )
    return path


@pytest.fixture
def sample_music(config: Config) -> Path:
    """An audio-only file long enough to need trimming."""
    path = config.in_dir / "music.m4a"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:duration={CLIP_SECONDS * 2}",
            "-c:a",
            "aac",
            str(path),
        ]
    )
    return path


@pytest.fixture
def corrupt_file(config: Config) -> Path:
    path = config.in_dir / "corrupt.mp4"
    path.write_bytes(b"this is definitely not a video container")
    return path
