"""Pull evenly-spaced frames from a clip and tile them into a contact sheet.

Shared by `proxy-preview` and the `punto` runner. Every ffmpeg call goes
through `ffmpeg.run_ffmpeg`.
"""

from __future__ import annotations

import math
from pathlib import Path

from .config import Config
from .ffmpeg import run_ffmpeg
from .probe import probe

FRAME_PATTERN = "f-%02d.png"
DEFAULT_COLS = 4
DEFAULT_CELL_WIDTH = 360


def extract_frames(source: Path, count: int, config: Config, dest_dir: Path) -> Path:
    """Write `count` evenly-spaced frames of `source` into `dest_dir`.

    Frames land at `f-01.png` ... and are sampled at the midpoints of `count`
    equal slices, so the first and last are inset from the exact ends.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for stale in dest_dir.glob("f-*.png"):
        stale.unlink()

    duration = probe(source, config).duration_s or 0.0
    for index in range(count):
        stamp = duration * (index + 0.5) / count
        run_ffmpeg(
            [
                "-ss",
                f"{stamp:.3f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(dest_dir / (FRAME_PATTERN % (index + 1))),
            ],
            config,
        )
    return dest_dir


def tile(
    frames_dir: Path,
    output: Path,
    config: Config,
    *,
    cols: int = DEFAULT_COLS,
    cell_width: int = DEFAULT_CELL_WIDTH,
) -> Path:
    """Tile every `f-*.png` in `frames_dir` into a `cols`-wide grid at `output`."""
    count = len(list(frames_dir.glob("f-*.png")))
    if count == 0:
        raise ValueError(f"no f-*.png frames in {frames_dir}")
    rows = math.ceil(count / cols)
    run_ffmpeg(
        [
            "-i",
            str(frames_dir / FRAME_PATTERN),
            "-vf",
            f"scale={cell_width}:-1,tile={cols}x{rows}:margin=6:padding=6:color=black",
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(output),
        ],
        config,
    )
    return output


def side_by_side(left: Path, right: Path, output: Path, config: Config) -> Path:
    """Put two images next to each other, scaled to a common height, at `output`."""
    run_ffmpeg(
        [
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            "[0:v]scale=-2:720[l];[1:v]scale=-2:720[r];[l][r]hstack=2",
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(output),
        ],
        config,
    )
    return output
