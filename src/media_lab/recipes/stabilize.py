"""Recipe: 2-pass video stabilization and camera motion smoothing using VidStab.

Detects shaky camera motion, compensates for manual zoom-out/zoom-in glitches,
and produces steady, cinematic camera framing.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..ffmpeg import run_ffmpeg
from ..paths import ensure_readable_source, ensure_writable_output, work_path
from ..probe import MediaInfo
from ..verify import Expectations, verify_render


@dataclass(frozen=True, slots=True)
class StabilizeResult:
    """Outcome of video stabilization pass."""

    output: Path
    media: MediaInfo
    smoothing: int


def stabilize_video(
    source: Path | str,
    output: Path | str,
    config: Config,
    *,
    smoothing: int = 20,
    shakiness: int = 8,
    force: bool = False,
) -> StabilizeResult:
    """Stabilize shaky footage or neutralize irregular manual zoom/panning motions."""
    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    trans_file = work_path(config, f"vidstab_{resolved_source.stem}", ".trf")

    # Pass 1: Motion detection
    pass1_cmd = [
        "-i",
        str(resolved_source),
        "-vf",
        f"vidstabdetect=stepsize=6:shakiness={shakiness}:accuracy=15:result={trans_file}",
        "-f",
        "null",
        "-",
    ]
    run_ffmpeg(pass1_cmd, config)

    # Pass 2: Transformation & smoothing
    pass2_cmd = [
        "-i",
        str(resolved_source),
        "-vf",
        f"vidstabtransform=input={trans_file}:smoothing={smoothing}:zoom=0:optzoom=1:interpol=bicubic",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(resolved_output),
    ]
    run_ffmpeg(pass2_cmd, config)

    # Clean up intermediate transform log
    if trans_file.exists():
        with contextlib.suppress(OSError):
            trans_file.unlink()

    media_info = verify_render(resolved_output, config, Expectations(requires_video=True))

    return StabilizeResult(
        output=resolved_output,
        media=media_info,
        smoothing=smoothing,
    )
