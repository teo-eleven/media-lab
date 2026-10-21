"""The single place that invokes ffmpeg directly.

Most work goes through kinocut. A few operations do not: kinocut 1.15.1's
`audio-bed` cannot run on macOS (it requires Linux-only memfd sealing), so
the music bed is built here instead.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .errors import FFmpegError

DEFAULT_FFMPEG_TIMEOUT_S = 900
LOUDNESS_STATS_PATTERN = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.DOTALL)
SILENT_LUFS = -70.0
ALPHA_STAT_PATTERN = re.compile(r"lavfi\.signalstats\.Y(MIN|MAX)=(\d+)")
# ffmpeg's native vp9 decoder does not expose the alpha plane; libvpx-vp9 does.
VP9_ALPHA_DECODER = "libvpx-vp9"
# Recipes only ever write ProRes .mov cutouts, so the WebM branch below is
# reachable only by direct callers of this function. It is kept because the
# alpha it decodes is otherwise invisible.
ALPHA_SAMPLE_FRAMES = 12


@dataclass(frozen=True, slots=True)
class FFmpegResult:
    """Outcome of one successful ffmpeg invocation."""

    args: tuple[str, ...]
    stderr: str


def run_ffmpeg(
    args: list[str], config: Config, *, timeout_s: int = DEFAULT_FFMPEG_TIMEOUT_S
) -> FFmpegResult:
    """Invoke the project's ffmpeg. Raises FFmpegError on any failure."""
    command = [str(config.ffmpeg), "-hide_banner", "-nostdin", "-y", *args]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed executable, no shell
            command, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(tuple(args), -1, f"timed out after {timeout_s}s") from exc

    if completed.returncode != 0:
        raise FFmpegError(tuple(args), completed.returncode, completed.stderr)
    return FFmpegResult(args=tuple(args), stderr=completed.stderr)


def measure_integrated_loudness(path: Path, config: Config) -> float:
    """Measure a file's integrated loudness in LUFS with an analysis pass."""
    result = run_ffmpeg(
        ["-i", str(path), "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        config,
        timeout_s=DEFAULT_FFMPEG_TIMEOUT_S,
    )
    match = LOUDNESS_STATS_PATTERN.search(result.stderr)
    if match is None:
        raise FFmpegError(("loudnorm",), 0, f"no loudness statistics reported for {path}")

    try:
        stats = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise FFmpegError(("loudnorm",), 0, f"unparseable loudness statistics: {exc}") from exc

    raw = stats.get("input_i")
    try:
        measured = float(raw)
    except (TypeError, ValueError) as exc:
        raise FFmpegError(("loudnorm",), 0, f"loudness statistics had no input_i: {raw!r}") from exc
    # ffmpeg reports -inf for pure silence; clamp so callers get a usable number.
    return SILENT_LUFS if measured < SILENT_LUFS else measured


def measure_alpha_spread(path: Path, config: Config) -> int:
    """Return the widest alpha variation seen across the clip, on a 0-255 scale.

    A cutout whose alpha never varies carries no matte at all: the subject was
    never separated from its background. Sampling several frames spread over
    the clip avoids both false alarms (the subject enters after frame one) and
    blind spots (the matte collapses part way through).
    """
    args = ["-hide_banner"]
    if path.suffix.lower() == ".webm":
        args += ["-c:v", VP9_ALPHA_DECODER]
    args += [
        "-i",
        str(path),
        # Normalising to 8-bit first keeps the statistics on one scale whatever
        # bit depth the source carries.
        "-vf",
        f"format=yuva444p,alphaextract,thumbnail={ALPHA_SAMPLE_FRAMES},signalstats,metadata=print",
        "-frames:v",
        str(ALPHA_SAMPLE_FRAMES),
        "-f",
        "null",
        "-",
    ]
    stderr = run_ffmpeg(args, config, timeout_s=DEFAULT_FFMPEG_TIMEOUT_S).stderr

    readings = ALPHA_STAT_PATTERN.findall(stderr)
    minimums = [int(value) for kind, value in readings if kind == "MIN"]
    maximums = [int(value) for kind, value in readings if kind == "MAX"]
    if not minimums or not maximums:
        raise FFmpegError(("alphaextract",), 0, f"no alpha statistics reported for {path}")
    return max(maximums) - min(minimums)


def run_filtergraph(
    inputs: Sequence[tuple[Sequence[str], str]],
    filter_complex: str,
    map_target: str,
    encode_args: Sequence[str],
    output: Path,
    config: Config,
    *,
    frames: int | None = None,
    timeout_s: int = DEFAULT_FFMPEG_TIMEOUT_S,
) -> FFmpegResult:
    """Run one `-filter_complex` graph. `inputs` is a list of (lead_args, path).

    The recipe layer builds the graph string and the encode args; this only
    assembles the command and hands it to run_ffmpeg.
    """
    args: list[str] = []
    for lead_args, path in inputs:
        args += [*lead_args, "-i", path]
    args += ["-filter_complex", filter_complex, "-map", map_target]
    if frames is not None:
        args += ["-frames:v", str(frames)]
    args += [*encode_args, str(output)]
    return run_ffmpeg(args, config, timeout_s=timeout_s)
