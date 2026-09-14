"""Read hard facts about a media file with ffprobe."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .errors import ProbeError

PROBE_TIMEOUT_S = 60
ALPHA_PIXEL_FORMAT_MARKERS = ("yuva", "rgba", "bgra", "argb", "abgr", "ya8", "ya16", "pal8")
# VP9-in-WebM stores alpha in BlockAdditional, so the pixel format still reads
# yuv420p. The container signals it with this stream tag instead.
ALPHA_MODE_TAG = "ALPHA_MODE"


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """What ffprobe actually reports about a file."""

    path: Path
    duration_s: float
    width: int
    height: int
    fps: float
    pixel_format: str
    has_video: bool
    has_audio: bool
    has_alpha: bool
    codec_video: str = ""
    codec_audio: str = ""

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 0.0
        return self.width / self.height


def _parse_fps(rate: str) -> float:
    if "/" not in rate:
        try:
            return float(rate)
        except ValueError:
            return 0.0
    numerator, _, denominator = rate.partition("/")
    try:
        den = float(denominator)
        return float(numerator) / den if den else 0.0
    except ValueError:
        return 0.0


def _parse_duration(payload: dict[str, Any], video: dict[str, Any] | None) -> float:
    for source in (payload.get("format", {}), video or {}):
        raw = source.get("duration")
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return 0.0


def _has_alpha_mode_tag(tags: Any) -> bool:
    """Matroska writers disagree on the case of this tag, so match case-insensitively."""
    if not isinstance(tags, dict):
        return False
    wanted = ALPHA_MODE_TAG.casefold()
    return any(str(key).casefold() == wanted and str(value) == "1" for key, value in tags.items())


def probe(path: Path | str, config: Config) -> MediaInfo:
    """Run ffprobe and return a typed summary. Raises ProbeError on failure."""
    target = Path(path)
    command = [
        str(config.ffprobe),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(target),
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed executable, no shell
            command, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"ffprobe timed out on {target}") from exc

    if completed.returncode != 0:
        raise ProbeError(f"ffprobe failed on {target}: {completed.stderr.strip()}")

    try:
        payload: dict[str, Any] = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe returned unparseable JSON for {target}: {exc}") from exc

    streams: list[dict[str, Any]] = payload.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    pixel_format = str(video.get("pix_fmt", "")) if video else ""
    codec_video = str(video.get("codec_name", "")) if video else ""
    codec_audio = str(audio.get("codec_name", "")) if audio else ""
    tags = video.get("tags", {}) if video else {}
    tagged_alpha = _has_alpha_mode_tag(tags)

    return MediaInfo(
        path=target,
        duration_s=_parse_duration(payload, video),
        width=int(video.get("width", 0)) if video else 0,
        height=int(video.get("height", 0)) if video else 0,
        fps=_parse_fps(str(video.get("avg_frame_rate", "0/0"))) if video else 0.0,
        pixel_format=pixel_format,
        has_video=video is not None,
        has_audio=audio is not None,
        has_alpha=(
            any(marker in pixel_format for marker in ALPHA_PIXEL_FORMAT_MARKERS) or tagged_alpha
        ),
        codec_video=codec_video,
        codec_audio=codec_audio,
    )
