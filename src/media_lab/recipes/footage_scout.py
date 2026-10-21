"""Footage Scout & Local Asset Library Recipe.

Discovers, classifies, and indexes background plates, B-roll cutaways, and sound effects
(whoosh, impact, pop, ding, riser) across the local repository.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import Config
from ..errors import ProbeError
from ..probe import MediaInfo, probe

SUPPORTED_AUDIO_EXTS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".aiff"}
)
SUPPORTED_VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})
SUPPORTED_IMAGE_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
)

ALL_SUPPORTED_EXTS: frozenset[str] = (
    SUPPORTED_AUDIO_EXTS | SUPPORTED_VIDEO_EXTS | SUPPORTED_IMAGE_EXTS
)

_IMPACT_KEYWORDS = ("impact", "boom", "hit", "thud", "punch", "slam", "crash")
_WHOOSH_KEYWORDS = ("whoosh", "swoosh", "swish")


@dataclass(frozen=True, slots=True)
class ScoutedAsset:
    """Indexed local media asset with technical metadata and classification."""

    path: Path
    name: str
    category: str
    media_type: str  # "video", "audio", "image"
    duration_s: float | None
    width: int | None
    height: int | None
    fps: float | None
    aspect_ratio: str | None
    orientation: str | None  # "vertical", "horizontal", "square", or None
    file_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        return result


def classify_media_category(path: Path, media_type: str) -> str:
    """Classify an asset into standard media-lab categories based on path and type."""
    normalized_str = (str(path.parent.name) + "/" + path.stem).lower()

    if media_type == "audio":
        if any(k in normalized_str for k in _WHOOSH_KEYWORDS):
            return "sfx_whoosh"
        if any(k in normalized_str for k in _IMPACT_KEYWORDS):
            return "sfx_impact"
        if "click" in normalized_str:
            return "sfx_click"
        if "pop" in normalized_str:
            return "sfx_pop"
        if any(k in normalized_str for k in ("ding", "chime", "bell", "notification")):
            return "sfx_ding"
        if any(k in normalized_str for k in ("riser", "build", "tension")):
            return "sfx_riser"
        if "sfx" in normalized_str:
            return "sfx"
        return "audio"

    if media_type == "image":
        if any(k in normalized_str for k in ("background", "bg", "plate", "backdrop", "studio")):
            return "plate_still"
        return "image"

    if media_type == "video":
        if any(k in normalized_str for k in ("background", "bg", "plate", "backdrop")):
            return "plate_video"
        if any(k in normalized_str for k in ("broll", "cutaway", "overlay")):
            return "broll"
        return "broll"

    return "unknown"


def _calculate_aspect_ratio_string(width: int, height: int) -> str | None:
    if width <= 0 or height <= 0:
        return None
    ratio = width / height
    if 0.52 <= ratio <= 0.60:
        return "9:16"
    if 1.70 <= ratio <= 1.85:
        return "16:9"
    if 0.95 <= ratio <= 1.05:
        return "1:1"
    if 0.75 <= ratio <= 0.85:
        return "4:5"
    if 2.20 <= ratio <= 2.45:
        return "21:9"
    return f"{width}:{height}"


def _calculate_orientation(width: int, height: int) -> str | None:
    if width <= 0 or height <= 0:
        return None
    if height > width:
        return "vertical"
    if width > height:
        return "horizontal"
    return "square"


def scout_library(
    directories: Sequence[Path | str],
    config: Config,
    *,
    media_type: str | None = None,
    category: str | None = None,
    query: str | None = None,
    orientation: str | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    recursive: bool = True,
) -> list[ScoutedAsset]:
    """Search and index local media files according to filter criteria."""
    scouted: list[ScoutedAsset] = []

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            continue

        iterator = dir_path.rglob("*") if recursive else dir_path.glob("*")
        for item in iterator:
            if not item.is_file():
                continue
            # Skip hidden files/directories (e.g., .DS_Store, .git)
            if any(part.startswith(".") for part in item.parts):
                continue

            suffix = item.suffix.lower()
            if suffix not in ALL_SUPPORTED_EXTS:
                continue

            if suffix in SUPPORTED_AUDIO_EXTS:
                m_type = "audio"
            elif suffix in SUPPORTED_IMAGE_EXTS:
                m_type = "image"
            else:
                m_type = "video"

            # Quick preliminary media_type filter
            if media_type and media_type != m_type:
                continue

            # Quick query match on name/path
            if query and query.lower() not in str(item).lower():
                continue

            # File size
            try:
                file_size = item.stat().st_size
            except OSError:
                file_size = 0

            # Probe media info
            try:
                info: MediaInfo | None = probe(item, config)
            except (ProbeError, OSError):
                info = None

            asset_cat = classify_media_category(item, m_type)

            # Category filter (allow partial or exact matching, e.g. 'sfx' matches 'sfx_whoosh')
            if category:
                cat_lower = category.lower()
                if cat_lower != asset_cat.lower() and not asset_cat.lower().startswith(cat_lower):
                    continue

            dur_s = info.duration_s if info else None
            w = info.width if (info and info.width > 0) else None
            h = info.height if (info and info.height > 0) else None
            fps = info.fps if (info and info.fps > 0) else None

            orient = _calculate_orientation(w or 0, h or 0)
            if orientation and orient != orientation:
                continue

            # Duration filter
            if min_duration is not None and (dur_s is None or dur_s < min_duration):
                continue
            if max_duration is not None and (dur_s is not None and dur_s > max_duration):
                continue

            aspect = _calculate_aspect_ratio_string(w or 0, h or 0)

            asset = ScoutedAsset(
                path=item,
                name=item.name,
                category=asset_cat,
                media_type=m_type,
                duration_s=dur_s,
                width=w,
                height=h,
                fps=fps,
                aspect_ratio=aspect,
                orientation=orient,
                file_size_bytes=file_size,
            )
            scouted.append(asset)

    # Sort results deterministically by category then name
    scouted.sort(key=lambda a: (a.category, a.name))
    return scouted


def format_scouted_table(assets: Sequence[ScoutedAsset]) -> str:
    """Format scouted media assets as an ASCII table."""
    if not assets:
        return "No matching assets found in library."

    header = (
        f"{'Category':<14} | {'Name':<32} | {'Type':<6} | "
        f"{'Dim/Ratio':<10} | {'Duration':<8} | {'Size':<8}"
    )
    divider = "-" * len(header)
    lines = [header, divider]

    for a in assets:
        dur_str = f"{a.duration_s:.1f}s" if a.duration_s is not None else "-"
        dim_str = f"{a.width}x{a.height}" if (a.width and a.height) else (a.aspect_ratio or "-")
        size_mb = f"{a.file_size_bytes / (1024 * 1024):.2f}MB"
        lines.append(
            f"{a.category:<14} | {a.name[:32]:<32} | {a.media_type:<6} | "
            f"{dim_str:<10} | {dur_str:<8} | {size_mb:<8}"
        )

    return "\n".join(lines)
