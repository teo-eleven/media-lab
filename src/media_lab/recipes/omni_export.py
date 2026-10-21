"""Recipe: Omni Export — Enterprise Multi-Aspect Campaign Deliverables.

Generates synchronized campaign assets across all major social media aspect ratios
(9:16 vertical, 1:1 square, 16:9 widescreen, 4:5 portrait) from a single master video,
using smart face/subject tracking to prevent awkward crops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..errors import ValidationError
from ..paths import ensure_readable_source
from ..probe import MediaInfo
from .smart_reframe import smart_reframe

SUPPORTED_OMNI_ASPECTS: tuple[str, ...] = ("9:16", "1:1", "16:9", "4:5")


@dataclass(frozen=True, slots=True)
class OmniDeliverable:
    """A single rendered deliverable for an aspect ratio."""

    aspect: str
    path: Path
    media: MediaInfo


@dataclass(frozen=True, slots=True)
class OmniExportResult:
    """Outcome of generating multi-aspect campaign deliverables."""

    source: Path
    output_dir: Path
    deliverables: tuple[OmniDeliverable, ...]


def omni_export(
    source: Path | str,
    output_dir: Path | str,
    config: Config,
    *,
    aspects: tuple[str, ...] = SUPPORTED_OMNI_ASPECTS,
    mode: str = "smart",
    force: bool = False,
) -> OmniExportResult:
    """Batch export a master video into multiple platform-optimized aspect ratios.

    Args:
        source: Master input video.
        output_dir: Directory where deliverables will be written.
        config: System configuration.
        aspects: Tuple of target aspect ratios (e.g. ('9:16', '1:1', '16:9', '4:5')).
        mode: Reframe mode ('smart', 'center', 'split').
        force: Overwrite existing deliverables.

    Returns:
        OmniExportResult with all created deliverables and metadata.
    """
    resolved_source = ensure_readable_source(source)
    resolved_dir = Path(output_dir).expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)

    deliverables: list[OmniDeliverable] = []

    for aspect in aspects:
        if aspect not in SUPPORTED_OMNI_ASPECTS:
            raise ValidationError(
                f"Unsupported aspect ratio '{aspect}'. Must be one of {SUPPORTED_OMNI_ASPECTS}"
            )

        slug = aspect.replace(":", "x")
        out_filename = f"{resolved_source.stem}_{slug}{resolved_source.suffix}"
        out_path = resolved_dir / out_filename

        result = smart_reframe(
            resolved_source,
            out_path,
            config,
            target_aspect=aspect,
            mode=mode,
            force=force,
        )

        deliverables.append(
            OmniDeliverable(
                aspect=aspect,
                path=result.output,
                media=result.media,
            )
        )

    return OmniExportResult(
        source=resolved_source,
        output_dir=resolved_dir,
        deliverables=tuple(deliverables),
    )
