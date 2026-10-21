"""Scale estimation from plate reference.

Matches a subject's scale and ground line to people or markers already present
in the background plate, preventing scale mismatch ("walking over people").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import MediaLabError, ValidationError
from .grounding import detect_foot_point


@dataclass(frozen=True, slots=True)
class ScaleEstimate:
    """Estimated scale and ground contact coordinate."""

    base_scale: float
    ground_y: int
    subject_height_px: int
    target_height_px: int


def estimate_scale_from_reference(
    subject_height: int,
    ref_height: int,
    ref_ground_y: int,
    *,
    height_ratio: float = 1.0,
) -> ScaleEstimate:
    """Calculate subject scale and ground line matching a plate reference person.

    Args:
        subject_height: Height of the unscaled cutout silhouette in pixels.
        ref_height: Height in pixels of a reference person at the target depth.
        ref_ground_y: Ground contact line Y of the reference person.
        height_ratio: Real-world height ratio of subject / reference (1.0 = equal).
    """
    if subject_height <= 0:
        raise ValidationError(f"subject_height must be positive, got {subject_height}")
    if ref_height <= 0:
        raise ValidationError(f"ref_height must be positive, got {ref_height}")
    if ref_ground_y <= 0:
        raise ValidationError(f"ref_ground_y must be positive, got {ref_ground_y}")
    if height_ratio <= 0.0:
        raise ValidationError(f"height_ratio must be positive, got {height_ratio}")

    target_height = int(ref_height * height_ratio)
    scale = target_height / float(subject_height)

    return ScaleEstimate(
        base_scale=scale,
        ground_y=ref_ground_y,
        subject_height_px=subject_height,
        target_height_px=target_height,
    )


def measure_silhouette_height(source_path: Path) -> int:
    """Measure the median height of a subject silhouette from an image or frame directory."""
    if source_path.is_file():
        im = Image.open(source_path).convert("RGBA")
        fp = detect_foot_point(im)
        if fp is None or fp.height <= 0:
            raise MediaLabError(f"no opaque subject silhouette detected in {source_path}")
        return fp.height

    if source_path.is_dir():
        frames = sorted(source_path.glob("f-*.png")) or sorted(source_path.glob("*.png"))
        if not frames:
            raise MediaLabError(f"no PNG frames found in directory: {source_path}")
        heights: list[int] = []
        for f in frames:
            im = Image.open(f).convert("RGBA")
            fp = detect_foot_point(im)
            if fp is not None and fp.height > 0:
                heights.append(fp.height)

        if not heights:
            raise MediaLabError(f"no opaque silhouettes detected in frames under {source_path}")
        return int(np.median(heights))

    raise MediaLabError(f"source path does not exist: {source_path}")
