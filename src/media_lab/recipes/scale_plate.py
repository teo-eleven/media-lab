"""Recipe for estimating subject scale and ground line from plate reference."""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..errors import MediaLabError
from ..scale_plate import ScaleEstimate, estimate_scale_from_reference, measure_silhouette_height


def estimate_plate_scale(
    source: Path | str,
    config: Config,
    *,
    ref_height: int,
    ref_ground_y: int,
    height_ratio: float = 1.0,
) -> ScaleEstimate:
    """Estimate subject scale and ground line Y matching reference plate dimensions.

    Args:
        source: Path to an input cutout image or directory of cutout frames.
        config: Project configuration.
        ref_height: Height in pixels of a reference person in the background plate.
        ref_ground_y: Ground contact line Y of the reference person.
        height_ratio: Ratio of subject real-world height vs reference person.
    """
    resolved_source = Path(source)
    if not resolved_source.is_absolute():
        resolved_source = (config.root / resolved_source).resolve()

    if not resolved_source.exists():
        raise MediaLabError(f"source path does not exist: {resolved_source}")

    subject_height = measure_silhouette_height(resolved_source)

    return estimate_scale_from_reference(
        subject_height=subject_height,
        ref_height=ref_height,
        ref_ground_y=ref_ground_y,
        height_ratio=height_ratio,
    )
