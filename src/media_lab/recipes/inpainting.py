"""Recipe: local object inpainting and background reconstruction.

Erases unwanted objects, timestamps, blemishes, or watermarks from images or video frames
using content-aware partial-differential-equation inpainting (Telea or Navier-Stokes).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from PIL import Image

from ..config import Config
from ..errors import ValidationError
from ..paths import ensure_readable_source, ensure_writable_output

VALID_METHODS: Final[frozenset[str]] = frozenset({"telea", "ns"})
IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


@dataclass(frozen=True, slots=True)
class InpaintResult:
    """Outcome of an inpainting operation."""

    output: Path
    width: int
    height: int
    erased_pixels: int
    method: str


def create_bbox_mask(width: int, height: int, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Create a single-channel binary uint8 mask for a bounding box (x, y, w, h)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    bx, by, bw, bh = bbox

    x1 = max(0, min(width - 1, bx))
    y1 = max(0, min(height - 1, by))
    x2 = max(0, min(width, bx + bw))
    y2 = max(0, min(height, by + bh))

    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return mask


def inpaint_image(
    source: Path | str,
    output: Path | str,
    config: Config,
    *,
    bbox: tuple[int, int, int, int] | None = None,
    mask_path: Path | str | None = None,
    method: str = "telea",
    inpaint_radius: int = 5,
    force: bool = False,
) -> InpaintResult:
    """Erase unwanted regions from an image using content-aware inpainting.

    Args:
        source: Input image path.
        output: Destination image path.
        config: System configuration.
        bbox: Bounding box (x, y, width, height) of the region to erase.
        mask_path: Path to binary or grayscale mask (white pixels = erase).
        method: Inpainting algorithm ('telea' or 'ns').
        inpaint_radius: Neighborhood radius in pixels for texture propagation.
        force: Allow overwriting existing output.

    Returns:
        InpaintResult with dimensions and count of erased pixels.
    """
    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    if method not in VALID_METHODS:
        raise ValidationError(
            f"unsupported inpainting method {method!r}, supported: {sorted(VALID_METHODS)}"
        )

    if bbox is None and mask_path is None:
        raise ValidationError("must provide either bbox or mask_path for inpainting")

    if inpaint_radius < 1 or inpaint_radius > 50:
        raise ValidationError(f"inpaint_radius must be between 1 and 50, got {inpaint_radius}")

    # Read image
    img_bgr = cv2.imread(str(resolved_source), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValidationError(f"could not read image from {resolved_source}")

    h, w = img_bgr.shape[:2]

    # Build or load mask
    if bbox is not None:
        bx, by, bw, bh = bbox
        if bx < 0 or by < 0 or bw <= 0 or bh <= 0:
            raise ValidationError(f"invalid bbox dimensions {bbox}")
        if bx >= w or by >= h:
            raise ValidationError(f"bbox origin ({bx}, {by}) is outside image bounds ({w}x{h})")
        mask = create_bbox_mask(w, h, bbox)
    else:
        assert mask_path is not None
        resolved_mask = ensure_readable_source(mask_path)
        with Image.open(resolved_mask) as mask_img:
            resized_mask = mask_img.convert("L").resize((w, h), Image.Resampling.NEAREST)
            mask = np.array(resized_mask, dtype=np.uint8)
            # Threshold to clean binary 0 or 255
            mask = np.where(mask > 128, 255, 0).astype(np.uint8)

    erased_count = int(np.count_nonzero(mask))
    if erased_count == 0:
        raise ValidationError("inpaint mask contains zero pixels to erase")

    # Slight dilation (1px) to guarantee clean edge coverage
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    cv_method = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    inpainted = cv2.inpaint(img_bgr, dilated_mask, inpaint_radius, cv_method)

    # Save output
    out_ext = resolved_output.suffix.lower()
    if out_ext in {".jpg", ".jpeg"}:
        cv2.imwrite(str(resolved_output), inpainted, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        cv2.imwrite(str(resolved_output), inpainted)

    if not resolved_output.is_file() or resolved_output.stat().st_size == 0:
        raise ValidationError(f"failed to produce inpainted file at {resolved_output}")

    return InpaintResult(
        output=resolved_output,
        width=w,
        height=h,
        erased_pixels=erased_count,
        method=method,
    )
