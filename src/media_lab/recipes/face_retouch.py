"""Recipe: portrait retouching, skin smoothing, and depth-of-field background blur.

Provides edge-preserving skin softening (bilateral surface blur with chrominance skin masking)
and portrait bokeh / defocus background blur.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..config import Config
from ..errors import ValidationError
from ..paths import ensure_readable_source, ensure_writable_output


@dataclass(frozen=True, slots=True)
class FaceRetouchResult:
    """Outcome of portrait retouching."""

    output: Path
    width: int
    height: int
    skin_smoothed: bool
    depth_blur_applied: bool


def detect_skin_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Detect skin pixels using YCrCb chrominance bounds."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    _y, cr, cb = cv2.split(ycrcb)

    # Standard skin chrominance cluster: Cb in [77, 127], Cr in [133, 173]
    skin_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
    return (skin_mask.astype(np.uint8)) * 255


def smooth_skin_layer(
    img_bgr: np.ndarray,
    strength: float = 0.6,
    feather_radius: int = 11,
) -> np.ndarray:
    """Apply edge-preserving bilateral smoothing strictly to skin regions."""
    if strength <= 0.0:
        return img_bgr

    skin_mask = detect_skin_mask(img_bgr)
    if np.count_nonzero(skin_mask) == 0:
        return img_bgr

    # Feather the skin mask so transitions are soft and natural
    ksize = feather_radius if feather_radius % 2 == 1 else feather_radius + 1
    feathered_mask = cv2.GaussianBlur(skin_mask.astype(np.float32) / 255.0, (ksize, ksize), 0)
    feathered_mask = np.clip(feathered_mask * strength, 0.0, 1.0)
    alpha = np.expand_dims(feathered_mask, axis=-1)

    # Bilateral filter preserves edges (eyes, teeth, glasses) while smoothing skin texture
    sigma_color = int(40 + (strength * 40))
    sigma_space = int(30 + (strength * 40))
    smoothed = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    result = (img_bgr.astype(np.float32) * (1.0 - alpha)) + (smoothed.astype(np.float32) * alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_depth_blur(
    img_bgr: np.ndarray,
    person_mask: np.ndarray | None = None,
    blur_sigma: float = 15.0,
) -> np.ndarray:
    """Simulate shallow depth-of-field bokeh by blurring background behind the subject."""
    if blur_sigma <= 0.0:
        return img_bgr

    h, w = img_bgr.shape[:2]

    # Large bokeh blur on entire canvas
    ksize = int(blur_sigma * 3)
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    ksize = max(3, ksize)
    bg_blurred = cv2.GaussianBlur(img_bgr, (ksize, ksize), blur_sigma)

    if person_mask is not None:
        # Resize person mask if needed
        if person_mask.shape[:2] != (h, w):
            p_mask = cv2.resize(person_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            p_mask = person_mask
        p_mask_f = p_mask.astype(np.float32) / 255.0
        # Soft feather edge
        p_mask_blurred = cv2.GaussianBlur(p_mask_f, (9, 9), 0)
        alpha = np.expand_dims(np.clip(p_mask_blurred, 0.0, 1.0), axis=-1)
    else:
        # Vignette / radial depth falloff if no explicit silhouette mask given
        # Center region sharp, perimeter blurred
        cy, cx = h / 2.0, w / 2.0
        y_coords, x_coords = np.ogrid[:h, :w]
        dist = np.sqrt(((x_coords - cx) / (w / 2.0)) ** 2 + ((y_coords - cy) / (h / 2.0)) ** 2)
        # Subject is in center (dist <= 0.5 has alpha 1.0, falls off to 0.0 at dist >= 1.0)
        falloff = np.clip(1.0 - (dist - 0.4) / 0.6, 0.0, 1.0)
        alpha = np.expand_dims(falloff.astype(np.float32), axis=-1)

    result = (img_bgr.astype(np.float32) * alpha) + (bg_blurred.astype(np.float32) * (1.0 - alpha))
    ret: np.ndarray = np.clip(result, 0, 255).astype(np.uint8)
    return ret


def retouch_portrait(
    source: Path | str,
    output: Path | str,
    config: Config,
    *,
    smooth_skin: bool = True,
    skin_strength: float = 0.5,
    depth_blur: bool = False,
    blur_sigma: float = 12.0,
    mask_path: Path | str | None = None,
    radiance: float = 0.0,
    force: bool = False,
) -> FaceRetouchResult:
    """Apply portrait retouching: skin smoothing, portrait depth bokeh, and tone radiance.

    Args:
        source: Input portrait image.
        output: Destination image path.
        config: System configuration.
        smooth_skin: Enable skin smoothing.
        skin_strength: Smoothing strength [0.0, 1.0].
        depth_blur: Enable background depth blur.
        blur_sigma: Gaussian blur radius for background bokeh.
        mask_path: Optional silhouette/subject mask for depth blur separation.
        radiance: Subtle skin radiance lift [0.0, 1.0].
        force: Allow overwriting existing output.

    Returns:
        FaceRetouchResult with processing flags.
    """
    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    if not (0.0 <= skin_strength <= 1.0):
        raise ValidationError(f"skin_strength must be between 0.0 and 1.0, got {skin_strength}")
    if blur_sigma < 0.0 or blur_sigma > 60.0:
        raise ValidationError(f"blur_sigma must be between 0.0 and 60.0, got {blur_sigma}")
    if not (0.0 <= radiance <= 1.0):
        raise ValidationError(f"radiance must be between 0.0 and 1.0, got {radiance}")

    img_bgr = cv2.imread(str(resolved_source), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValidationError(f"could not read image from {resolved_source}")

    h, w = img_bgr.shape[:2]
    current = img_bgr

    # 1. Depth Bokeh Blur
    if depth_blur:
        p_mask: np.ndarray | None = None
        if mask_path is not None:
            resolved_mask = ensure_readable_source(mask_path)
            with Image.open(resolved_mask) as m_img:
                m_arr = np.array(m_img.convert("L").resize((w, h), Image.Resampling.NEAREST))
                p_mask = m_arr
        current = apply_depth_blur(current, person_mask=p_mask, blur_sigma=blur_sigma)

    # 2. Skin Smoothing
    if smooth_skin:
        current = smooth_skin_layer(current, strength=skin_strength)

    # 3. Radiance / Warm Luminance Lift
    if radiance > 0.0:
        # Lift luminance slightly in Lab space
        lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel += radiance * 15.0  # Mild exposure lift
        b_channel += radiance * 4.0  # Warm golden skin undertone
        merged_lab = cv2.merge(
            [np.clip(l_channel, 0, 255), np.clip(a_channel, 0, 255), np.clip(b_channel, 0, 255)]
        )
        current = cv2.cvtColor(merged_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Save output
    out_ext = resolved_output.suffix.lower()
    if out_ext in {".jpg", ".jpeg"}:
        cv2.imwrite(str(resolved_output), current, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        cv2.imwrite(str(resolved_output), current)

    if not resolved_output.is_file() or resolved_output.stat().st_size == 0:
        raise ValidationError(f"failed to produce retouched file at {resolved_output}")

    return FaceRetouchResult(
        output=resolved_output,
        width=w,
        height=h,
        skin_smoothed=smooth_skin,
        depth_blur_applied=depth_blur,
    )
