"""Colour matching and scene relighting for composite subjects.

Implements Reinhard statistical colour transfer in decorrelated (l, alpha, beta) space
plus directional key lighting, ambient tint/gamma, ground bounce, and contrast/saturation.
Preserves alpha channel transparency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from PIL import Image, ImageFilter

# Reinhard LMS conversion matrices
_RGB_TO_LMS = np.array(
    [
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444],
    ],
    dtype=np.float32,
)

_LMS_TO_LAB = np.array(
    [
        [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
        [1.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0)],
        [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0],
    ],
    dtype=np.float32,
)

_LAB_TO_LMS = np.array(
    [
        [np.sqrt(3.0) / 3.0, np.sqrt(6.0) / 6.0, np.sqrt(2.0) / 2.0],
        [np.sqrt(3.0) / 3.0, np.sqrt(6.0) / 6.0, -np.sqrt(2.0) / 2.0],
        [np.sqrt(3.0) / 3.0, -2.0 * np.sqrt(6.0) / 6.0, 0.0],
    ],
    dtype=np.float32,
)

_LMS_TO_RGB = np.array(
    [
        [4.4679, -3.5873, 0.1193],
        [-1.2186, 2.3809, -0.1624],
        [0.0497, -0.2439, 1.2045],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True, slots=True)
class RelightParams:
    """Scene lighting and colour adjustments."""

    ambient: tuple[float, float, float] = (1.0, 1.0, 1.0)
    bright: float = 1.0
    gamma: float = 1.0
    key_dir: tuple[float, float] = (0.0, 0.0)
    key_amt: float = 0.0
    bounce: tuple[float, float, float] = (0.5, 0.5, 0.5)
    bounce_amt: float = 0.0
    sat: float = 1.0
    contrast: float = 1.0
    blur: float = 0.0


def _rgb_to_lab(rgb_norm: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Convert normalized RGB (0-1) to Reinhard's (l, alpha, beta) space."""
    clipped = np.clip(rgb_norm, 1e-6, 1.0)
    lms = np.dot(clipped, _RGB_TO_LMS.T)
    log_lms = np.log10(np.clip(lms, 1e-6, None))
    return cast(np.ndarray[Any, Any], np.dot(log_lms, _LMS_TO_LAB.T))


def _lab_to_rgb(lab: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Convert Reinhard's (l, alpha, beta) space back to normalized RGB (0-1)."""
    log_lms = np.dot(lab, _LAB_TO_LMS.T)
    lms = 10.0**log_lms
    rgb = np.dot(lms, _LMS_TO_RGB.T)
    return cast(np.ndarray[Any, Any], np.clip(rgb, 0.0, 1.0))


def transfer_colour(
    source_rgba: Image.Image,
    target_ref: Image.Image,
    *,
    strength: float = 1.0,
) -> Image.Image:
    """Transfer the statistical colour mood of target_ref onto source_rgba."""
    if strength <= 0.0:
        return source_rgba

    src_arr = np.asarray(source_rgba.convert("RGBA")).astype(np.float32)
    tgt_arr = np.asarray(target_ref.convert("RGB")).astype(np.float32)

    src_rgb = src_arr[..., :3] / 255.0
    src_alpha = src_arr[..., 3]
    tgt_rgb = tgt_arr[..., :3] / 255.0

    mask = src_alpha > 10
    if not np.any(mask):
        return source_rgba

    src_lab = _rgb_to_lab(src_rgb)
    tgt_lab = _rgb_to_lab(tgt_rgb)

    src_pixels = src_lab[mask]
    src_mean = np.mean(src_pixels, axis=0)
    src_std = np.std(src_pixels, axis=0) + 1e-6

    tgt_pixels = tgt_lab.reshape(-1, 3)
    tgt_mean = np.mean(tgt_pixels, axis=0)
    tgt_std = np.std(tgt_pixels, axis=0) + 1e-6

    res_lab = src_lab.copy()
    norm = (src_lab - src_mean) * (tgt_std / src_std) + tgt_mean
    res_lab[mask] = (1.0 - strength) * src_lab[mask] + strength * norm[mask]

    res_rgb = (_lab_to_rgb(res_lab) * 255.0).astype(np.uint8)
    out_arr = np.dstack([res_rgb, src_alpha.astype(np.uint8)])
    return Image.fromarray(out_arr, "RGBA")


def apply_relight(
    image: Image.Image,
    params: RelightParams,
) -> Image.Image:
    """Apply directional key lighting, ambient tint, gamma, ground bounce, and contrast."""
    arr = np.asarray(image.convert("RGBA")).astype(np.float32)
    h, w = arr.shape[:2]
    rgb = arr[..., :3]

    # Ambient tint + brightness + gamma
    for c in range(3):
        rgb[..., c] *= params.ambient[c]
    rgb *= params.bright
    rgb = 255.0 * np.clip(rgb / 255.0, 0.0, 1.0) ** params.gamma

    # Directional key gradient
    kx, ky = params.key_dir
    if kx != 0.0 or ky != 0.0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx / max(1, w) - 0.5) * 2.0
        ny = (yy / max(1, h) - 0.5) * 2.0
        g = -(nx * kx + ny * ky)
        ptp = np.ptp(g)
        if ptp > 1e-6:
            g = (g - g.min()) / ptp
        grad = 1.0 + (g[..., None] - 0.5) * 2.0 * params.key_amt
        rgb *= grad

    # Ground bounce: tint the lower ~38% of the subject
    if params.bounce_amt > 0.0:
        yy = np.arange(h, dtype=np.float32)[:, None]
        by = np.clip((yy / max(1, h) - 0.62) / 0.38, 0.0, 1.0)[..., None]
        bcol = np.array(params.bounce, dtype=np.float32) * 255.0
        rgb = rgb * (1.0 - by * params.bounce_amt) + bcol * (by * params.bounce_amt)

    # Saturation and contrast
    if params.sat != 1.0 or params.contrast != 1.0:
        gray = rgb.mean(axis=2, keepdims=True)
        rgb = gray + (rgb - gray) * params.sat
        rgb = 128.0 + (rgb - 128.0) * params.contrast

    out = np.clip(np.dstack([np.clip(rgb, 0.0, 255.0), arr[..., 3]]), 0, 255).astype(np.uint8)
    res_img = Image.fromarray(out, "RGBA")

    if params.blur > 0.0:
        res_img = res_img.filter(ImageFilter.GaussianBlur(params.blur))

    return res_img
