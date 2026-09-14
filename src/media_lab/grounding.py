"""Geometric grounding, foot-pinning, zoom-normalisation and contact shadow baking.

Locks a subject's silhouette to the ground plane, normalises subtle camera zoom
variations across frames, and bakes a multi-layer directional contact shadow
so the subject reads as standing firmly on the ground rather than floating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True, slots=True)
class FootPoint:
    """Foot contact coordinates and silhouette height for one frame."""

    fx: int
    fy: int
    height: int


@dataclass(frozen=True, slots=True)
class FrameTransform:
    """Calculated position and scaling for placing a subject on canvas."""

    fx: int
    fy: int
    scale: float
    pos_x: int
    pos_y: int
    ground_x: int
    ground_y: int


def detect_foot_point(
    image: Image.Image,
    *,
    alpha_threshold: int = 40,
    band_height: int = 80,
) -> FootPoint | None:
    """Find the contact point (lowest opaque row and bottom centroid) of a silhouette.

    Returns None if no pixels exceed alpha_threshold.
    """
    arr = np.asarray(image.convert("RGBA"))
    alpha = arr[..., 3]
    ys, xs = np.where(alpha > alpha_threshold)
    if len(ys) == 0:
        return None

    fy = int(ys.max())
    ty = int(ys.min())
    height = fy - ty

    band = ys > (fy - band_height)
    fx = int(np.median(xs[band])) if np.any(band) else int(np.median(xs))
    return FootPoint(fx=fx, fy=fy, height=height)


def compute_grounding_transforms(
    foot_points: list[FootPoint | None],
    *,
    canvas_size: tuple[int, int] = (2160, 3840),
    ground_y: int = 3560,
    dx: int = 0,
    base_scale: float = 1.0,
    zoom_normalise: bool = True,
) -> list[FrameTransform]:
    """Compute per-frame scale and placement coordinates.

    Uses median filtering to lock feet to the ground plane and smooth zoom deltas.
    """
    canvas_w, _canvas_h = canvas_size
    valid = [fp for fp in foot_points if fp is not None]
    if not valid:
        default_fy = 1000
        med_h = 1000.0
    else:
        default_fy = int(np.median([fp.fy for fp in valid]))
        med_h = float(np.median([fp.height for fp in valid]))

    transforms: list[FrameTransform] = []
    ground_x = int(canvas_w / 2 + dx)

    for fp in foot_points:
        if fp is None:
            cur_fx = canvas_w // 4
            cur_fy = default_fy
            z = 1.0
        else:
            cur_fx = fp.fx
            cur_fy = default_fy  # lock base to median foot height
            if zoom_normalise and fp.height > 0:
                raw_z = med_h / float(fp.height)
                z = max(0.90, min(1.10, raw_z))
            else:
                z = 1.0

        effective_scale = base_scale * z
        scaled_fx = cur_fx * effective_scale
        scaled_fy = cur_fy * effective_scale

        pos_x = int(canvas_w / 2 - scaled_fx + dx)
        pos_y = int(ground_y - scaled_fy)

        transforms.append(
            FrameTransform(
                fx=cur_fx,
                fy=cur_fy,
                scale=effective_scale,
                pos_x=pos_x,
                pos_y=pos_y,
                ground_x=ground_x,
                ground_y=ground_y,
            )
        )

    return transforms


def render_contact_shadow(
    canvas_size: tuple[int, int],
    ground_x: int,
    ground_y: int,
    *,
    sun_dir: tuple[float, float] = (0.0, 1.0),
    opacity: float = 1.0,
) -> Image.Image:
    """Render a 3-layer directional contact shadow on a transparent canvas.

    Layer 1: Directional soft falloff ellipse offset by sun vector.
    Layer 2: Medium contact falloff ellipse.
    Layer 3: Tight ambient occlusion directly under the feet.
    """
    shadow = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    if opacity <= 0.0:
        return shadow

    clamped_opacity = max(0.0, min(1.0, opacity))
    sdx, _sdy = sun_dir
    off_x = int(-sdx * 260)

    # Layer 1: Wide directional shadow
    l1 = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    d1 = ImageDraw.Draw(l1)
    a1 = int(70 * clamped_opacity)
    d1.ellipse(
        [ground_x - 260 + off_x, ground_y - 24, ground_x + 260 + off_x, ground_y + 90],
        fill=(0, 0, 0, a1),
    )
    l1 = l1.filter(ImageFilter.GaussianBlur(28))

    # Layer 2: Medium contact
    l2 = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    d2 = ImageDraw.Draw(l2)
    a2 = int(120 * clamped_opacity)
    d2.ellipse(
        [ground_x - 150, ground_y - 8, ground_x + 150, ground_y + 44],
        fill=(0, 0, 0, a2),
    )
    l2 = l2.filter(ImageFilter.GaussianBlur(10))

    # Layer 3: Tight ambient occlusion
    l3 = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    d3 = ImageDraw.Draw(l3)
    a3 = int(150 * clamped_opacity)
    d3.ellipse(
        [ground_x - 95, ground_y + 2, ground_x + 95, ground_y + 30],
        fill=(0, 0, 0, a3),
    )
    l3 = l3.filter(ImageFilter.GaussianBlur(4))

    base = Image.alpha_composite(l1, l2)
    return Image.alpha_composite(base, l3)
