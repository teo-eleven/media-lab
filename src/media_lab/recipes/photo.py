"""Photo and carousel editing recipe for social media.

Provides single-photo and batch editing: aspect ratio framing (1:1, 4:5, 9:16, 16:9),
subject placement on new backdrops, Reinhard color transfer, contact shadow,
curated visual looks, and clarity sharpening.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ..colour_transfer import transfer_colour
from ..config import Config
from ..errors import MediaLabError, ValidationError
from ..grounding import detect_foot_point, render_contact_shadow
from ..kino import KinoRunner
from ..paths import ensure_readable_source, ensure_writable_output
from ..recipes.cutout import cut_out_person
from ..validation import check_choice

PHOTO_ASPECTS: Sequence[str] = ("original", "1:1", "4:5", "9:16", "16:9")
PHOTO_LOOKS: Sequence[str] = ("warm", "cool", "vintage", "noir", "vibrant", "punchy", "cinematic")
ALIGN_CHOICES: Sequence[str] = ("bottom", "center")

ASPECT_SIZES: dict[str, tuple[int, int]] = {
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True, slots=True)
class PhotoResult:
    """Outcome of one photo edit."""

    input_path: Path
    output_path: Path
    width: int
    height: int
    aspect: str


@dataclass(frozen=True, slots=True)
class BatchPhotoResult:
    """Outcome of batch photo processing."""

    results: tuple[PhotoResult, ...]
    output_dir: Path


def _cover_crop(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scale and center-crop image to completely fill target dimensions."""
    orig_w, orig_h = image.size
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def _apply_photo_look(image: Image.Image, look: str) -> Image.Image:
    """Apply a visual look to an RGB/RGBA image."""
    has_alpha = image.mode == "RGBA"
    alpha: Image.Image | None = None
    if has_alpha:
        alpha = image.getchannel("A")
        rgb_img = image.convert("RGB")
    else:
        rgb_img = image.convert("RGB")

    arr = np.array(rgb_img, dtype=np.float32)

    if look == "warm":
        arr[..., 0] = np.clip(arr[..., 0] * 1.07, 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] * 0.93, 0, 255)
        out_img = Image.fromarray(arr.astype(np.uint8))
    elif look == "cool":
        arr[..., 0] = np.clip(arr[..., 0] * 0.93, 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] * 1.08, 0, 255)
        out_img = Image.fromarray(arr.astype(np.uint8))
    elif look == "vintage":
        arr[..., 0] = np.clip(arr[..., 0] * 1.05 + 10, 0, 255)
        arr[..., 1] = np.clip(arr[..., 1] * 1.02 + 5, 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] * 0.90 + 15, 0, 255)
        out_img = Image.fromarray(arr.astype(np.uint8))
        out_img = ImageEnhance.Contrast(out_img).enhance(0.9)
    elif look == "noir":
        gray = rgb_img.convert("L")
        gray_enhanced = ImageEnhance.Contrast(gray).enhance(1.25)
        out_img = gray_enhanced.convert("RGB")
    elif look == "vibrant":
        out_img = ImageEnhance.Color(rgb_img).enhance(1.35)
        out_img = ImageEnhance.Contrast(out_img).enhance(1.05)
    elif look == "punchy":
        out_img = ImageEnhance.Contrast(rgb_img).enhance(1.20)
        out_img = ImageEnhance.Color(out_img).enhance(1.15)
    elif look == "cinematic":
        # Teal & orange mood shift
        arr[..., 0] = np.clip(arr[..., 0] * 1.05, 0, 255)
        arr[..., 1] = np.clip(arr[..., 1] * 0.98, 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] * 0.92, 0, 255)
        out_img = Image.fromarray(arr.astype(np.uint8))
        out_img = ImageEnhance.Contrast(out_img).enhance(1.12)
    else:
        out_img = rgb_img

    if has_alpha and alpha is not None:
        out_img = out_img.convert("RGBA")
        out_img.putalpha(alpha)
    return out_img


def edit_photo(
    source: Path | str,
    output: Path | str,
    config: Config,
    runner: KinoRunner | None = None,
    *,
    aspect: str = "original",
    bg: Path | str | None = None,
    cutout: bool = False,
    colour_match: bool = True,
    shadow: bool = True,
    align: str = "bottom",
    subject_scale: float = 0.85,
    look: str | None = None,
    sharpen: bool = False,
    force: bool = False,
) -> PhotoResult:
    """Edit a single photo with framing, background replacement, looks, and clarity."""
    check_choice(aspect, PHOTO_ASPECTS, "aspect")
    check_choice(align, ALIGN_CHOICES, "align")
    if look is not None:
        check_choice(look, PHOTO_LOOKS, "look")

    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    # Cutout subject if requested and runner is available
    working_source = resolved_source
    if cutout:
        if runner is None:
            raise ValidationError("cutout requires KinoRunner to be provided")
        tmp_cutout = resolved_output.parent / f"{resolved_source.stem}_cutout.png"
        cut_out_person(resolved_source, tmp_cutout, config, runner, force=True)
        working_source = tmp_cutout

    with Image.open(working_source) as src_im:
        subject_img = src_im.convert("RGBA" if (cutout or bg or src_im.mode == "RGBA") else "RGB")

    # Determine canvas size
    if aspect == "original":
        canvas_w, canvas_h = subject_img.size
    else:
        canvas_w, canvas_h = ASPECT_SIZES[aspect]

    if bg is not None:
        resolved_bg = ensure_readable_source(bg)
        with Image.open(resolved_bg) as bg_im:
            canvas = _cover_crop(bg_im.convert("RGB"), canvas_w, canvas_h)

        # Scale subject to requested fraction of canvas height
        subj_w, subj_h = subject_img.size
        target_subj_h = int(round(canvas_h * subject_scale))
        scale = target_subj_h / subj_h
        target_subj_w = int(round(subj_w * scale))
        scaled_subject = subject_img.resize(
            (target_subj_w, target_subj_h), Image.Resampling.LANCZOS
        )

        # Position on canvas
        pos_x = (canvas_w - target_subj_w) // 2
        pos_y = canvas_h - target_subj_h if align == "bottom" else (canvas_h - target_subj_h) // 2

        # Optional Reinhard color match with background
        if colour_match and scaled_subject.mode == "RGBA":
            scaled_subject = transfer_colour(scaled_subject, canvas, strength=0.85)

        # Optional contact shadow
        if shadow and scaled_subject.mode == "RGBA":
            foot = detect_foot_point(scaled_subject)
            if foot is not None:
                ground_x = foot.fx + pos_x
                ground_y = foot.fy + pos_y
                shadow_layer = render_contact_shadow(
                    (canvas_w, canvas_h),
                    ground_x,
                    ground_y,
                    opacity=0.85,
                )
                canvas.paste(shadow_layer, (0, 0), shadow_layer.getchannel("A"))

        # Paste subject over canvas
        if scaled_subject.mode == "RGBA":
            canvas.paste(scaled_subject, (pos_x, pos_y), scaled_subject.getchannel("A"))
        else:
            canvas.paste(scaled_subject, (pos_x, pos_y))

        final_img = canvas
    else:
        if aspect != "original":
            final_img = _cover_crop(subject_img, canvas_w, canvas_h)
        else:
            final_img = subject_img.copy()

    # Apply look
    if look is not None:
        final_img = _apply_photo_look(final_img, look)

    # Apply unsharp mask sharpening
    if sharpen:
        final_img = final_img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))

    # Convert to RGB if target is JPEG
    if resolved_output.suffix.lower() in (".jpg", ".jpeg"):
        final_img = final_img.convert("RGB")
        final_img.save(resolved_output, quality=95, optimize=True)
    elif resolved_output.suffix.lower() == ".webp":
        final_img.save(resolved_output, quality=95, method=6)
    else:
        final_img.save(resolved_output)

    return PhotoResult(
        input_path=resolved_source,
        output_path=resolved_output,
        width=final_img.size[0],
        height=final_img.size[1],
        aspect=aspect,
    )


def process_photo_batch(
    source_dir: Path | str,
    output_dir: Path | str,
    config: Config,
    runner: KinoRunner | None = None,
    *,
    aspect: str = "4:5",
    bg: Path | str | None = None,
    look: str | None = None,
    sharpen: bool = False,
    force: bool = False,
) -> BatchPhotoResult:
    """Process a directory of photos with uniform aspect framing and styling."""
    src_dir = Path(source_dir)
    if not src_dir.is_absolute():
        src_dir = (config.root / src_dir).resolve()
    if not src_dir.is_dir():
        raise MediaLabError(f"source directory does not exist: {src_dir}")

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = (config.root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise MediaLabError(f"no images found in {src_dir}")

    results = []
    for img in images:
        target = out_dir / img.name
        res = edit_photo(
            img,
            target,
            config,
            runner,
            aspect=aspect,
            bg=bg,
            look=look,
            sharpen=sharpen,
            force=force,
        )
        results.append(res)

    return BatchPhotoResult(results=tuple(results), output_dir=out_dir)
