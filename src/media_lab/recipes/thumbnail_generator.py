"""Recipe: Smart Viral Thumbnail Generator (MrBeast / YouTube Style).

Generates high-retention covers with:
1. Sharp frame extraction from video source.
2. Subject cutout with alpha channel.
3. Neon/fluorescent stroke outline with soft Gaussian glow halo.
4. Contrast-tuned bokeh backdrop.
5. High-impact kinetic typography title badge.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..config import Config
from ..ffmpeg import run_ffmpeg
from ..ml_runner import MlRunner
from ..paths import ensure_readable_source, ensure_writable_output, work_directory
from ..probe import probe
from .cutout import cut_out_person

STROKE_PALETTE: dict[str, tuple[int, int, int]] = {
    "yellow": (255, 230, 0),
    "cyan": (0, 240, 255),
    "white": (255, 255, 255),
    "red": (255, 45, 85),
    "green": (57, 255, 20),
    "magenta": (255, 0, 128),
}


@dataclass(frozen=True, slots=True)
class ThumbnailResult:
    """Outcome of generating a viral thumbnail."""

    output: Path
    width: int
    height: int
    title: str | None
    style: str


def create_glow_outline(
    rgba_image: Image.Image,
    color: str = "yellow",
    stroke_width: int = 8,
    glow: bool = True,
) -> Image.Image:
    """Create a colored neon stroke and blurred halo around the alpha mask."""
    if rgba_image.mode != "RGBA":
        rgba_image = rgba_image.convert("RGBA")

    width, height = rgba_image.size
    alpha = rgba_image.split()[3]

    rgb_color = STROKE_PALETTE.get(color.lower(), (255, 230, 0))

    # Dilate alpha mask to expand border
    # MaxFilter size must be odd
    filter_size = max(3, stroke_width * 2 + 1)
    if filter_size % 2 == 0:
        filter_size += 1

    dilated_alpha = alpha.filter(ImageFilter.MaxFilter(filter_size))

    # Create solid color layer tinted by dilated alpha
    stroke_layer = Image.new("RGBA", (width, height), (*rgb_color, 255))
    stroke_layer.putalpha(dilated_alpha)

    if not glow:
        return stroke_layer

    # Create outer glow halo
    glow_alpha = dilated_alpha.filter(ImageFilter.GaussianBlur(radius=stroke_width * 1.5))
    glow_layer = Image.new("RGBA", (width, height), (*rgb_color, 200))
    glow_layer.putalpha(glow_alpha)

    # Composite stroke over halo
    glow_layer.alpha_composite(stroke_layer)
    return glow_layer


def generate_viral_thumbnail(
    source: Path | str,
    output: Path | str,
    config: Config,
    ml_runner: MlRunner,
    *,
    title: str | None = None,
    style: str = "viral",
    stroke_color: str = "yellow",
    stroke_width: int = 8,
    glow: bool = True,
    target_aspect: str = "16:9",
    timestamp_s: float | None = None,
    bg_image: Path | str | None = None,
    force: bool = False,
) -> ThumbnailResult:
    """Generate high-CTR social media cover image."""
    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    canvas_w, canvas_h = (1280, 720) if target_aspect == "16:9" else (1080, 1920)

    work = work_directory(config, "thumbnail_gen")
    src_info = probe(resolved_source, config)

    # 1. Extract optimal frame
    frame_path = work / f"{resolved_source.stem}_frame.png"
    if src_info.has_video:
        dur = src_info.duration_s or 1.0
        ts = timestamp_s if timestamp_s is not None else min(1.0, dur * 0.25)
        extract_cmd = [
            "-ss",
            f"{ts:.2f}",
            "-i",
            str(resolved_source),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ]
        run_ffmpeg(extract_cmd, config)
    else:
        # Already an image
        frame_path = resolved_source

    # 2. Build Backdrop Canvas
    if bg_image:
        bg_p = Path(bg_image)
        if not bg_p.is_absolute():
            bg_p = (config.root / bg_p).resolve()
        base_bg = Image.open(bg_p).convert("RGBA")
    else:
        # Blur extracted frame for high contrast bokeh
        raw_bg = Image.open(frame_path).convert("RGBA")
        raw_bg = raw_bg.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        base_bg = raw_bg.filter(ImageFilter.GaussianBlur(radius=20))

    canvas = base_bg.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)

    # 3. Subject Cutout & Neon Glow
    cutout_path = work / f"{resolved_source.stem}_cutout.png"
    subject_applied = False
    try:
        from ..kino import KinoRunner

        runner = KinoRunner.from_config(config)
        cut_out_person(frame_path, cutout_path, config, runner, force=True)
        if cutout_path.exists():
            with Image.open(cutout_path) as cut_img:
                cut_rgba = cut_img.convert("RGBA")
                # Scale subject to fit nicely on canvas (~80% height)
                sub_h = int(canvas_h * 0.85)
                scale = sub_h / max(1, cut_rgba.height)
                sub_w = int(cut_rgba.width * scale)
                cut_resized = cut_rgba.resize((sub_w, sub_h), Image.Resampling.LANCZOS)

                # Generate glow
                if stroke_color != "none" and stroke_width > 0:
                    glow_img = create_glow_outline(
                        cut_resized, color=stroke_color, stroke_width=stroke_width, glow=glow
                    )
                    # Position subject centered-right
                    pos_x = (
                        canvas_w - sub_w - int(canvas_w * 0.05)
                        if target_aspect == "16:9"
                        else (canvas_w - sub_w) // 2
                    )
                    pos_y = canvas_h - sub_h
                    canvas.alpha_composite(glow_img, (pos_x, pos_y))
                    canvas.alpha_composite(cut_resized, (pos_x, pos_y))
                else:
                    pos_x = (
                        canvas_w - sub_w - int(canvas_w * 0.05)
                        if target_aspect == "16:9"
                        else (canvas_w - sub_w) // 2
                    )
                    pos_y = canvas_h - sub_h
                    canvas.alpha_composite(cut_resized, (pos_x, pos_y))

                subject_applied = True
    except Exception:
        pass

    if not subject_applied:
        # Fallback: paste sharp original frame directly in center/fill
        with Image.open(frame_path) as raw_frame:
            scaled_frame = raw_frame.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS).convert(
                "RGBA"
            )
            canvas.paste(scaled_frame, (0, 0))

    # 4. Bold Typography Badge Overlay
    if title:
        draw = ImageDraw.Draw(canvas)
        text_str = title.upper()
        font_size = int(canvas_h * 0.09)

        font: ImageFont.ImageFont | ImageFont.FreeTypeFont
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except OSError:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except OSError:
                font = ImageFont.load_default()

        # Bounding box of text
        bbox = draw.textbbox((0, 0), text_str, font=font)
        t_w = int(bbox[2] - bbox[0])
        t_h = int(bbox[3] - bbox[1])

        pad_x = int(font_size * 0.4)
        pad_y = int(font_size * 0.25)

        # Place title on left side for 16:9, or top-center for 9:16
        if target_aspect == "16:9":
            pos_x = int(canvas_w * 0.08)
            pos_y = int(canvas_h * 0.40)
        else:
            pos_x = int((canvas_w - (t_w + pad_x * 2)) // 2)
            pos_y = int(canvas_h * 0.15)

        badge_rect = [
            pos_x - pad_x,
            pos_y - pad_y,
            pos_x + t_w + pad_x,
            pos_y + t_h + pad_y,
        ]

        # Draw translucent background pill
        badge_overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        b_draw = ImageDraw.Draw(badge_overlay)
        b_draw.rounded_rectangle(
            badge_rect, radius=18, fill=(0, 0, 0, 200), outline=(255, 230, 0, 240), width=3
        )
        canvas.alpha_composite(badge_overlay)

        # Draw text with shadow
        draw = ImageDraw.Draw(canvas)
        draw.text((pos_x + 3, pos_y + 3), text_str, fill=(0, 0, 0, 255), font=font)
        draw.text((pos_x, pos_y), text_str, fill=(255, 255, 255, 255), font=font)

    # 5. Final Save (RGB JPEG/PNG)
    final_img = canvas.convert("RGB")
    final_img.save(resolved_output, quality=95)

    return ThumbnailResult(
        output=resolved_output,
        width=canvas_w,
        height=canvas_h,
        title=title,
        style=style,
    )
