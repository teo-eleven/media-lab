"""Recipe: typography & graphical title badges for images and videos.

Renders social-media style title cards, highlighted badge capsules, and lower-thirds
with automatic text wrapping, custom fonts, stroke, drop shadow, and pill backgrounds.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

from ..config import Config
from ..errors import ValidationError
from ..ffmpeg import run_ffmpeg
from ..paths import ensure_readable_source, ensure_writable_output, work_path
from ..probe import probe
from ..verify import Expectations, verify_render

IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
VIDEO_EXTENSIONS: Final[frozenset[str]] = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})

VALID_POSITIONS: Final[frozenset[str]] = frozenset(
    {"top", "center", "bottom", "top-left", "top-right", "bottom-left", "bottom-right"}
)

FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "Arial.ttf",
    "DejaVuSans-Bold.ttf",
)


@dataclass(frozen=True, slots=True)
class TypographyStyle:
    """Styling specification for typography rendering."""

    position: str = "bottom"
    font_size: int = 44
    text_color: str = "#FFFFFF"
    stroke_color: str | None = "#000000"
    stroke_width: int = 2
    badge: bool = True
    badge_color: str = "#1A1A1AE6"
    badge_radius: int = 16
    badge_padding: tuple[int, int] = (24, 14)
    shadow: bool = True
    shadow_offset: tuple[int, int] = (2, 4)
    shadow_blur: int = 6
    max_width_ratio: float = 0.85


@dataclass(frozen=True, slots=True)
class TypographyResult:
    """Outcome of applying typography overlay."""

    output: Path
    width: int
    height: int
    lines_rendered: int
    is_video: bool


def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a suitable TrueType system font or fallback to default."""
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, font_size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        return ImageFont.load_default()


def _parse_rgba(color_str: str, default_alpha: int = 255) -> tuple[int, int, int, int]:
    """Parse hex or CSS color string into (R, G, B, A) tuple."""
    clean = color_str.strip()
    if clean.startswith("#") and len(clean) == 9:
        # #RRGGBBAA
        r = int(clean[1:3], 16)
        g = int(clean[3:5], 16)
        b = int(clean[5:7], 16)
        a = int(clean[7:9], 16)
        return (r, g, b, a)

    try:
        rgb = ImageColor.getrgb(clean)
        if len(rgb) == 4:
            return (rgb[0], rgb[1], rgb[2], rgb[3])
        return (rgb[0], rgb[1], rgb[2], default_alpha)
    except Exception:
        return (255, 255, 255, default_alpha)


def render_typography_overlay(
    canvas_w: int,
    canvas_h: int,
    text: str,
    style: TypographyStyle,
) -> tuple[Image.Image, int]:
    """Render a transparent RGBA image containing styled text and badge card."""
    font = _load_font(style.font_size)

    # Estimate line length for word wrapping
    max_text_width = int(canvas_w * style.max_width_ratio) - (style.badge_padding[0] * 2)
    sample_char_w = max(1, style.font_size // 2)
    max_chars_per_line = max(10, max_text_width // sample_char_w)

    raw_lines: list[str] = []
    for paragraph in text.split("\n"):
        wrapped = textwrap.wrap(paragraph, width=max_chars_per_line)
        raw_lines.extend(wrapped if wrapped else [""])

    lines = [line.strip() for line in raw_lines if line.strip()]
    if not lines:
        lines = [text.strip()]

    # Measure text bounding box
    dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    line_metrics: list[tuple[int, int]] = []
    total_text_h = 0
    max_line_w = 0
    line_spacing = int(style.font_size * 0.3)

    for line in lines:
        bbox = dummy_draw.textbbox((0, 0), line, font=font, stroke_width=style.stroke_width)
        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])
        line_metrics.append((w, h))
        max_line_w = max(max_line_w, w)
        total_text_h += h

    total_text_h += max(0, len(lines) - 1) * line_spacing

    # Compute card box dimensions
    pad_x, pad_y = style.badge_padding
    card_w = max_line_w + (pad_x * 2)
    card_h = total_text_h + (pad_y * 2)

    # Position card box on canvas
    margin_x = int(canvas_w * 0.05)
    margin_y = int(canvas_h * 0.08)

    pos = style.position.lower()
    if pos == "top":
        card_x = (canvas_w - card_w) // 2
        card_y = margin_y
    elif pos == "center":
        card_x = (canvas_w - card_w) // 2
        card_y = (canvas_h - card_h) // 2
    elif pos == "top-left":
        card_x = margin_x
        card_y = margin_y
    elif pos == "top-right":
        card_x = canvas_w - card_w - margin_x
        card_y = margin_y
    elif pos == "bottom-left":
        card_x = margin_x
        card_y = canvas_h - card_h - margin_y
    elif pos == "bottom-right":
        card_x = canvas_w - card_w - margin_x
        card_y = canvas_h - card_h - margin_y
    else:  # default "bottom"
        card_x = (canvas_w - card_w) // 2
        card_y = canvas_h - card_h - margin_y

    card_x = max(0, min(canvas_w - card_w, card_x))
    card_y = max(0, min(canvas_h - card_h, card_y))

    # Base RGBA transparent canvas
    overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # Optional Drop Shadow pass
    if style.shadow:
        shadow_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        s_draw = ImageDraw.Draw(shadow_layer)
        off_x, off_y = style.shadow_offset
        s_box = [
            card_x + off_x,
            card_y + off_y,
            card_x + card_w + off_x,
            card_y + card_h + off_y,
        ]
        s_color = (0, 0, 0, 160)
        if style.badge:
            s_draw.rounded_rectangle(s_box, radius=style.badge_radius, fill=s_color)
        if style.shadow_blur > 0:
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(style.shadow_blur))
        overlay = Image.alpha_composite(overlay, shadow_layer)

    draw = ImageDraw.Draw(overlay)

    # Draw Badge Capsule
    if style.badge:
        b_color = _parse_rgba(style.badge_color, default_alpha=230)
        draw.rounded_rectangle(
            [card_x, card_y, card_x + card_w, card_y + card_h],
            radius=style.badge_radius,
            fill=b_color,
        )

    # Draw Text Lines
    curr_y = card_y + pad_y
    txt_color = _parse_rgba(style.text_color, default_alpha=255)
    strk_color = _parse_rgba(style.stroke_color, default_alpha=255) if style.stroke_color else None

    for i, line in enumerate(lines):
        lw, lh = line_metrics[i]
        lx = card_x + (card_w - lw) // 2
        draw.text(
            (lx, curr_y),
            line,
            font=font,
            fill=txt_color,
            stroke_width=style.stroke_width,
            stroke_fill=strk_color,
        )
        curr_y += lh + line_spacing

    return (overlay, len(lines))


def apply_typography(
    source: Path | str,
    output: Path | str,
    text: str,
    config: Config,
    *,
    style: TypographyStyle | None = None,
    start_s: float = 0.0,
    duration_s: float | None = None,
    fade_s: float = 0.25,
    force: bool = False,
) -> TypographyResult:
    """Apply styled typography / title badge overlay to an image or video.

    Args:
        source: Path to source image or video.
        output: Path to destination file.
        text: Text to render.
        config: System configuration.
        style: TypographyStyle settings.
        start_s: Start timestamp for video overlay in seconds.
        duration_s: Duration of video overlay in seconds (None = whole video).
        fade_s: Fade in/out duration for video overlay.
        force: Allow overwriting existing output.

    Returns:
        TypographyResult with dimensions and line count.
    """
    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    if not text.strip():
        raise ValidationError("text content cannot be empty")

    resolved_style = style if style is not None else TypographyStyle()
    if resolved_style.position.lower() not in VALID_POSITIONS:
        opts = sorted(VALID_POSITIONS)
        raise ValidationError(
            f"unsupported position {resolved_style.position!r}, supported: {opts}"
        )

    suffix = resolved_source.suffix.lower()
    is_video = suffix in VIDEO_EXTENSIONS

    if not is_video and suffix not in IMAGE_EXTENSIONS:
        raise ValidationError(f"unrecognized media format: {suffix}")

    if not is_video:
        # Still Image Branch
        with Image.open(resolved_source) as img:
            base = img.convert("RGBA")
            cw, ch = base.size
            overlay, lines_count = render_typography_overlay(cw, ch, text, resolved_style)
            combined = Image.alpha_composite(base, overlay)

            out_ext = resolved_output.suffix.lower()
            if out_ext in {".jpg", ".jpeg"}:
                combined.convert("RGB").save(resolved_output, quality=95)
            else:
                combined.save(resolved_output)

        return TypographyResult(
            output=resolved_output,
            width=cw,
            height=ch,
            lines_rendered=lines_count,
            is_video=False,
        )

    # Video Branch
    source_info = probe(resolved_source, config)
    if not source_info.has_video or source_info.width is None or source_info.height is None:
        raise ValidationError(f"source {resolved_source} has no valid video stream")

    vw = source_info.width
    vh = source_info.height
    video_dur = source_info.duration_s or 5.0

    overlay, lines_count = render_typography_overlay(vw, vh, text, resolved_style)

    # Save overlay card to work directory
    overlay_path = work_path(config, f"typo_card_{resolved_source.stem}", ".png")
    overlay.save(overlay_path)

    # Build FFmpeg command
    actual_dur = duration_s if duration_s is not None else (video_dur - start_s)
    end_s = start_s + actual_dur

    filter_complex: str
    if fade_s > 0 and actual_dur > (fade_s * 2):
        fd = min(fade_s, actual_dur / 2)
        out_st = max(start_s, end_s - fd)
        filter_complex = (
            f"[1:v]format=yuva420p,"
            f"fade=t=in:st={start_s}:d={fd}:alpha=1,"
            f"fade=t=out:st={out_st}:d={fd}:alpha=1[ov_fade];"
            f"[0:v][ov_fade]overlay=0:0:enable='between(t,{start_s},{end_s})':eof_action=pass[out_v]"
        )
    else:
        filter_complex = (
            f"[0:v][1:v]overlay=0:0:enable='between(t,{start_s},{end_s})':eof_action=pass[out_v]"
        )

    cmd: list[str] = [
        "-i",
        str(resolved_source),
        "-loop",
        "1",
        "-t",
        str(end_s),
        "-i",
        str(overlay_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out_v]",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
    ]

    if source_info.has_audio:
        cmd.extend(["-map", "0:a", "-c:a", "copy"])
    else:
        cmd.append("-an")

    cmd.append(str(resolved_output))
    run_ffmpeg(cmd, config)

    final_info = verify_render(
        resolved_output,
        config,
        Expectations(
            duration_s=source_info.duration_s,
            width=vw,
            height=vh,
            requires_video=True,
            requires_audio=source_info.has_audio,
        ),
    )

    return TypographyResult(
        output=resolved_output,
        width=final_info.width or vw,
        height=final_info.height or vh,
        lines_rendered=lines_count,
        is_video=True,
    )
