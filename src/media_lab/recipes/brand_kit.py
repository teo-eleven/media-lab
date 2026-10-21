"""Recipe: Brand Kit — Logo Watermarking and Corporate Branding Engine.

Inspired by Creatopy, Celtra, and Enterprise Video Marketing workflows.
Places vector or transparent raster logos with proportional scale, opacity,
and safe-zone margins to ensure professional broadcast delivery.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..errors import ValidationError
from ..ffmpeg import run_ffmpeg
from ..paths import ensure_readable_source, ensure_writable_output
from ..probe import MediaInfo, probe
from ..verify import Expectations, verify_render

VALID_BRAND_POSITIONS: frozenset[str] = frozenset(
    {"top-left", "top-right", "bottom-left", "bottom-right", "center"}
)


@dataclass(frozen=True, slots=True)
class BrandResult:
    """Outcome of applying a brand logo watermark."""

    output: Path
    media: MediaInfo
    position: str
    opacity: float
    scale_pct: float


def apply_brand_watermark(
    source: Path | str,
    logo: Path | str,
    output: Path | str,
    config: Config,
    *,
    position: str = "top-right",
    opacity: float = 0.85,
    scale_pct: float = 12.0,
    margin_px: int = 30,
    force: bool = False,
) -> BrandResult:
    """Overlay a brand logo watermark with customizable position and opacity.

    Args:
        source: Source video file.
        logo: Image file (PNG with alpha channel recommended).
        output: Destination video file.
        config: System configuration.
        position: 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'.
        opacity: Logo opacity from 0.05 to 1.0 (default 0.85).
        scale_pct: Logo width as percentage of video width (default 12.0).
        margin_px: Margin distance from the edges in pixels (default 30).
        force: Overwrite output if it already exists.

    Returns:
        BrandResult with the verified media information.
    """
    if position not in VALID_BRAND_POSITIONS:
        raise ValidationError(
            f"Invalid brand position '{position}'. Must be one of {sorted(VALID_BRAND_POSITIONS)}"
        )
    if not 0.05 <= opacity <= 1.0:
        raise ValidationError(f"Logo opacity must be between 0.05 and 1.0, got {opacity}")
    if not 1.0 <= scale_pct <= 50.0:
        raise ValidationError(
            f"Logo scale percentage must be between 1.0 and 50.0, got {scale_pct}"
        )

    resolved_source = ensure_readable_source(source)
    resolved_logo = ensure_readable_source(logo)
    resolved_output = ensure_writable_output(output, config, force=force)

    source_info = probe(resolved_source, config)
    if not source_info.has_video:
        raise ValidationError(f"Source video has no video stream: {resolved_source}")

    src_w = source_info.width
    target_logo_w = max(16, int(round(src_w * (scale_pct / 100.0))))
    target_logo_w = target_logo_w - (target_logo_w % 2)  # Ensure even width

    pos_expressions = {
        "top-left": f"{margin_px}:{margin_px}",
        "top-right": f"W-w-{margin_px}:{margin_px}",
        "bottom-left": f"{margin_px}:H-h-{margin_px}",
        "bottom-right": f"W-w-{margin_px}:H-h-{margin_px}",
        "center": "(W-w)/2:(H-h)/2",
    }
    overlay_xy = pos_expressions[position]

    filter_complex = (
        f"[1:v]scale={target_logo_w}:-2,format=rgba,colorchannelmixer=aa={opacity:.2f}[logo];"
        f"[0:v][logo]overlay={overlay_xy}:format=auto[outv]"
    )

    cmd = [
        "-i",
        str(resolved_source),
        "-i",
        str(resolved_logo),
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
    ]

    if source_info.has_audio:
        cmd.extend(["-map", "0:a", "-c:a", "copy"])

    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-movflags",
            "+faststart",
            str(resolved_output),
        ]
    )

    run_ffmpeg(cmd, config)

    rendered_info = verify_render(
        resolved_output,
        config,
        expected=Expectations(
            duration_s=source_info.duration_s,
            width=source_info.width,
            height=source_info.height,
            requires_audio=source_info.has_audio,
        ),
    )

    return BrandResult(
        output=resolved_output,
        media=rendered_info,
        position=position,
        opacity=opacity,
        scale_pct=scale_pct,
    )
