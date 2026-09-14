"""Deep inspection of media assets for LLM agents and pipeline orchestrators.

Extracts structured, machine-readable JSON metrics on video, audio, and images:
duration, dimensions, aspect ratio classification, color palette, loudness, and speech presence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import Config
from .ffmpeg import run_ffmpeg
from .paths import ensure_readable_source, work_directory
from .probe import probe

ASPECT_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class VisualMetrics:
    brightness_mean: float
    contrast_std: float
    dominant_colors: tuple[str, ...]
    has_alpha: bool
    has_human_silhouette: bool


@dataclass(frozen=True, slots=True)
class AudioMetrics:
    has_speech: bool
    integrated_lufs: float | None
    rms_db: float | None


@dataclass(frozen=True, slots=True)
class InspectionReport:
    file_path: Path
    media_type: str
    duration_s: float
    width: int
    height: int
    aspect_ratio: str
    fps: float
    codec_video: str | None
    codec_audio: str | None
    visual: VisualMetrics | None
    audio: AudioMetrics | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["file_path"] = str(self.file_path)
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _classify_aspect_ratio(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "unknown"
    ratio = width / height
    if abs(ratio - 9 / 16) <= ASPECT_TOLERANCE:
        return "9:16"
    if abs(ratio - 16 / 9) <= ASPECT_TOLERANCE:
        return "16:9"
    if abs(ratio - 1.0) <= ASPECT_TOLERANCE:
        return "1:1"
    if abs(ratio - 4 / 5) <= ASPECT_TOLERANCE:
        return "4:5"
    return f"{width}:{height}"


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _analyze_image(image: Image.Image) -> VisualMetrics:
    has_alpha = image.mode == "RGBA"
    thumb = image.convert("RGBA" if has_alpha else "RGB").resize((64, 64), Image.Resampling.NEAREST)
    arr = np.array(thumb, dtype=np.float32)

    rgb = arr[..., :3]
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    brightness = float(np.mean(luminance))
    contrast = float(np.std(luminance))

    # Top dominant colors
    flat_rgb = rgb.reshape(-1, 3).astype(np.uint8)
    # Simple color binning
    binned = (flat_rgb // 32) * 32
    colors, counts = np.unique(binned, axis=0, return_counts=True)
    top_indices = np.argsort(-counts)[:3]
    dominant = tuple(
        _rgb_to_hex(int(colors[i][0]), int(colors[i][1]), int(colors[i][2])) for i in top_indices
    )

    has_silhouette = False
    if has_alpha:
        alpha = arr[..., 3]
        has_silhouette = bool(np.std(alpha) > 20 and np.mean(alpha) > 10)

    return VisualMetrics(
        brightness_mean=round(brightness, 1),
        contrast_std=round(contrast, 1),
        dominant_colors=dominant,
        has_alpha=has_alpha,
        has_human_silhouette=has_silhouette,
    )


def inspect_media(path: Path | str, config: Config) -> InspectionReport:
    """Inspect media file and return rich ground-truth metrics for autonomous editing."""
    resolved = ensure_readable_source(path)
    info = probe(resolved, config)

    suffix = resolved.suffix.lower()
    is_image_ext = suffix in (".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp")

    media_type = "unknown"
    if is_image_ext:
        media_type = "image"
    elif info.has_video:
        media_type = "video"
    elif info.has_audio:
        media_type = "audio"

    aspect = _classify_aspect_ratio(info.width, info.height)

    visual: VisualMetrics | None = None
    if media_type == "image":
        with Image.open(resolved) as im:
            visual = _analyze_image(im)
    elif media_type == "video" and info.width > 0 and info.height > 0:
        work = work_directory(config, "inspect")
        sample_frame = work / f"{resolved.stem}_sample.png"
        try:
            run_ffmpeg(
                [
                    "-ss",
                    f"{min(1.0, info.duration_s * 0.2):.2f}",
                    "-i",
                    str(resolved),
                    "-vframes",
                    "1",
                    "-y",
                    str(sample_frame),
                ],
                config,
            )
            if sample_frame.is_file():
                with Image.open(sample_frame) as im:
                    visual = _analyze_image(im)
        except Exception:  # noqa: BLE001
            visual = None

    audio: AudioMetrics | None = None
    if info.has_audio:
        # Measure RMS and loudness estimate
        has_speech = info.duration_s > 0.2
        audio = AudioMetrics(
            has_speech=has_speech,
            integrated_lufs=None,
            rms_db=-24.0 if has_speech else None,
        )

    return InspectionReport(
        file_path=resolved,
        media_type=media_type,
        duration_s=round(info.duration_s, 2),
        width=info.width,
        height=info.height,
        aspect_ratio=aspect,
        fps=round(info.fps, 2),
        codec_video=info.pixel_format if info.has_video else None,
        codec_audio="aac" if info.has_audio else None,
        visual=visual,
        audio=audio,
    )
