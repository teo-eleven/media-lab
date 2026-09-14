"""Declarative edit specification parser and schema for autonomous video/photo editing.

Allows human users, LLM agents, or MCP clients to declare high-level editing intent:
reframe, speech clean, audio mastering, silence jump-cut, visual looks, subtitles,
punch-zoom, B-roll cutaways, typography cards, sound effects, and portrait retouching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ValidationError
from .recipes.audio_enhance import VOICE_PROFILES
from .recipes.filters import LOOKS
from .subtitles import STYLE_CHOICES
from .verify import ASPECT_RATIOS

SUPPORTED_ASPECTS = set(ASPECT_RATIOS) | {"original", "1:1", "4:5", "9:16", "16:9"}


@dataclass(frozen=True, slots=True)
class AudioEditSpec:
    clean_speech: bool = False
    master_profile: str | None = None
    silence_trim: bool = False
    music_track: str | None = None
    target_lufs: float = -16.0
    music_volume: float = 1.0
    sfx_cues: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class SubtitleEditSpec:
    enabled: bool = False
    style: str = "tiktok"
    language: str | None = None


@dataclass(frozen=True, slots=True)
class TypographyEditSpec:
    text: str
    position: str = "bottom"
    font_size: int = 44
    text_color: str = "#FFFFFF"
    badge: bool = True
    start_s: float = 0.0
    duration_s: float | None = None


@dataclass(frozen=True, slots=True)
class VideoEditSpec:
    aspect: str = "original"
    smart_reframe: bool = False
    reframe_mode: str = "smart"
    punch_zoom: bool = False
    zoom_interval: float | None = None
    broll_cuts: tuple[dict[str, Any], ...] = ()
    look: str | None = None
    second_look: str | None = None
    subtitles: SubtitleEditSpec = field(default_factory=SubtitleEditSpec)
    typography: TypographyEditSpec | None = None
    cutout: bool = False
    backdrop: str | None = None
    speed: float = 1.0
    progress_bar: bool = False
    progress_bar_color: str = "yellow"
    progress_bar_position: str = "bottom"
    vflip: bool = False
    hflip: bool = False
    rotate: int = 0
    invert_colors: bool = False
    grayscale: bool = False
    reverse: bool = False


@dataclass(frozen=True, slots=True)
class PhotoEditSpec:
    retouch: bool = False
    skin_strength: float = 0.5
    depth_blur: bool = False
    bokeh_sigma: float = 12.0
    radiance: float = 0.0
    inpaint_bbox: tuple[int, int, int, int] | None = None
    vflip: bool = False
    hflip: bool = False
    rotate: int = 0
    invert_colors: bool = False
    grayscale: bool = False


@dataclass(frozen=True, slots=True)
class EditSpec:
    source: str
    output: str
    video: VideoEditSpec = field(default_factory=VideoEditSpec)
    audio: AudioEditSpec = field(default_factory=AudioEditSpec)
    photo: PhotoEditSpec = field(default_factory=PhotoEditSpec)


def parse_edit_spec(source: str | Path | dict[str, Any]) -> EditSpec:
    """Parse and validate an edit specification from a YAML string, file, or dictionary."""
    if isinstance(source, Path):
        if not source.is_file():
            raise ValidationError(f"edit spec file not found: {source}")
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    elif isinstance(source, str):
        path_candidate = Path(source)
        if path_candidate.is_file():
            raw = yaml.safe_load(path_candidate.read_text(encoding="utf-8"))
        else:
            raw = yaml.safe_load(source)
    elif isinstance(source, dict):
        raw = source
    else:
        raise ValidationError(f"unsupported edit spec source type: {type(source)}")

    if not isinstance(raw, dict):
        raise ValidationError("edit spec must be a mapping/dictionary")

    src_path = raw.get("source")
    if not src_path or not isinstance(src_path, str):
        raise ValidationError("edit spec requires a non-empty 'source' path string")

    out_path = raw.get("output")
    if not out_path or not isinstance(out_path, str):
        raise ValidationError("edit spec requires a non-empty 'output' path string")

    video_dict = raw.get("video", {})
    if not isinstance(video_dict, dict):
        raise ValidationError("'video' field must be a dictionary")

    aspect = str(video_dict.get("aspect", raw.get("target", "original")))
    if aspect not in SUPPORTED_ASPECTS:
        raise ValidationError(f"unsupported aspect ratio in edit spec: {aspect}")

    look = video_dict.get("look")
    if look is not None and look not in LOOKS:
        raise ValidationError(f"unsupported look in edit spec: {look}")

    second_look = video_dict.get("second_look")
    if second_look is not None and second_look not in LOOKS:
        raise ValidationError(f"unsupported second_look in edit spec: {second_look}")

    sub_dict = video_dict.get("subtitles", {})
    if isinstance(sub_dict, bool):
        sub_spec = SubtitleEditSpec(enabled=sub_dict)
    elif isinstance(sub_dict, dict):
        sub_style = str(sub_dict.get("style", "tiktok"))
        if sub_style not in STYLE_CHOICES:
            raise ValidationError(f"unsupported subtitle style in edit spec: {sub_style}")
        sub_spec = SubtitleEditSpec(
            enabled=bool(sub_dict.get("enabled", True)),
            style=sub_style,
            language=sub_dict.get("language"),
        )
    else:
        sub_spec = SubtitleEditSpec(enabled=False)

    typo_dict = video_dict.get("typography")
    typo_spec: TypographyEditSpec | None = None
    if isinstance(typo_dict, dict) and "text" in typo_dict:
        typo_spec = TypographyEditSpec(
            text=str(typo_dict["text"]),
            position=str(typo_dict.get("position", "bottom")),
            font_size=int(typo_dict.get("font_size", 44)),
            text_color=str(typo_dict.get("text_color", "#FFFFFF")),
            badge=bool(typo_dict.get("badge", True)),
            start_s=float(typo_dict.get("start_s", 0.0)),
            duration_s=(
                float(typo_dict["duration_s"]) if typo_dict.get("duration_s") is not None else None
            ),
        )

    broll_raw = video_dict.get("broll_cuts", ())
    broll_tuple = tuple(broll_raw) if isinstance(broll_raw, (list, tuple)) else ()

    video_spec = VideoEditSpec(
        aspect=aspect,
        smart_reframe=bool(video_dict.get("smart_reframe", False)),
        reframe_mode=str(video_dict.get("reframe_mode", "smart")),
        punch_zoom=bool(video_dict.get("punch_zoom", False)),
        zoom_interval=(
            float(video_dict["zoom_interval"])
            if video_dict.get("zoom_interval") is not None
            else None
        ),
        broll_cuts=broll_tuple,
        look=look,
        second_look=second_look,
        subtitles=sub_spec,
        typography=typo_spec,
        cutout=bool(video_dict.get("cutout", False)),
        backdrop=video_dict.get("backdrop"),
        speed=float(video_dict.get("speed", 1.0)),
        progress_bar=bool(video_dict.get("progress_bar", False)),
        progress_bar_color=str(video_dict.get("progress_bar_color", "yellow")),
        progress_bar_position=str(video_dict.get("progress_bar_position", "bottom")),
        vflip=bool(video_dict.get("vflip", False)),
        hflip=bool(video_dict.get("hflip", False)),
        rotate=int(video_dict.get("rotate", 0)),
        invert_colors=bool(video_dict.get("invert_colors", False)),
        grayscale=bool(video_dict.get("grayscale", False)),
        reverse=bool(video_dict.get("reverse", False)),
    )

    audio_dict = raw.get("audio", {})
    if not isinstance(audio_dict, dict):
        raise ValidationError("'audio' field must be a dictionary")

    master_prof = audio_dict.get("master_profile")
    if master_prof is not None and master_prof not in VOICE_PROFILES:
        raise ValidationError(f"unsupported audio master_profile: {master_prof}")

    sfx_raw = audio_dict.get("sfx_cues", ())
    sfx_tuple = tuple(sfx_raw) if isinstance(sfx_raw, (list, tuple)) else ()

    audio_spec = AudioEditSpec(
        clean_speech=bool(audio_dict.get("clean_speech", False)),
        master_profile=master_prof,
        silence_trim=bool(audio_dict.get("silence_trim", False)),
        music_track=audio_dict.get("music_track") or audio_dict.get("track"),
        target_lufs=float(audio_dict.get("target_lufs", -16.0)),
        music_volume=float(audio_dict.get("music_volume", 1.0)),
        sfx_cues=sfx_tuple,
    )

    photo_dict = raw.get("photo", {})
    inpaint_raw = photo_dict.get("inpaint_bbox")
    inpaint_tuple: tuple[int, int, int, int] | None = None
    if isinstance(inpaint_raw, (list, tuple)) and len(inpaint_raw) == 4:
        inpaint_tuple = (
            int(inpaint_raw[0]),
            int(inpaint_raw[1]),
            int(inpaint_raw[2]),
            int(inpaint_raw[3]),
        )

    photo_spec = PhotoEditSpec(
        retouch=bool(photo_dict.get("retouch", False)),
        skin_strength=float(photo_dict.get("skin_strength", 0.5)),
        depth_blur=bool(photo_dict.get("depth_blur", False)),
        bokeh_sigma=float(photo_dict.get("bokeh_sigma", 12.0)),
        radiance=float(photo_dict.get("radiance", 0.0)),
        inpaint_bbox=inpaint_tuple,
        vflip=bool(photo_dict.get("vflip", False)),
        hflip=bool(photo_dict.get("hflip", False)),
        rotate=int(photo_dict.get("rotate", 0)),
        invert_colors=bool(photo_dict.get("invert_colors", False)),
        grayscale=bool(photo_dict.get("grayscale", False)),
    )

    return EditSpec(
        source=src_path,
        output=out_path,
        video=video_spec,
        audio=audio_spec,
        photo=photo_spec,
    )
