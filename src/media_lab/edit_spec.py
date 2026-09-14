"""Declarative edit specification parser and schema for autonomous video/photo editing.

Allows human users or LLM agents to declare high-level editing intent:
reframe, speech clean, look, subtitles, and music mix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ValidationError
from .recipes.filters import LOOKS
from .subtitles import STYLE_CHOICES
from .verify import ASPECT_RATIOS

SUPPORTED_ASPECTS = set(ASPECT_RATIOS) | {"original", "1:1", "4:5", "9:16", "16:9"}


@dataclass(frozen=True, slots=True)
class AudioEditSpec:
    clean_speech: bool = False
    music_track: str | None = None
    target_lufs: float = -16.0
    music_volume: float = 1.0


@dataclass(frozen=True, slots=True)
class SubtitleEditSpec:
    enabled: bool = False
    style: str = "tiktok"
    language: str | None = None


@dataclass(frozen=True, slots=True)
class VideoEditSpec:
    aspect: str = "original"
    look: str | None = None
    second_look: str | None = None
    subtitles: SubtitleEditSpec = field(default_factory=SubtitleEditSpec)
    cutout: bool = False
    backdrop: str | None = None


@dataclass(frozen=True, slots=True)
class EditSpec:
    source: str
    output: str
    video: VideoEditSpec = field(default_factory=VideoEditSpec)
    audio: AudioEditSpec = field(default_factory=AudioEditSpec)


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

    video_spec = VideoEditSpec(
        aspect=aspect,
        look=look,
        second_look=second_look,
        subtitles=sub_spec,
        cutout=bool(video_dict.get("cutout", False)),
        backdrop=video_dict.get("backdrop"),
    )

    audio_dict = raw.get("audio", {})
    if not isinstance(audio_dict, dict):
        raise ValidationError("'audio' field must be a dictionary")

    audio_spec = AudioEditSpec(
        clean_speech=bool(audio_dict.get("clean_speech", False)),
        music_track=audio_dict.get("music_track") or audio_dict.get("track"),
        target_lufs=float(audio_dict.get("target_lufs", -16.0)),
        music_volume=float(audio_dict.get("music_volume", 1.0)),
    )

    return EditSpec(
        source=src_path,
        output=out_path,
        video=video_spec,
        audio=audio_spec,
    )
