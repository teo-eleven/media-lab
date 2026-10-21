"""Subtitle formatting and ASS styling generator.

Supports standard SRT and styled ASS (Advanced SubStation Alpha) presets
tailored for social media video (TikTok/Reels dynamic, clean minimalist, opaque box).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

STYLE_CHOICES = ("tiktok", "clean", "box")


@dataclass(frozen=True, slots=True)
class WordToken:
    word: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class SubtitleSegment:
    id: int
    start: float
    end: float
    text: str
    words: tuple[WordToken, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubtitleSegment:
        words = tuple(
            WordToken(
                word=str(w.get("word", "")).strip(),
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
            )
            for w in data.get("words", [])
            if str(w.get("word", "")).strip()
        )
        return cls(
            id=int(data.get("id", 0)),
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            text=str(data.get("text", "")).strip(),
            words=words,
        )


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds into SRT timestamp format: HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis -= 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def format_timestamp_ass(seconds: float) -> str:
    """Format seconds into ASS timestamp format: H:MM:SS.cc."""
    if seconds < 0:
        seconds = 0.0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        secs += 1
        centis -= 100
    return f"{hrs}:{mins:02d}:{secs:02d}.{centis:02d}"


def generate_srt(segments: Sequence[SubtitleSegment]) -> str:
    """Generate SRT subtitle file content."""
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        if not seg.text:
            continue
        start_str = format_timestamp_srt(seg.start)
        end_str = format_timestamp_srt(seg.end)
        lines.append(f"{i}\n{start_str} --> {end_str}\n{seg.text}\n")
    return "\n".join(lines).strip() + "\n"


def _build_ass_styles(style: str, width: int, height: int) -> str:
    scale_factor = height / 1080.0

    if style == "tiktok":
        font_size = int(round(52 * scale_factor))
        margin_v = int(round(height * 0.40))  # Placed around middle-center
        return (
            f"Style: Default,Arial Black,{font_size},&H0000FFFF,&H000000FF,&H00000000,&H00000000,"
            f"-1,0,0,0,100,100,0,0,1,4,1.5,5,40,40,{margin_v},1"
        )
    if style == "box":
        font_size = int(round(38 * scale_factor))
        margin_v = int(round(80 * scale_factor))
        return (
            f"Style: Default,Helvetica,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H90000000,"
            f"0,0,0,0,100,100,0,0,3,0,0,2,40,40,{margin_v},1"
        )
    # Default 'clean'
    font_size = int(round(38 * scale_factor))
    margin_v = int(round(80 * scale_factor))
    return (
        f"Style: Default,Helvetica,{font_size},&H00FFFFFF,&H000000FF,&H00151515,&H00000000,"
        f"0,0,0,0,100,100,0,0,1,2.5,1,2,40,40,{margin_v},1"
    )


def generate_ass(
    segments: Sequence[SubtitleSegment],
    *,
    style: str = "tiktok",
    width: int = 1080,
    height: int = 1920,
) -> str:
    """Generate ASS subtitle file content with styling presets."""
    style_def = _build_ass_styles(style, width, height)

    format_line = (
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding"
    )
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
{format_line}
{style_def}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dialogues: list[str] = []

    for seg in segments:
        if not seg.text:
            continue

        # If tiktok style and word timestamps are available, chunk into 3-5 words
        if style == "tiktok" and seg.words:
            words = list(seg.words)
            chunk_size = 4
            for i in range(0, len(words), chunk_size):
                chunk = words[i : i + chunk_size]
                start_str = format_timestamp_ass(chunk[0].start)
                end_str = format_timestamp_ass(chunk[-1].end)
                chunk_text = " ".join(w.word for w in chunk).upper()
                dialogues.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{chunk_text}")
        else:
            start_str = format_timestamp_ass(seg.start)
            end_str = format_timestamp_ass(seg.end)
            clean_text = seg.text.replace("\n", "\\N")
            dialogues.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{clean_text}")

    return header + "\n".join(dialogues) + "\n"
