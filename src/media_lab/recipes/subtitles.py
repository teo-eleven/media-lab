"""Recipe for generating and burning subtitles using Whisper and ASS styling.

Runs Whisper as a child process through `ml_runner`, formats subtitles with
custom social media styles (tiktok, clean, box), and optionally burns them into video.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..errors import ValidationError
from ..ffmpeg import run_ffmpeg
from ..ml_runner import MlRunner
from ..paths import ensure_readable_source, ensure_writable_output, work_directory
from ..probe import probe
from ..subtitles import (
    STYLE_CHOICES,
    SubtitleSegment,
    generate_ass,
    generate_srt,
)
from ..validation import check_choice
from ..verify import Expectations, verify_render

_WHISPER_INFER = Path(__file__).resolve().parent.parent / "ml" / "whisper_infer.py"
SUBTITLE_EXTENSIONS = (".ass", ".srt")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm")


@dataclass(frozen=True, slots=True)
class SubtitlesResult:
    """Outcome of subtitle generation."""

    subtitle_file: Path
    video_file: Path | None
    segment_count: int
    language: str


def generate_subtitles(
    source: Path | str,
    output: Path | str,
    config: Config,
    ml_runner: MlRunner,
    *,
    style: str = "tiktok",
    model: str = "base",
    language: str | None = None,
    burn: bool = False,
    output_subs: Path | str | None = None,
    force: bool = False,
) -> SubtitlesResult:
    """Transcribe speech and generate styled subtitles or burn onto video.

    Args:
        source: Path to input audio or video file.
        output: Destination subtitle file (.ass/.srt) or destination video (.mp4/.mov).
        config: Configuration object.
        ml_runner: Subprocess runner for ML scripts.
        style: Subtitle style preset ('tiktok', 'clean', 'box').
        model: Pretrained Whisper model (default: 'base').
        language: Language code (e.g. 'en', 'ro', 'es').
        burn: Burn subtitles into video. Automatically set to True if output is a video extension.
        output_subs: Optional path to save subtitle file when burning onto video.
        force: Allow overwriting existing output.
    """
    check_choice(style, STYLE_CHOICES, "style")

    resolved_source = ensure_readable_source(source)
    source_info = probe(resolved_source, config)
    if not source_info.has_audio:
        raise ValidationError(f"source has no audio to transcribe: {resolved_source}")

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = (config.root / output_path).resolve()

    suffix = output_path.suffix.lower()
    is_video_target = suffix in VIDEO_EXTENSIONS
    should_burn = burn or is_video_target

    if should_burn and not source_info.has_video:
        raise ValidationError(
            f"burning subtitles requires a video source, but {resolved_source} has no video"
        )

    work = work_directory(config, "subtitles")
    transcript_json = work / f"{resolved_source.stem}_transcript.json"

    args = [
        str(resolved_source),
        str(transcript_json),
        "--model",
        model,
    ]
    if language:
        args.extend(["--language", language])

    ml_runner.run(_WHISPER_INFER, args)

    if not transcript_json.is_file():
        raise ValidationError(f"transcript JSON missing at {transcript_json}")

    raw_data = json.loads(transcript_json.read_text(encoding="utf-8"))
    detected_lang = str(raw_data.get("language", language or "unknown"))
    segments = [SubtitleSegment.from_dict(s) for s in raw_data.get("segments", [])]

    width = source_info.width or 1080
    height = source_info.height or 1920

    ass_content = generate_ass(segments, style=style, width=width, height=height)
    srt_content = generate_srt(segments)

    created_sub_path: Path
    created_video_path: Path | None = None

    if should_burn:
        resolved_video_out = ensure_writable_output(output_path, config, force=force)
        sub_tmp = work / f"{resolved_source.stem}_{style}.ass"
        sub_tmp.write_text(ass_content, encoding="utf-8")
        created_sub_path = sub_tmp

        if output_subs is not None:
            saved_subs = ensure_writable_output(output_subs, config, force=force)
            if saved_subs.suffix.lower() == ".srt":
                saved_subs.write_text(srt_content, encoding="utf-8")
            else:
                saved_subs.write_text(ass_content, encoding="utf-8")
            created_sub_path = saved_subs

        escaped_sub_path = sub_tmp.as_posix().replace("\\", "/").replace(":", r"\:")
        run_ffmpeg(
            [
                "-i",
                str(resolved_source),
                "-vf",
                f"ass='{escaped_sub_path}'",
                "-c:a",
                "copy",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-t",
                f"{source_info.duration_s:.3f}",
                str(resolved_video_out),
            ],
            config,
        )
        verify_render(
            resolved_video_out,
            config,
            Expectations(duration_s=source_info.duration_s, requires_audio=True),
        )
        created_video_path = resolved_video_out
    else:
        resolved_sub_out = ensure_writable_output(output_path, config, force=force)
        if resolved_sub_out.suffix.lower() == ".srt":
            resolved_sub_out.write_text(srt_content, encoding="utf-8")
        else:
            resolved_sub_out.write_text(ass_content, encoding="utf-8")
        created_sub_path = resolved_sub_out

    return SubtitlesResult(
        subtitle_file=created_sub_path,
        video_file=created_video_path,
        segment_count=len(segments),
        language=detected_lang,
    )
