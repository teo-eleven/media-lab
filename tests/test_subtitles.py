"""Tests for subtitles formatting, ASS styling, recipe and CLI."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from media_lab.cli import main
from media_lab.config import Config
from media_lab.errors import MediaLabError, PathSafetyError, ValidationError
from media_lab.ml_runner import MlResult, MlRunner
from media_lab.recipes.subtitles import generate_subtitles
from media_lab.subtitles import (
    SubtitleSegment,
    WordToken,
    format_timestamp_ass,
    format_timestamp_srt,
    generate_ass,
    generate_srt,
)


def _generate_synthetic_audio(path: Path, config: Config, duration: float = 1.0) -> Path:
    subprocess.run(
        [
            str(config.ffmpeg),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
    )
    return path


def _generate_synthetic_video_with_audio(path: Path, config: Config, duration: float = 1.0) -> Path:
    subprocess.run(
        [
            str(config.ffmpeg),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size=160x120:rate=24",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )
    return path


def _generate_synthetic_video_no_audio(path: Path, config: Config, duration: float = 1.0) -> Path:
    subprocess.run(
        [
            str(config.ffmpeg),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size=160x120:rate=24",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def _fake_whisper_runner(
    sample_text: str = "Hello and welcome to media lab.",
) -> Callable[..., MlResult]:
    """Fake ML runner that simulates Whisper writing JSON transcript."""

    def run(
        self: MlRunner,
        script: Path | str,
        args: Sequence[str] = (),
        *,
        timeout_s: int = 0,
        cwd: Path | None = None,
    ) -> MlResult:
        argv = list(args)
        out_json = Path(argv[1])
        out_json.parent.mkdir(parents=True, exist_ok=True)

        words = [
            {"word": "Hello", "start": 0.0, "end": 0.3, "probability": 0.99},
            {"word": "and", "start": 0.3, "end": 0.5, "probability": 0.95},
            {"word": "welcome", "start": 0.5, "end": 0.9, "probability": 0.98},
            {"word": "to", "start": 0.9, "end": 1.1, "probability": 0.97},
            {"word": "media", "start": 1.1, "end": 1.4, "probability": 0.99},
            {"word": "lab", "start": 1.4, "end": 1.8, "probability": 0.99},
        ]
        data = {
            "text": sample_text,
            "language": "en",
            "duration_s": 0.1,
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.8,
                    "text": sample_text,
                    "words": words,
                }
            ],
        }
        out_json.write_text(json.dumps(data), encoding="utf-8")
        return MlResult(
            command=(str(script), *argv),
            stdout="Fake whisper completed",
            stderr="",
            duration_s=0.1,
        )

    return run


@pytest.fixture
def ml_runner(config: Config) -> MlRunner:
    return MlRunner.from_config(config)


def test_timestamp_formatting() -> None:
    assert format_timestamp_srt(0.0) == "00:00:00,000"
    assert format_timestamp_srt(65.123) == "00:01:05,123"
    assert format_timestamp_srt(3661.05) == "01:01:01,050"
    assert format_timestamp_srt(-5.0) == "00:00:00,000"

    assert format_timestamp_ass(0.0) == "0:00:00.00"
    assert format_timestamp_ass(65.123) == "0:01:05.12"
    assert format_timestamp_ass(3661.05) == "1:01:01.05"
    assert format_timestamp_ass(-5.0) == "0:00:00.00"


def test_generate_srt_and_ass() -> None:
    words = (
        WordToken(word="hello", start=0.0, end=0.5),
        WordToken(word="world", start=0.5, end=1.0),
    )
    seg = SubtitleSegment(id=0, start=0.0, end=1.0, text="hello world", words=words)

    srt = generate_srt([seg])
    assert "1" in srt
    assert "00:00:00,000 --> 00:00:01,000" in srt
    assert "hello world" in srt

    ass_tiktok = generate_ass([seg], style="tiktok", width=1080, height=1920)
    assert "[Script Info]" in ass_tiktok
    assert "PlayResX: 1080" in ass_tiktok
    assert "&H0000FFFF" in ass_tiktok  # Yellow primary color
    assert "HELLO WORLD" in ass_tiktok  # Uppercased words for tiktok style

    ass_clean = generate_ass([seg], style="clean", width=1080, height=1920)
    assert "&H00FFFFFF" in ass_clean
    assert "hello world" in ass_clean

    ass_box = generate_ass([seg], style="box", width=1080, height=1920)
    assert "BorderStyle: 3" in ass_box or "3,0,0,2" in ass_box


def test_subtitles_generate_ass_file_happy_path(
    config: Config, ml_runner: MlRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MlRunner, "run", _fake_whisper_runner())
    audio_path = _generate_synthetic_audio(config.in_dir / "speech.wav", config)
    out_ass = config.out_dir / "speech.ass"

    result = generate_subtitles(audio_path, out_ass, config, ml_runner, style="tiktok")

    assert result.subtitle_file.is_file()
    assert result.subtitle_file == out_ass
    assert result.video_file is None
    assert result.segment_count == 1
    assert result.language == "en"

    content = out_ass.read_text(encoding="utf-8")
    assert "[Script Info]" in content
    assert "HELLO AND WELCOME TO" in content


def test_subtitles_generate_srt_file(
    config: Config, ml_runner: MlRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MlRunner, "run", _fake_whisper_runner())
    audio_path = _generate_synthetic_audio(config.in_dir / "speech2.wav", config)
    out_srt = config.out_dir / "speech.srt"

    result = generate_subtitles(audio_path, out_srt, config, ml_runner)

    assert result.subtitle_file == out_srt
    content = out_srt.read_text(encoding="utf-8")
    assert "-->" in content
    assert "Hello and welcome" in content


def test_subtitles_burn_onto_video(
    config: Config, ml_runner: MlRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MlRunner, "run", _fake_whisper_runner())
    video_path = _generate_synthetic_video_with_audio(config.in_dir / "video_speech.mp4", config)
    out_video = config.out_dir / "burned_subtitles.mp4"
    save_subs = config.out_dir / "saved_subtitles.srt"

    result = generate_subtitles(
        video_path,
        out_video,
        config,
        ml_runner,
        style="clean",
        burn=True,
        output_subs=save_subs,
    )

    assert result.video_file is not None
    assert result.video_file.is_file()
    assert result.video_file == out_video
    assert save_subs.is_file()


def test_subtitles_missing_source_fails(config: Config, ml_runner: MlRunner) -> None:
    with pytest.raises(MediaLabError, match="does not exist"):
        generate_subtitles(
            config.in_dir / "nonexistent.mp4",
            config.out_dir / "subs.ass",
            config,
            ml_runner,
        )


def test_subtitles_source_without_audio_fails(config: Config, ml_runner: MlRunner) -> None:
    silent_video = _generate_synthetic_video_no_audio(config.in_dir / "silent_video.mp4", config)
    with pytest.raises(ValidationError, match="has no audio to transcribe"):
        generate_subtitles(
            silent_video,
            config.out_dir / "subs.ass",
            config,
            ml_runner,
        )


def test_subtitles_burn_on_audio_only_fails(
    config: Config, ml_runner: MlRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MlRunner, "run", _fake_whisper_runner())
    audio_path = _generate_synthetic_audio(config.in_dir / "audio_only.wav", config)
    with pytest.raises(ValidationError, match="requires a video source"):
        generate_subtitles(
            audio_path,
            config.out_dir / "burn_fail.mp4",
            config,
            ml_runner,
            burn=True,
        )


def test_subtitles_invalid_style(config: Config, ml_runner: MlRunner) -> None:
    audio_path = _generate_synthetic_audio(config.in_dir / "audio_valid.wav", config)
    with pytest.raises(ValidationError, match="style must be one of"):
        generate_subtitles(
            audio_path,
            config.out_dir / "subs.ass",
            config,
            ml_runner,
            style="nonexistent_style",
        )


def test_subtitles_overwrite_protection(
    config: Config, ml_runner: MlRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MlRunner, "run", _fake_whisper_runner())
    audio_path = _generate_synthetic_audio(config.in_dir / "speech_prot.wav", config)
    out_ass = config.out_dir / "protected.ass"
    out_ass.parent.mkdir(parents=True, exist_ok=True)
    out_ass.write_text("existing", encoding="utf-8")

    with pytest.raises(PathSafetyError, match="output already exists"):
        generate_subtitles(
            audio_path,
            out_ass,
            config,
            ml_runner,
            force=False,
        )

    # With force=True, it succeeds
    result = generate_subtitles(
        audio_path,
        out_ass,
        config,
        ml_runner,
        force=True,
    )
    assert result.subtitle_file == out_ass


def test_cli_subtitles(
    config: Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("media_lab.cli.load_config", lambda: config)
    monkeypatch.setattr(MlRunner, "run", _fake_whisper_runner())
    audio_path = _generate_synthetic_audio(config.in_dir / "cli_speech.wav", config)
    out_ass = config.out_dir / "cli_subs.ass"

    rc = main(["subtitles", str(audio_path), "-o", str(out_ass), "--style", "tiktok"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "subtitles generated" in captured
    assert "cli_subs.ass" in captured
