"""Tests for audio stem separation and clean speech recipe and CLI."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from media_lab.cli import main
from media_lab.config import Config
from media_lab.errors import MediaLabError, PathSafetyError, ValidationError
from media_lab.ml_runner import MlResult, MlRunner
from media_lab.recipes.stems import separate_stems


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


def _fake_demucs_runner(config: Config) -> Callable[..., MlResult]:
    """Fake ML runner that simulates demucs creating stem audio files."""

    def run(
        self: MlRunner,
        script: Path | str,
        args: Sequence[str] = (),
        *,
        timeout_s: int = 0,
        cwd: Path | None = None,
    ) -> MlResult:
        argv = list(args)
        out_dir = Path(argv[1])
        out_dir.mkdir(parents=True, exist_ok=True)

        two_stems = "vocals"
        if "--two-stems" in argv:
            two_stems = argv[argv.index("--two-stems") + 1]

        fmt = "wav"
        if "--format" in argv:
            fmt = argv[argv.index("--format") + 1]

        if two_stems != "none":
            _generate_synthetic_audio(out_dir / f"{two_stems}.{fmt}", config)
            _generate_synthetic_audio(out_dir / f"no_{two_stems}.{fmt}", config)
        else:
            for name in ("drums", "bass", "other", "vocals"):
                _generate_synthetic_audio(out_dir / f"{name}.{fmt}", config)

        return MlResult(
            command=(str(script), *argv),
            stdout="Fake demucs separation complete",
            stderr="",
            duration_s=0.1,
        )

    return run


@pytest.fixture
def ml_runner(config: Config) -> MlRunner:
    return MlRunner.from_config(config)


def test_stems_audio_two_stems_happy_path(
    config: Config, ml_runner: MlRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MlRunner, "run", _fake_demucs_runner(config))
    audio_path = _generate_synthetic_audio(config.in_dir / "sample.wav", config)
    out_dir = config.out_dir / "stems_test"

    result = separate_stems(audio_path, out_dir, config, ml_runner, two_stems="vocals")

    assert "vocals" in result.stems
    assert "no_vocals" in result.stems
    assert result.stems["vocals"].is_file()
    assert result.stems["no_vocals"].is_file()
    assert result.cleaned_video is None


def test_stems_four_stems(
    config: Config, ml_runner: MlRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MlRunner, "run", _fake_demucs_runner(config))
    audio_path = _generate_synthetic_audio(config.in_dir / "full_sample.wav", config)
    out_dir = config.out_dir / "stems_4"

    result = separate_stems(audio_path, out_dir, config, ml_runner, two_stems="none")

    assert set(result.stems.keys()) == {"drums", "bass", "other", "vocals"}
    for path in result.stems.values():
        assert path.is_file()


def test_stems_clean_speech_on_video(
    config: Config, ml_runner: MlRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MlRunner, "run", _fake_demucs_runner(config))
    video_path = _generate_synthetic_video_with_audio(config.in_dir / "clip_noisy.mp4", config)
    out_dir = config.out_dir / "stems_clean"
    clean_video_out = config.out_dir / "clip_cleaned.mp4"

    result = separate_stems(
        video_path,
        out_dir,
        config,
        ml_runner,
        clean_speech=True,
        output_video=clean_video_out,
    )

    assert result.cleaned_video is not None
    assert result.cleaned_video.is_file()
    assert result.cleaned_video == clean_video_out


def test_stems_missing_source_fails(config: Config, ml_runner: MlRunner) -> None:
    with pytest.raises(MediaLabError, match="does not exist"):
        separate_stems(
            config.in_dir / "nonexistent.wav",
            config.out_dir / "stems_fail",
            config,
            ml_runner,
        )


def test_stems_source_without_audio_fails(config: Config, ml_runner: MlRunner) -> None:
    video_no_audio = _generate_synthetic_video_no_audio(config.in_dir / "silent.mp4", config)
    with pytest.raises(ValidationError, match="has no audio stream"):
        separate_stems(
            video_no_audio,
            config.out_dir / "stems_silent",
            config,
            ml_runner,
        )


def test_stems_clean_speech_on_audio_only_fails(
    config: Config, ml_runner: MlRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MlRunner, "run", _fake_demucs_runner(config))
    audio_path = _generate_synthetic_audio(config.in_dir / "pure_audio.wav", config)
    with pytest.raises(ValidationError, match="requires a video source"):
        separate_stems(
            audio_path,
            config.out_dir / "stems_fail_video",
            config,
            ml_runner,
            clean_speech=True,
        )


def test_stems_invalid_choices(config: Config, ml_runner: MlRunner) -> None:
    audio_path = _generate_synthetic_audio(config.in_dir / "valid.wav", config)
    with pytest.raises(ValidationError, match="two_stems must be one of"):
        separate_stems(
            audio_path,
            config.out_dir / "stems_invalid",
            config,
            ml_runner,
            two_stems="invalid_stem",
        )
    with pytest.raises(ValidationError, match="audio_format must be one of"):
        separate_stems(
            audio_path,
            config.out_dir / "stems_invalid2",
            config,
            ml_runner,
            audio_format="ogg",
        )


def test_stems_overwrite_protection(
    config: Config, ml_runner: MlRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MlRunner, "run", _fake_demucs_runner(config))
    video_path = _generate_synthetic_video_with_audio(config.in_dir / "protect.mp4", config)
    out_dir = config.out_dir / "stems_protect"
    out_video = config.out_dir / "protected_video.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    out_video.write_bytes(b"existing")

    with pytest.raises(PathSafetyError, match="output already exists"):
        separate_stems(
            video_path,
            out_dir,
            config,
            ml_runner,
            clean_speech=True,
            output_video=out_video,
            force=False,
        )

    # With force=True, it succeeds
    result = separate_stems(
        video_path,
        out_dir,
        config,
        ml_runner,
        clean_speech=True,
        output_video=out_video,
        force=True,
    )
    assert result.cleaned_video == out_video


def test_cli_stems(
    config: Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("media_lab.cli.load_config", lambda: config)
    monkeypatch.setattr(MlRunner, "run", _fake_demucs_runner(config))
    video_path = _generate_synthetic_video_with_audio(config.in_dir / "cli_sample.mp4", config)
    out_dir = config.out_dir / "cli_stems"
    clean_out = config.out_dir / "cli_clean.mp4"

    rc = main(
        [
            "stems",
            str(video_path),
            "-o",
            str(out_dir),
            "--clean-speech",
            "--clean-video-out",
            str(clean_out),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "separated stems saved" in captured
    assert "clean speech video" in captured
