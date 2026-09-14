"""Tests for Audio Studio: audio_enhance, silence_trim, and sfx recipes and CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from media_lab.cli import main
from media_lab.config import Config
from media_lab.errors import ValidationError
from media_lab.recipes.audio_enhance import enhance_audio
from media_lab.recipes.sfx import (
    SFX_KINDS,
    SfxCue,
    add_sfx,
    generate_procedural_sfx,
)
from media_lab.recipes.silence_trim import (
    calculate_keep_intervals,
    detect_silence_intervals,
    trim_silence,
)


def _generate_synthetic_speech_video(path: Path, config: Config, duration: float = 2.0) -> Path:
    """Generate synthetic video with audio tone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(config.ffmpeg),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size=320x240:rate=25",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=300:duration={duration}",
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


def _generate_video_with_pause(path: Path, config: Config) -> Path:
    """Generate video containing 1s tone, 1s pure silence, and 1s tone (total 3s)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Audio: 1s sine + 1s silence (volume=0) + 1s sine
    filtergraph = (
        "sine=f=440:d=1.0[a1];"
        "aevalsrc=0:d=1.0[a2];"
        "sine=f=440:d=1.0[a3];"
        "[a1][a2][a3]concat=n=3:v=0:a=1[a]"
    )
    subprocess.run(
        [
            str(config.ffmpeg),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3.0:size=320x240:rate=25",
            "-filter_complex",
            filtergraph,
            "-map",
            "0:v",
            "-map",
            "[a]",
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


def test_audio_enhance_voice_profiles(config: Config) -> None:
    src = _generate_synthetic_speech_video(config.in_dir / "speech.mp4", config)

    for prof in ("podcast", "warm", "crisp", "radio"):
        out = config.out_dir / f"enhanced_{prof}.mp4"
        res = enhance_audio(src, out, config, profile=prof, target_lufs=-14.0)
        assert res.output.is_file()
        assert res.profile == prof
        assert res.media.has_audio is True
        assert res.media.has_video is True


def test_audio_enhance_validation_errors(config: Config) -> None:
    src = _generate_synthetic_speech_video(config.in_dir / "valid.mp4", config)
    out = config.out_dir / "out.mp4"

    with pytest.raises(ValidationError, match="unsupported voice profile"):
        enhance_audio(src, out, config, profile="unknown_profile")

    with pytest.raises(ValidationError, match="target_lufs must be between"):
        enhance_audio(src, out, config, target_lufs=-50.0)


def test_silence_detect_and_trim(config: Config) -> None:
    src = _generate_video_with_pause(config.in_dir / "with_pause.mp4", config)
    out = config.out_dir / "trimmed.mp4"

    intervals = detect_silence_intervals(src, config, min_silence_s=0.3, noise_db=-30.0)
    assert len(intervals) >= 1

    keep = calculate_keep_intervals(3.0, intervals)
    assert len(keep) >= 1

    res = trim_silence(src, out, config, min_silence_s=0.3, noise_db=-30.0)
    assert res.output.is_file()
    assert res.media.has_video is True
    assert res.media.has_audio is True
    assert res.new_duration_s < res.original_duration_s


def test_silence_trim_no_silence(config: Config) -> None:
    src = _generate_synthetic_speech_video(config.in_dir / "continuous.mp4", config, duration=1.0)
    out = config.out_dir / "continuous_out.mp4"

    res = trim_silence(src, out, config, min_silence_s=0.5)
    assert res.output.is_file()
    assert res.cuts_count == 0


def test_sfx_procedural_generation(config: Config) -> None:
    for kind in SFX_KINDS:
        out = config.work_dir / f"test_{kind}.wav"
        res_path = generate_procedural_sfx(kind, out, config)
        assert res_path.is_file()
        assert res_path.stat().st_size > 0


def test_sfx_mixing_on_video(config: Config) -> None:
    src = _generate_synthetic_speech_video(config.in_dir / "sfx_base.mp4", config, duration=2.0)
    out = config.out_dir / "sfx_mixed.mp4"

    cues = [
        SfxCue(kind="whoosh", at_s=0.2, volume=0.8),
        SfxCue(kind="ding", at_s=1.0, volume=0.9),
    ]
    res = add_sfx(src, out, config, cues)
    assert res.output.is_file()
    assert res.cues_count == 2
    assert res.media.has_audio is True


def test_sfx_validation_errors(config: Config) -> None:
    src = _generate_synthetic_speech_video(config.in_dir / "sfx_err.mp4", config)
    out = config.out_dir / "sfx_err_out.mp4"

    with pytest.raises(ValidationError, match="at least one SfxCue"):
        add_sfx(src, out, config, [])

    with pytest.raises(ValidationError, match="unsupported procedural SFX"):
        generate_procedural_sfx("invalid_sfx", out, config)


def test_cli_audio_enhance(
    config: Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("media_lab.cli.load_config", lambda: config)
    src = _generate_synthetic_speech_video(config.in_dir / "cli_speech.mp4", config, duration=1.0)
    out = config.out_dir / "cli_enhanced.mp4"

    rc = main(["audio-enhance", str(src), "-o", str(out), "--profile", "warm"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "audio enhanced:" in captured
    assert out.is_file()


def test_cli_cut_silence(
    config: Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("media_lab.cli.load_config", lambda: config)
    src = _generate_video_with_pause(config.in_dir / "cli_pause.mp4", config)
    out = config.out_dir / "cli_trimmed.mp4"

    rc = main(["cut-silence", str(src), "-o", str(out), "--min-silence", "0.3"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "silence trimmed:" in captured
    assert out.is_file()


def test_cli_sfx(
    config: Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("media_lab.cli.load_config", lambda: config)
    src = _generate_synthetic_speech_video(config.in_dir / "cli_sfx.mp4", config, duration=1.0)
    out = config.out_dir / "cli_sfx_out.mp4"

    rc = main(["sfx", str(src), "-o", str(out), "--kind", "pop", "--at", "0.2"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "sfx mixed:" in captured
    assert out.is_file()
