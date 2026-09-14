"""Tests for Prompt Agent: natural language interpretation, planning, and execution."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from media_lab.cli import main
from media_lab.config import Config
from media_lab.kino import KinoRunner
from media_lab.ml_runner import MlRunner
from media_lab.prompt_agent import execute_prompt, interpret_prompt, plan_prompt


def _generate_synthetic_video(path: Path, config: Config, duration: float = 2.0) -> Path:
    """Generate short 16:9 video with audio tone for testing."""
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
            f"testsrc=duration={duration}:size=640x360:rate=25",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=400:duration={duration}",
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


def test_interpret_prompt_romanian() -> None:
    prompt = (
        "Fă-mi un short vertical cu subtitrări galbene, "
        "taie pauzele, curăță vocea și adaugă whoosh la început"
    )
    spec = interpret_prompt(prompt, "in/test.mp4", "out/short.mp4")

    assert spec.video.aspect == "9:16"
    assert spec.video.smart_reframe is True
    assert spec.video.subtitles.enabled is True
    assert spec.video.subtitles.style == "boxed_yellow"
    assert spec.audio.silence_trim is True
    assert spec.audio.clean_speech is True
    assert spec.audio.master_profile == "podcast"
    assert len(spec.audio.sfx_cues) >= 1
    assert spec.audio.sfx_cues[0]["kind"] == "whoosh"


def test_interpret_prompt_english() -> None:
    prompt = (
        "Make a vertical reels video, master audio for podcast, "
        "add retention punch zoom, and title 'Media Lab 2.0'"
    )
    spec = interpret_prompt(prompt, "in/clip.mp4", "out/reels.mp4")

    assert spec.video.aspect == "9:16"
    assert spec.video.smart_reframe is True
    assert spec.audio.master_profile == "podcast"
    assert spec.video.punch_zoom is True
    assert spec.video.typography is not None
    assert spec.video.typography.text == "Media Lab 2.0"


def test_interpret_prompt_photo_retouch() -> None:
    prompt = "Retușează portretul: netezește tenul, adaugă bokeh pe fundal și titlu 'Speaker' jos"
    spec = interpret_prompt(prompt, "in/person.jpg", "out/retouched.jpg")

    assert spec.photo.retouch is True
    assert spec.photo.skin_strength > 0.0
    assert spec.photo.depth_blur is True
    assert spec.video.typography is not None
    assert spec.video.typography.text == "Speaker"


def test_plan_prompt_output() -> None:
    plan = plan_prompt(
        "Taie pauzele, curata vocea si pune look cinematic",
        "in/talk.mp4",
        "out/final.mp4",
    )
    assert len(plan.operations) >= 3
    assert any("pauze" in op.lower() for op in plan.operations)
    assert any("vocal" in op.lower() or "voce" in op.lower() for op in plan.operations)
    assert any("cinematic" in op.lower() for op in plan.operations)


def test_execute_prompt_end_to_end(config: Config) -> None:
    src = _generate_synthetic_video(config.in_dir / "raw_talk.mp4", config, duration=2.0)
    out = config.out_dir / "agent_result.mp4"
    runner = KinoRunner.from_config(config)
    ml_runner = MlRunner.from_config(config)

    prompt = "Curata vocea cu profil radio, zoom dinamic si titlu 'Pro' sus"
    result = execute_prompt(prompt, src, out, config, runner, ml_runner, force=True)

    assert result.output.is_file()
    assert result.media.has_video is True
    assert len(result.steps_executed) >= 2


def test_cli_prompt(
    config: Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("media_lab.cli.load_config", lambda: config)
    src = _generate_synthetic_video(config.in_dir / "cli_raw.mp4", config, duration=1.5)
    out = config.out_dir / "cli_agent_out.mp4"

    # Plan only mode
    rc_plan = main(
        ["prompt", "Fa un short cu look cald", "-i", str(src), "-o", str(out), "--plan-only"]
    )
    assert rc_plan == 0
    captured_plan = capsys.readouterr().out
    assert "Plan derived from:" in captured_plan

    # Execution mode
    rc_exec = main(
        ["prompt", "Fa un short cu look cald", "-i", str(src), "-o", str(out), "--force"]
    )
    assert rc_exec == 0
    assert out.is_file()
