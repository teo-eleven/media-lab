"""Tests for Prompt Agent: natural language interpretation, planning, and execution."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from media_lab.cli import main
from media_lab.config import Config
from media_lab.kino import KinoRunner
from media_lab.ml_runner import MlRunner
from media_lab.prompt_agent import chat_agent, execute_prompt, interpret_prompt, plan_prompt


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


def test_chat_agent_conversational_flows(config: Config) -> None:
    # 1. Greeting
    greet_res = chat_agent("salut", None, config)
    assert greet_res.intent == "greeting"
    assert greet_res.executable is False
    assert "Media Lab Studio" in greet_res.reply
    assert len(greet_res.suggested_prompts) > 0

    # 2. Help
    help_res = chat_agent("ce poti face in studio?", None, config)
    assert help_res.intent == "help"
    assert help_res.executable is False
    assert "Ce pot face pentru tine" in help_res.reply
    assert len(help_res.suggested_prompts) > 0

    # 3. Inspect with file
    src = _generate_synthetic_video(config.in_dir / "chat_inspect.mp4", config, duration=1.0)
    inspect_res = chat_agent("analizeaza clipul", src, config)
    assert inspect_res.intent == "inspect"
    assert src.name in inspect_res.reply
    assert "Rezoluție" in inspect_res.reply
    assert len(inspect_res.suggested_prompts) > 0

    # 4. Inspect without file
    inspect_none = chat_agent("ce proprietăți are clipul?", None, config)
    assert inspect_none.intent == "inspect"
    assert "Selectează un fișier" in inspect_none.reply

    # 5. Ideation
    idea_res = chat_agent("da-mi o idee virala de editare", None, config)
    assert idea_res.intent == "ideation"
    assert "Rețete Virale" in idea_res.reply
    assert len(idea_res.suggested_prompts) > 0

    # 6. Clarification for ambiguous prompt
    vague_res = chat_agent("ceva dragut", None, config)
    assert vague_res.intent == "clarification"
    assert vague_res.executable is False
    assert len(vague_res.suggested_prompts) > 0

    # 7. Plan for concrete editing prompt
    plan_res = chat_agent(
        "fa un short vertical 9:16 cu subtitrari galbene si taie pauzele", src, config
    )
    assert plan_res.intent == "plan"
    assert plan_res.executable is True
    assert len(plan_res.plan_operations) >= 2

    # 8. Silent video: only audio operations requested
    silent_src = _generate_synthetic_video(config.in_dir / "chat_silent.mp4", config, duration=1.0)
    # Strip audio stream with ffmpeg
    subprocess.run(
        [
            str(config.ffmpeg),
            "-y",
            "-i",
            str(silent_src),
            "-c:v",
            "copy",
            "-an",
            str(config.in_dir / "chat_no_audio.mp4"),
        ],
        check=True,
    )
    no_audio_file = config.in_dir / "chat_no_audio.mp4"

    silent_audio_only = chat_agent("curata vocea si taie pauzele", no_audio_file, config)
    assert silent_audio_only.intent == "clarification"
    assert silent_audio_only.executable is False
    assert "nu conține o pistă audio" in silent_audio_only.reply

    # 9. Silent video: video + audio operations requested
    silent_combo = chat_agent(
        "fa un short vertical 9:16 cu zoom dinamic, curata vocea si taie pauzele",
        no_audio_file,
        config,
    )
    assert silent_combo.intent == "plan"
    assert silent_combo.executable is True
    assert any("omis" in op for op in silent_combo.plan_operations)

    # 10. Execute prompt on silent video end-to-end (must NOT crash)
    out_silent = config.out_dir / "silent_render_test.mp4"
    runner = KinoRunner.from_config(config)
    ml_runner = MlRunner.from_config(config)
    res = execute_prompt(
        "fa un short vertical 9:16 cu zoom dinamic, curata vocea si taie pauzele",
        no_audio_file,
        out_silent,
        config,
        runner,
        ml_runner,
        force=True,
    )
    assert res.output.is_file()
    assert res.media.has_video is True
    assert any("skipped" in step for step in res.steps_executed)


def test_geometric_transforms_prompt(config: Config) -> None:
    # 1. Test interpret_prompt with "pune solul cu sus in jos"
    spec_vflip = interpret_prompt(
        "pune solul cu susul in jos si oglindeste clipul",
        "in/clip.mp4",
        "out/flip.mp4",
    )
    assert spec_vflip.video.vflip is True
    assert spec_vflip.video.hflip is True

    # 2. Test rotate & invert colors & grayscale
    spec_rot = interpret_prompt(
        "roteste 90 grade, fa alb negru si negativ",
        "in/clip.mp4",
        "out/rot.mp4",
    )
    assert spec_rot.video.rotate == 90
    assert spec_rot.video.grayscale is True
    assert spec_rot.video.invert_colors is True

    # 3. Test chat_agent planning for "pune solul cu sus in jos"
    src = _generate_synthetic_video(config.in_dir / "geo_test.mp4", config, duration=1.0)
    chat_vflip = chat_agent("pune solul cu sus in jos", src, config)
    assert chat_vflip.intent == "plan"
    assert chat_vflip.executable is True
    assert any("vflip" in op.lower() for op in chat_vflip.plan_operations)

    # 4. End-to-end execution of vflip
    out_vflip = config.out_dir / "upside_down_render.mp4"
    runner = KinoRunner.from_config(config)
    ml_runner = MlRunner.from_config(config)
    exec_res = execute_prompt(
        "pune solul cu sus in jos",
        src,
        out_vflip,
        config,
        runner,
        ml_runner,
        force=True,
    )
    assert exec_res.output.is_file()
    assert exec_res.media.has_video is True
    assert "vflip" in exec_res.steps_executed

    # 5. Visual filters: brightness, contrast, blur, sharpen, mute
    spec_filters = interpret_prompt(
        "fa clipul mai luminos, creste contrastul, blureaza si da pe mut",
        "in/clip.mp4",
        "out/filtered.mp4",
    )
    assert spec_filters.video.brightness > 0
    assert spec_filters.video.contrast > 1.0
    assert spec_filters.video.blur > 0
    assert spec_filters.video.mute_audio is True
