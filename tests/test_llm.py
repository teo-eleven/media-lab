"""Tests for Cognitive LLM Brain, state diffing, negation, and camera motion handling."""

from __future__ import annotations

from pathlib import Path

from media_lab.edit_spec import (
    AudioEditSpec,
    EditSpec,
    SubtitleEditSpec,
    VideoEditSpec,
)
from media_lab.llm import CognitiveBrain, LLMConfig, merge_specs
from media_lab.probe import MediaInfo


def test_merge_specs() -> None:
    base = EditSpec(
        source="in/raw.mp4",
        output="out/v1.mp4",
        video=VideoEditSpec(aspect="9:16", punch_zoom=True, progress_bar=True),
        audio=AudioEditSpec(clean_speech=True),
    )
    diff = {"video": {"punch_zoom": False, "stabilize": True}}
    merged = merge_specs(base, diff)

    assert merged.video.aspect == "9:16"
    assert merged.video.progress_bar is True
    assert merged.video.punch_zoom is False
    assert merged.video.stabilize is True
    assert merged.audio.clean_speech is True


def test_cognitive_brain_continuity_and_negation() -> None:
    brain = CognitiveBrain(LLMConfig(provider="local"))

    # Initial edit: user created a vertical short with punch zoom and subtitles
    base_spec = EditSpec(
        source="in/clip.mp4",
        output="out/studio_render_clip.mp4",
        video=VideoEditSpec(
            aspect="9:16",
            smart_reframe=True,
            punch_zoom=True,
            progress_bar=True,
            subtitles=SubtitleEditSpec(enabled=True, style="boxed_yellow"),
        ),
        audio=AudioEditSpec(clean_speech=True, master_profile="podcast"),
    )

    # Follow-up iterative prompt matching user's exact request:
    prompt = (
        "este exact la fel ca înainte, scoate acel zoom de pe persoana, "
        "fiindcă in faza inițială când am filmat, am dat zoom out când se apropia de mine "
        "si in videoclipul final se vede prost... poți face asta?"
    )

    media_info = MediaInfo(
        path=Path("in/clip.mp4"),
        duration_s=10.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_video=True,
        has_audio=True,
        has_alpha=False,
        codec_video="h264",
        codec_audio="aac",
        pixel_format="yuv420p",
    )

    result = brain.reason(
        prompt,
        Path("in/clip.mp4"),
        Path("out/studio_render_clip_v2.mp4"),
        media_info=media_info,
        previous_spec=base_spec,
    )

    assert result.executable is True
    assert result.spec is not None

    # Continuity: settings from base_spec must be preserved
    assert result.spec.video.aspect == "9:16"
    assert result.spec.video.smart_reframe is True
    assert result.spec.video.subtitles.enabled is True
    assert result.spec.video.progress_bar is True
    assert result.spec.audio.clean_speech is True
    assert result.spec.audio.master_profile == "podcast"

    # Negation: punch zoom must be disabled
    assert result.spec.video.punch_zoom is False

    # Camera defect correction: 2-pass VidStab must be enabled
    assert result.spec.video.stabilize is True

    # Explanation must clearly address continuity, zoom negation, and stabilization
    assert "Păstrez setările" in result.reply or "Am înțeles contextul" in result.reply
    assert "VidStab" in result.reply or "stabiliz" in result.reply.lower()


def test_cognitive_brain_camera_stabilization_trigger() -> None:
    brain = CognitiveBrain(LLMConfig(provider="local"))

    prompt = "stabilizează filmarea deoarece camera tremură și se vede prost când am dat zoom out"
    result = brain.reason(
        prompt,
        Path("in/raw.mp4"),
        Path("out/stab.mp4"),
    )

    assert result.executable is True
    assert result.spec is not None
    assert result.spec.video.stabilize is True
    assert result.spec.video.punch_zoom is False


def test_cognitive_brain_negation_triggers() -> None:
    brain = CognitiveBrain(LLMConfig(provider="local"))

    # Test "fără zoom"
    res1 = brain.reason(
        "Fă un short 9:16 fără zoom și cu subtitrări",
        Path("in/raw.mp4"),
        Path("out/v1.mp4"),
    )
    assert res1.spec is not None
    assert res1.spec.video.aspect == "9:16"
    assert res1.spec.video.punch_zoom is False
    assert res1.spec.video.subtitles.enabled is True

    # Test "scoate subtitrările"
    base = EditSpec(
        source="in/raw.mp4",
        output="out/v1.mp4",
        video=VideoEditSpec(subtitles=SubtitleEditSpec(enabled=True)),
    )
    res2 = brain.reason(
        "scoate subtitrările și lasă restul la fel ca înainte",
        Path("in/raw.mp4"),
        Path("out/v2.mp4"),
        previous_spec=base,
    )
    assert res2.spec is not None
    assert res2.spec.video.subtitles.enabled is False


def test_cognitive_brain_silent_media_handling() -> None:
    brain = CognitiveBrain(LLMConfig(provider="local"))

    silent_info = MediaInfo(
        path=Path("in/silent.mp4"),
        duration_s=5.0,
        width=1280,
        height=720,
        fps=30.0,
        has_video=True,
        has_audio=False,
        has_alpha=False,
        codec_video="h264",
        codec_audio=None,
        pixel_format="yuv420p",
    )

    result = brain.reason(
        "curăță vocea podcast, taie pauzele și fă un short 9:16",
        Path("in/silent.mp4"),
        Path("out/silent_short.mp4"),
        media_info=silent_info,
    )

    assert result.spec is not None
    assert result.spec.video.aspect == "9:16"
    # Audio operations must be disabled on silent input
    assert result.spec.audio.clean_speech is False
    assert result.spec.audio.silence_trim is False
    assert result.spec.audio.master_profile is None


def test_cognitive_brain_greeting() -> None:
    brain = CognitiveBrain(LLMConfig(provider="local"))
    res = brain.reason("Salut, cum mă poți ajuta?", None, None)
    assert res.intent == "greeting"
    assert res.executable is False
    assert "Media Lab" in res.reply
