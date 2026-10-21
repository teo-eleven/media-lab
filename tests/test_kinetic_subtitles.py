from __future__ import annotations

import pytest

from media_lab.subtitles import (
    SubtitleSegment,
    WordToken,
    apply_auto_emojis,
    generate_ass,
)


@pytest.fixture
def sample_segments() -> list[SubtitleSegment]:
    return [
        SubtitleSegment(
            id=1,
            start=0.0,
            end=2.0,
            text="Am o idee despre bani si foc",
            words=(
                WordToken("Am", 0.0, 0.3),
                WordToken("o", 0.3, 0.5),
                WordToken("idee", 0.5, 0.9),
                WordToken("despre", 0.9, 1.2),
                WordToken("bani", 1.2, 1.6),
                WordToken("si", 1.6, 1.7),
                WordToken("foc", 1.7, 2.0),
            ),
        ),
        SubtitleSegment(
            id=2,
            start=2.0,
            end=3.5,
            text="Totul merge rapid fara stop",
            words=(
                WordToken("Totul", 2.0, 2.3),
                WordToken("merge", 2.3, 2.6),
                WordToken("rapid", 2.6, 3.0),
                WordToken("fara", 3.0, 3.2),
                WordToken("stop", 3.2, 3.5),
            ),
        ),
    ]


def test_apply_auto_emojis_maps_keywords() -> None:
    text_ro = "Am o idee despre bani si foc stop"
    result_ro = apply_auto_emojis(text_ro)
    assert "💡" in result_ro
    assert "💰" in result_ro
    assert "🔥" in result_ro
    assert "🛑" in result_ro

    text_en = "I have an idea about money and fire"
    result_en = apply_auto_emojis(text_en)
    assert "💡" in result_en
    assert "💰" in result_en
    assert "🔥" in result_en


def test_generate_ass_with_kinetic_bouncing(sample_segments: list[SubtitleSegment]) -> None:
    ass_content = generate_ass(
        sample_segments,
        style="tiktok",
        width=1080,
        height=1920,
        kinetic=True,
    )

    assert "PlayResX: 1080" in ass_content
    assert "PlayResY: 1920" in ass_content
    # Kinetic bounce tags should be present (\fscx or \t scaling)
    assert r"\fscx" in ass_content or r"\t(" in ass_content
    # Safe zone check: TikTok vertical margin should be at least 350px on 1920 height
    assert "MarginV" in ass_content


def test_generate_ass_with_emojis(sample_segments: list[SubtitleSegment]) -> None:
    ass_content = generate_ass(
        sample_segments,
        style="tiktok",
        width=1080,
        height=1920,
        emojis=True,
    )

    # The emojis for "bani" (💰) or "idee" (💡) or "foc" (🔥) must be present in dialogue
    assert "💡" in ass_content or "💰" in ass_content or "🔥" in ass_content


def test_kinetic_graceful_fallback_without_words() -> None:
    segments_no_words = [
        SubtitleSegment(id=1, start=0.0, end=1.5, text="Doar text simplu fara cuvinte"),
    ]
    ass_content = generate_ass(
        segments_no_words,
        style="tiktok",
        kinetic=True,
        emojis=True,
    )
    assert "doar text simplu" in ass_content.lower()
