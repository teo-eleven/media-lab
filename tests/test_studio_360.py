"""Tests for Studio 360 features: Assembly, Speed Ramping, Progress Bar, Beat Sync, and Narrator."""

from __future__ import annotations

from pathlib import Path

import pytest

from media_lab.cli import main
from media_lab.config import Config
from media_lab.ffmpeg import run_ffmpeg
from media_lab.probe import probe
from media_lab.recipes.assembly import assemble_clips
from media_lab.recipes.beat_sync import detect_beats
from media_lab.recipes.narrator import generate_narration
from media_lab.recipes.progress_bar import add_progress_bar
from media_lab.recipes.speed import change_speed


def _generate_synthetic_clip(
    path: Path, config: Config, duration: float = 3.0, color: str = "blue"
) -> Path:
    """Generate a test video clip with audio."""
    cmd = [
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s=320x240:r=30:d={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=44100:duration={duration}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(path),
    ]
    run_ffmpeg(cmd, config)
    return path


def test_assemble_clips(config: Config) -> None:
    clip1 = _generate_synthetic_clip(config.in_dir / "clip1.mp4", config, duration=3.0, color="red")
    clip2 = _generate_synthetic_clip(
        config.in_dir / "clip2.mp4", config, duration=3.0, color="green"
    )
    out = config.out_dir / "assembled.mp4"

    res = assemble_clips(
        [clip1, clip2],
        out,
        config,
        transition="fade",
        transition_duration_s=0.5,
        aspect="16:9",
        force=True,
    )

    assert res.output.exists()
    assert res.clips_count == 2
    assert res.transition == "fade"
    # 3.0 + 3.0 - 0.5 = 5.5s
    assert abs(res.total_duration_s - 5.5) < 0.3
    info = probe(out, config)
    assert info.has_video
    assert info.width == 1920
    assert info.height == 1080


def test_change_speed_slow_and_fast(config: Config) -> None:
    clip = _generate_synthetic_clip(config.in_dir / "speed_src.mp4", config, duration=2.0)
    out_slow = config.out_dir / "speed_slow.mp4"
    out_fast = config.out_dir / "speed_fast.mp4"

    # Slow motion (0.5x) -> duration doubles to ~4.0s
    res_slow = change_speed(clip, out_slow, config, speed=0.5, force=True)
    assert res_slow.output.exists()
    assert abs(res_slow.new_duration_s - 4.0) < 0.3

    # Fast forward (2.0x) -> duration halves to ~1.0s
    res_fast = change_speed(clip, out_fast, config, speed=2.0, force=True)
    assert res_fast.output.exists()
    assert abs(res_fast.new_duration_s - 1.0) < 0.3


def test_add_progress_bar_bottom_and_top(config: Config) -> None:
    clip = _generate_synthetic_clip(config.in_dir / "pb_src.mp4", config, duration=2.0)
    out_bottom = config.out_dir / "pb_bottom.mp4"
    out_top = config.out_dir / "pb_top.mp4"

    res_b = add_progress_bar(
        clip, out_bottom, config, position="bottom", height=6, color="yellow", force=True
    )
    assert res_b.output.exists()
    assert res_b.position == "bottom"
    assert res_b.height == 6

    res_t = add_progress_bar(
        clip, out_top, config, position="top", height=8, color="red", force=True
    )
    assert res_t.output.exists()
    assert res_t.position == "top"
    assert res_t.height == 8


def test_detect_beats(config: Config) -> None:
    # Generate audio with 3 distinct beep pulses
    audio_path = config.in_dir / "beats.wav"
    filtergraph = (
        "sine=f=440:d=0.2[b1];"
        "aevalsrc=0:d=0.5[s1];"
        "sine=f=880:d=0.2[b2];"
        "aevalsrc=0:d=0.5[s2];"
        "sine=f=440:d=0.2[b3];"
        "[b1][s1][b2][s2][b3]concat=n=5:v=0:a=1[out]"
    )
    cmd = ["-f", "lavfi", "-i", filtergraph, "-c:a", "pcm_s16le", str(audio_path)]
    run_ffmpeg(cmd, config)

    res = detect_beats(audio_path, config, sensitivity=0.8)
    assert res.duration_s > 1.0
    assert res.total_beats >= 1


def test_generate_narration_standalone_and_mux(config: Config) -> None:
    out_audio = config.out_dir / "narrate.wav"
    res_a = generate_narration(
        "Acesta este un test de narator.",
        out_audio,
        config,
        force=True,
    )
    assert res_a.output.exists()
    assert res_a.duration_s > 0.0

    # Test attaching to video
    clip = _generate_synthetic_clip(config.in_dir / "narrate_bg.mp4", config, duration=3.0)
    out_video = config.out_dir / "narrate_video.mp4"
    res_v = generate_narration(
        "Acesta este naratorul peste clip.",
        out_video,
        config,
        video_source=clip,
        start_s=0.5,
        force=True,
    )
    assert res_v.output.exists()
    info = probe(out_video, config)
    assert info.has_video
    assert info.has_audio


def test_cli_phase5_commands(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("media_lab.cli.load_config", lambda: config)
    clip = _generate_synthetic_clip(config.in_dir / "cli_src.mp4", config, duration=2.0)
    out_spd = config.out_dir / "cli_speed.mp4"
    out_pb = config.out_dir / "cli_pb.mp4"

    # CLI speed
    ret = main(["speed", str(clip), "-o", str(out_spd), "--speed", "1.5", "--force"])
    assert ret == 0
    assert out_spd.exists()

    # CLI progress-bar
    ret_pb = main(["progress-bar", str(clip), "-o", str(out_pb), "--color", "red", "--force"])
    assert ret_pb == 0
    assert out_pb.exists()

    # CLI beat-sync
    ret_bt = main(["beat-sync", str(clip), "--json"])
    assert ret_bt == 0
