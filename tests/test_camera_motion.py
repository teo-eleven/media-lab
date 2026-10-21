"""Tests for camera motion recipe, dynamic subject tracking, and smart reframe modes."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from media_lab.config import Config
from media_lab.errors import ValidationError
from media_lab.llm import CognitiveBrain, LLMConfig
from media_lab.mcp_server import handle_rpc_request
from media_lab.recipes.camera_motion import (
    apply_camera_motion,
)
from media_lab.recipes.smart_reframe import smart_reframe


def _generate_synthetic_clip(
    path: Path, config: Config, duration: float = 2.0, with_audio: bool = True
) -> Path:
    """Generate a clean synthetic test video with horizontal movement."""
    cmd = [
        str(config.ffmpeg),
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size=640x360:rate=30",
    ]
    if with_audio:
        cmd.extend(["-f", "lavfi", "-i", f"sine=frequency=1000:duration={duration}"])
        cmd.extend(["-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", str(path)])
    else:
        cmd.extend(["-c:v", "libx264", "-an", "-pix_fmt", "yuv420p", str(path)])

    subprocess.run(cmd, check=True, capture_output=True)
    return path


def test_camera_motion_modes(config: Config) -> None:
    src = _generate_synthetic_clip(config.in_dir / "motion_test.mp4", config, duration=1.5)

    # 1. Dynamic Subject Tracking
    out_track = config.out_dir / "out_track.mp4"
    res_track = apply_camera_motion(src, out_track, config, motion="track", force=True)
    assert res_track.output.is_file()
    assert res_track.motion == "track"
    assert res_track.media.has_video is True
    assert res_track.media.has_audio is True

    # 2. Cinematic Push-In Zoom
    out_push = config.out_dir / "out_push.mp4"
    res_push = apply_camera_motion(src, out_push, config, motion="push_in", force=True)
    assert res_push.output.is_file()
    assert res_push.motion == "push_in"

    # 3. Cinematic Pull-Out Zoom
    out_pull = config.out_dir / "out_pull.mp4"
    res_pull = apply_camera_motion(src, out_pull, config, motion="pull_out", force=True)
    assert res_pull.output.is_file()
    assert res_pull.motion == "pull_out"

    # 4. Pan Left
    out_pan_l = config.out_dir / "out_pan_l.mp4"
    res_pan_l = apply_camera_motion(src, out_pan_l, config, motion="pan_left", force=True)
    assert res_pan_l.output.is_file()
    assert res_pan_l.motion == "pan_left"

    # 5. Pan Right
    out_pan_r = config.out_dir / "out_pan_r.mp4"
    res_pan_r = apply_camera_motion(src, out_pan_r, config, motion="pan_right", force=True)
    assert res_pan_r.output.is_file()
    assert res_pan_r.motion == "pan_right"

    # 6. Handheld Organic Drift
    out_hh = config.out_dir / "out_hh.mp4"
    res_hh = apply_camera_motion(src, out_hh, config, motion="handheld", force=True)
    assert res_hh.output.is_file()
    assert res_hh.motion == "handheld"


def test_camera_motion_silent_video(config: Config) -> None:
    src = _generate_synthetic_clip(
        config.in_dir / "motion_silent.mp4", config, duration=1.0, with_audio=False
    )
    out = config.out_dir / "out_silent_track.mp4"

    res = apply_camera_motion(src, out, config, motion="track", force=True)
    assert res.output.is_file()
    assert res.media.has_audio is False
    assert res.media.has_video is True


def test_camera_motion_validation_errors(config: Config) -> None:
    src = _generate_synthetic_clip(config.in_dir / "motion_err.mp4", config, duration=1.0)
    out = config.out_dir / "out_err.mp4"

    with pytest.raises(ValidationError, match="unsupported camera motion"):
        apply_camera_motion(src, out, config, motion="teleport")


def test_smart_reframe_track_mode(config: Config) -> None:
    src = _generate_synthetic_clip(config.in_dir / "wide_track.mp4", config, duration=1.5)
    out = config.out_dir / "reframe_track.mp4"

    res = smart_reframe(
        src,
        out,
        config,
        target_aspect="9:16",
        mode="track",
        target_width=360,
        target_height=640,
        force=True,
    )
    assert res.output.is_file()
    assert res.target_aspect == "9:16"
    assert res.media.width == 360
    assert res.media.height == 640
    assert res.media.has_video is True


def test_mcp_apply_camera_motion(config: Config) -> None:
    src = _generate_synthetic_clip(config.in_dir / "mcp_motion.mp4", config, duration=1.0)
    out = config.out_dir / "mcp_motion_out.mp4"

    req = {
        "jsonrpc": "2.0",
        "id": 101,
        "method": "tools/call",
        "params": {
            "name": "apply_camera_motion",
            "arguments": {
                "source": str(src),
                "output": str(out),
                "motion": "push_in",
                "force": True,
            },
        },
    }
    resp = handle_rpc_request(req, config)
    assert resp is not None
    assert "result" in resp
    assert resp["result"]["isError"] is False
    import json

    res_data = json.loads(resp["result"]["content"][0]["text"])
    assert res_data["motion"] == "push_in"
    assert Path(res_data["output"]).is_file()


def test_cognitive_brain_camera_motion_dialogue() -> None:
    brain = CognitiveBrain(LLMConfig(provider="local"))

    user_prompt = (
        "ar trebui sa faci si niște mișcări consecvente cu camera sa meargă cumva "
        "cu mișcarea ei, poți face asta?"
    )

    result = brain.reason(
        user_prompt,
        Path("in/actor.mp4"),
        Path("out/actor_follow.mp4"),
    )

    assert result.executable is True
    assert result.spec is not None
    assert result.intent == "consultation"

    # Camera motion must be configured to dynamic tracking
    assert result.spec.video.camera_motion == "track"
    assert result.spec.video.reframe_mode == "track"
    assert result.spec.video.stabilize is True

    # Brain explanation must clearly respond in Romanian to user's question
    assert "Da, absolut!" in result.reply
    assert "Kinetic Camera Tracking" in result.reply or "urmărire" in result.reply.lower()
    assert len(result.operations) > 0
