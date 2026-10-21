"""Recipe: camera motion control and dynamic subject tracking.

Supports dynamic kinetic subject tracking (following person across time with camera inertia),
cinematic push-in zooms, pull-outs, directional pans, and organic handheld drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..errors import ValidationError
from ..ffmpeg import run_ffmpeg
from ..paths import ensure_readable_source, ensure_writable_output
from ..probe import MediaInfo, probe
from ..verify import Expectations, verify_render
from .smart_reframe import (
    analyze_video_subject_timed_trajectory,
    build_dynamic_crop_expression,
    smooth_timed_trajectory,
)

VALID_CAMERA_MOTIONS: frozenset[str] = frozenset(
    {
        "track",
        "follow",
        "push_in",
        "slow_zoom_in",
        "pull_out",
        "slow_zoom_out",
        "pan_left",
        "pan_right",
        "handheld",
    }
)


@dataclass(frozen=True, slots=True)
class CameraMotionResult:
    """Outcome of applying dynamic camera motion."""

    output: Path
    media: MediaInfo
    motion: str


def apply_camera_motion(
    source: Path | str,
    output: Path | str,
    config: Config,
    *,
    motion: str = "track",
    zoom_factor: float = 1.15,
    smoothing: float = 1.5,
    force: bool = False,
) -> CameraMotionResult:
    """Apply cinematic camera motion or kinetic subject tracking to a video.

    Args:
        source: Input video path.
        output: Destination video path.
        config: System configuration.
        motion: Motion type:
            - 'track' / 'follow': Dynamically pans frame following subject trajectory across time.
            - 'push_in' / 'slow_zoom_in': Cinematic slow push-in zoom towards center.
            - 'pull_out' / 'slow_zoom_out': Cinematic slow pull-out zoom revealing environment.
            - 'pan_left': Smooth horizontal camera glide from right to left.
            - 'pan_right': Smooth horizontal camera glide from left to right.
            - 'handheld': Subtle organic handheld camera movement.
        zoom_factor: Digital crop scale factor for panning or zoom movements (default 1.15x).
        smoothing: Temporal smoothing window in seconds for subject tracking (default 1.5s).
        force: Overwrite output if it already exists.

    Returns:
        CameraMotionResult with output path, media metadata, and motion used.
    """
    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    motion_lower = motion.lower().strip()
    if motion_lower not in VALID_CAMERA_MOTIONS:
        raise ValidationError(
            f"unsupported camera motion {motion!r}, supported: {sorted(VALID_CAMERA_MOTIONS)}"
        )

    source_info = probe(resolved_source, config)
    if not source_info.has_video or source_info.width is None or source_info.height is None:
        raise ValidationError(f"source {resolved_source} has no valid video stream")

    sw = source_info.width
    sh = source_info.height
    dur = max(0.5, source_info.duration_s)
    fps = max(15.0, source_info.fps if source_info.fps else 30.0)

    # Compute crop dimensions constrained by zoom_factor
    zf = max(1.02, min(2.0, zoom_factor))
    cw = int(round(sw / zf))
    ch = int(round(sh / zf))
    cw = cw - (cw % 2)
    ch = ch - (ch % 2)

    max_x = max(0, sw - cw)
    max_y = max(0, sh - ch)

    if motion_lower in ("track", "follow"):
        # Dynamic kinetic subject tracking
        timed = analyze_video_subject_timed_trajectory(resolved_source, config)
        smoothed = smooth_timed_trajectory(timed, smoothing_window_s=smoothing)
        x_expr = build_dynamic_crop_expression(sw, cw, smoothed)
        cy = max_y // 2
        cy = cy - (cy % 2)
        filtergraph = f"[0:v]crop=w={cw}:h={ch}:x='{x_expr}':y={cy},scale={sw}:{sh},setsar=1[out_v]"

    elif motion_lower in ("push_in", "slow_zoom_in"):
        d_frames = max(1, int(round(dur * fps)))
        step = (zf - 1.0) / d_frames
        filtergraph = (
            f"[0:v]zoompan=z='min(1.0+{step:.6f}*on,{zf:.3f})':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={sw}x{sh}:fps={fps:.2f},"
            f"setsar=1[out_v]"
        )

    elif motion_lower in ("pull_out", "slow_zoom_out"):
        d_frames = max(1, int(round(dur * fps)))
        step = (zf - 1.0) / d_frames
        filtergraph = (
            f"[0:v]zoompan=z='max({zf:.3f}-{step:.6f}*on,1.001)':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={sw}x{sh}:fps={fps:.2f},"
            f"setsar=1[out_v]"
        )

    elif motion_lower == "pan_left":
        cy = max_y // 2
        cy = cy - (cy % 2)
        filtergraph = (
            f"[0:v]crop=w={cw}:h={ch}:x='clip(trunc(({max_x}*(1-t/{dur:.3f}))/2)*2,0,{max_x})':"
            f"y={cy},scale={sw}:{sh},setsar=1[out_v]"
        )

    elif motion_lower == "pan_right":
        cy = max_y // 2
        cy = cy - (cy % 2)
        filtergraph = (
            f"[0:v]crop=w={cw}:h={ch}:x='clip(trunc(({max_x}*(t/{dur:.3f}))/2)*2,0,{max_x})':"
            f"y={cy},scale={sw}:{sh},setsar=1[out_v]"
        )

    elif motion_lower == "handheld":
        # Subtle organic handheld drift (using gentle 1.05x crop)
        hh_cw = int(round(sw / 1.05))
        hh_ch = int(round(sh / 1.05))
        hh_cw = hh_cw - (hh_cw % 2)
        hh_ch = hh_ch - (hh_ch % 2)
        hh_max_x = max(0, sw - hh_cw)
        hh_max_y = max(0, sh - hh_ch)
        x0 = hh_max_x // 2
        y0 = hh_max_y // 2
        amp_x = hh_max_x * 0.4
        amp_y = hh_max_y * 0.4
        x_expr = f"clip(trunc(({x0}+{amp_x:.1f}*(sin(1.1*t)+0.5*cos(2.3*t)))/2)*2,0,{hh_max_x})"
        y_expr = f"clip(trunc(({y0}+{amp_y:.1f}*(cos(0.9*t)+0.5*sin(1.7*t)))/2)*2,0,{hh_max_y})"
        filtergraph = (
            f"[0:v]crop=w={hh_cw}:h={hh_ch}:x='{x_expr}':y='{y_expr}',"
            f"scale={sw}:{sh},setsar=1[out_v]"
        )

    else:
        raise ValidationError(f"unhandled camera motion: {motion}")

    cmd: list[str] = [
        "-i",
        str(resolved_source),
        "-filter_complex",
        filtergraph,
        "-map",
        "[out_v]",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
    ]

    if source_info.has_audio:
        cmd.extend(["-map", "0:a", "-c:a", "copy"])
    else:
        cmd.append("-an")

    cmd.append(str(resolved_output))
    run_ffmpeg(cmd, config)

    final_info = verify_render(
        resolved_output,
        config,
        Expectations(
            width=sw,
            height=sh,
            requires_video=True,
            requires_audio=source_info.has_audio,
        ),
    )

    return CameraMotionResult(
        output=resolved_output,
        media=final_info,
        motion=motion_lower,
    )
