"""Recipe: smart vertical reframe with subject tracking, center cropping, and split-blur.

Reframes horizontal or widescreen media to vertical formats (9:16, 1:1, 4:5)
using salience/face center-of-interest analysis, centered crop, or dynamic split-blur backdrop.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
from ..errors import ValidationError
from ..ffmpeg import run_ffmpeg
from ..paths import ensure_readable_source, ensure_writable_output
from ..probe import MediaInfo, probe
from ..verify import ASPECT_RATIOS, Expectations, verify_render

DEFAULT_REFRAME_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "4:3": (1440, 1080),
    "16:9": (1920, 1080),
}

VALID_REFRAME_MODES: frozenset[str] = frozenset(
    {"smart", "center", "split", "track", "dynamic", "follow"}
)


@dataclass(frozen=True, slots=True)
class ReframeResult:
    """Outcome of reframing a video."""

    output: Path
    media: MediaInfo
    target_aspect: str
    mode: str
    crop_box: tuple[int, int, int, int] | None = None


def detect_frame_subject_center(rgb_frame: np.ndarray) -> float:
    """Detect horizontal center of interest [0.0, 1.0] in an RGB image.

    Combines skin-color chrominance and spatial gradient energy to locate
    human subjects or prominent foreground objects.
    """
    if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
        return 0.5

    h, w, _ = rgb_frame.shape
    if h == 0 or w == 0:
        return 0.5

    r = rgb_frame[..., 0].astype(np.float32)
    g = rgb_frame[..., 1].astype(np.float32)
    b = rgb_frame[..., 2].astype(np.float32)

    # Simplified YCbCr conversion for skin detection
    # Cb = 128 - 0.168736*R - 0.331264*G + 0.5*B
    # Cr = 128 + 0.5*R - 0.418688*G - 0.081312*B
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    skin_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
    skin_cols = np.sum(skin_mask, axis=0)

    # Column luminance energy / contrast
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    lum_variance = np.std(lum, axis=0)

    # Combined column weights
    weights = (skin_cols * 3.0) + lum_variance
    total_weight = float(np.sum(weights))

    if total_weight <= 1e-4:
        return 0.5

    indices = np.arange(w, dtype=np.float32)
    center_idx = float(np.sum(indices * weights) / total_weight)
    return float(np.clip(center_idx / w, 0.0, 1.0))


def analyze_video_subject_timed_trajectory(
    source: Path,
    config: Config,
    *,
    fps_sample: float = 2.0,
    max_samples: int = 40,
) -> list[tuple[float, float]]:
    """Sample video frames across timeline and detect (timestamp_s, center_x) pairs.

    Returns:
        List of (time_s, center_x) where center_x is in [0.0, 1.0].
    """
    cmd = [
        str(config.ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-i",
        str(source),
        "-vf",
        f"fps={fps_sample},scale=160:90",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, check=False, timeout=60)
    except subprocess.TimeoutExpired:
        return [(0.0, 0.5)]

    if proc.returncode != 0 or not proc.stdout:
        return [(0.0, 0.5)]

    frame_bytes = 160 * 90 * 3
    raw = proc.stdout
    total_frames = len(raw) // frame_bytes
    if total_frames == 0:
        return [(0.0, 0.5)]

    step = max(1, total_frames // max_samples)
    points: list[tuple[float, float]] = []

    for idx in range(0, total_frames, step):
        offset = idx * frame_bytes
        frame_data = raw[offset : offset + frame_bytes]
        if len(frame_data) < frame_bytes:
            break
        arr = np.frombuffer(frame_data, dtype=np.uint8).reshape((90, 160, 3))
        center_x = detect_frame_subject_center(arr)
        time_s = round(idx / fps_sample, 2)
        points.append((time_s, center_x))

    return points if points else [(0.0, 0.5)]


def analyze_video_subject_trajectory(
    source: Path,
    config: Config,
    *,
    fps_sample: float = 1.0,
    max_samples: int = 30,
) -> list[float]:
    """Sample video frames across timeline and detect horizontal subject centers."""
    timed = analyze_video_subject_timed_trajectory(
        source, config, fps_sample=fps_sample, max_samples=max_samples
    )
    return [p[1] for p in timed]


def smooth_timed_trajectory(
    points: list[tuple[float, float]],
    smoothing_window_s: float = 1.5,
) -> list[tuple[float, float]]:
    """Apply temporal Gaussian smoothing to eliminate micro-jitters."""
    if len(points) <= 2:
        return points

    times = [p[0] for p in points]
    values = [p[1] for p in points]
    smoothed: list[float] = []

    for t in times:
        weights = [
            float(np.exp(-0.5 * ((t - other_t) / max(0.2, smoothing_window_s)) ** 2))
            for other_t in times
        ]
        total_w = sum(weights)
        smoothed_val = sum(w * v for w, v in zip(weights, values, strict=True)) / total_w
        smoothed.append(float(smoothed_val))

    return list(zip(times, smoothed, strict=True))


def build_dynamic_crop_expression(
    sw: int,
    cw: int,
    points: list[tuple[float, float]],
    deadband_threshold: float = 0.03,
) -> str:
    """Build an FFmpeg piecewise interpolation expression for dynamic crop x coordinate.

    If horizontal subject movement across the clip is within the deadband threshold,
    locks to the median position to prevent unnecessary drift on static shots.
    Otherwise, generates an easing piecewise linear expression that tracks the
    subject's movement with even pixel alignment for codec compatibility.
    """
    max_crop_x = sw - cw
    if max_crop_x <= 0 or not points:
        return "0"

    xs = [p[1] for p in points]
    # If the subject doesn't move significantly across the video, lock to median
    if (max(xs) - min(xs)) < deadband_threshold:
        median_x = float(np.median(xs))
        fixed_x = int(round(median_x * sw - cw / 2))
        fixed_x = max(0, min(max_crop_x, fixed_x))
        fixed_x = fixed_x - (fixed_x % 2)
        return str(fixed_x)

    # Downsample points so FFmpeg expression stays compact and smooth
    filtered_points: list[tuple[float, float]] = [points[0]]
    for pt in points[1:]:
        last_t, last_x = filtered_points[-1]
        if (pt[0] - last_t >= 0.8) or abs(pt[1] - last_x) >= 0.04:
            filtered_points.append(pt)
    if filtered_points[-1] != points[-1]:
        filtered_points.append(points[-1])

    crop_keyframes: list[tuple[float, int]] = []
    for t, norm_x in filtered_points:
        cx = int(round(norm_x * sw - cw / 2))
        cx = max(0, min(max_crop_x, cx))
        cx = cx - (cx % 2)
        crop_keyframes.append((t, cx))

    if len(crop_keyframes) == 1:
        return str(crop_keyframes[0][1])

    # Build nested if expression:
    # if(lt(t, t1), (cx0 + (cx1-cx0)*(t-t0)/(t1-t0)), if(lt(t, t2), ...))
    expr = str(crop_keyframes[-1][1])
    for i in range(len(crop_keyframes) - 2, -1, -1):
        t0, cx0 = crop_keyframes[i]
        t1, cx1 = crop_keyframes[i + 1]
        dt = max(0.01, round(t1 - t0, 3))
        if cx0 == cx1:
            segment = f"{cx0}"
        else:
            diff = cx1 - cx0
            segment = f"({cx0}+{diff}*(t-{t0:.2f})/{dt:.2f})"
        expr = f"if(lt(t,{t1:.2f}),{segment},{expr})"

    return f"clip(trunc(({expr})/2)*2,0,{max_crop_x})"


def calculate_crop_box(
    src_width: int,
    src_height: int,
    target_aspect_ratio: float,
    subject_center_x: float = 0.5,
) -> tuple[int, int, int, int]:
    """Compute (x, y, width, height) crop coordinates for an aspect ratio
    and subject center.
    """
    src_ratio = src_width / src_height

    if src_ratio > target_aspect_ratio:
        # Source is wider than target aspect ratio (e.g. 16:9 to 9:16)
        crop_h = src_height
        crop_w = int(round(crop_h * target_aspect_ratio))
        # Ensure even
        crop_w = crop_w - (crop_w % 2)
        crop_h = crop_h - (crop_h % 2)

        # Center crop horizontally around subject center
        desired_x = int(round(subject_center_x * src_width - crop_w / 2))
        crop_x = max(0, min(src_width - crop_w, desired_x))
        crop_x = crop_x - (crop_x % 2)
        crop_y = 0
    else:
        # Source is taller than target aspect ratio (e.g. 9:16 to 16:9)
        crop_w = src_width
        crop_h = int(round(crop_w / target_aspect_ratio))
        crop_w = crop_w - (crop_w % 2)
        crop_h = crop_h - (crop_h % 2)

        crop_x = 0
        crop_y = max(0, min(src_height - crop_h, (src_height - crop_h) // 2))
        crop_y = crop_y - (crop_y % 2)

    return (crop_x, crop_y, crop_w, crop_h)


def smart_reframe(
    source: Path | str,
    output: Path | str,
    config: Config,
    *,
    target_aspect: str = "9:16",
    mode: str = "smart",
    target_width: int | None = None,
    target_height: int | None = None,
    force: bool = False,
) -> ReframeResult:
    """Reframe video to a target aspect ratio using smart tracking, center crop, or split blur.

    Args:
        source: Input video path.
        output: Destination video path.
        config: System configuration.
        target_aspect: Target aspect ratio preset ("9:16", "1:1", "4:5", "16:9", "4:3").
        mode: Reframe mode: "smart" (tracks subject), "center" (fixed center),
            or "split" (blurred background fill with centered foreground).
        target_width: Explicit output width (optional, defaults to preset resolution).
        target_height: Explicit output height (optional, defaults to preset resolution).
        force: Overwrite output if it already exists.

    Returns:
        ReframeResult with media info and crop coordinates.
    """
    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    if target_aspect not in ASPECT_RATIOS:
        raise ValidationError(
            f"unsupported target_aspect {target_aspect!r}, supported: {sorted(ASPECT_RATIOS)}"
        )

    if mode not in VALID_REFRAME_MODES:
        raise ValidationError(
            f"unsupported reframe mode {mode!r}, supported: {sorted(VALID_REFRAME_MODES)}"
        )

    source_info = probe(resolved_source, config)
    if not source_info.has_video or source_info.width is None or source_info.height is None:
        raise ValidationError(f"source {resolved_source} has no valid video stream")

    sw = source_info.width
    sh = source_info.height
    aspect_ratio_num = ASPECT_RATIOS[target_aspect]

    # Resolve target dimensions
    default_dims = DEFAULT_REFRAME_RESOLUTIONS.get(target_aspect, (1080, 1920))
    tw = target_width if target_width is not None else default_dims[0]
    th = target_height if target_height is not None else default_dims[1]

    # Ensure even dimensions for standard h264/hevc encoders
    tw = tw - (tw % 2)
    th = th - (th % 2)

    crop_box: tuple[int, int, int, int] | None = None

    if mode == "split":
        # Split-blur mode: blurred scaled background with centered foreground
        filtergraph = (
            f"[0:v]split=2[bg_in][fg_in];"
            f"[bg_in]scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th},boxblur=25:5[bg];"
            f"[fg_in]scale={tw}:{th}:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[out_v]"
        )
    elif mode in ("track", "dynamic", "follow"):
        timed_points = analyze_video_subject_timed_trajectory(resolved_source, config)
        smoothed_points = smooth_timed_trajectory(timed_points)
        median_x = float(np.median([p[1] for p in smoothed_points]))
        cx_med, cy, cw, ch = calculate_crop_box(sw, sh, aspect_ratio_num, median_x)
        crop_box = (cx_med, cy, cw, ch)
        dynamic_x_expr = build_dynamic_crop_expression(sw, cw, smoothed_points)
        filtergraph = f"[0:v]crop={cw}:{ch}:{dynamic_x_expr}:{cy},scale={tw}:{th}[out_v]"
    else:
        if mode == "smart":
            centers = analyze_video_subject_trajectory(resolved_source, config)
            # Use median centroid for rock-solid stability without jitters
            subject_x = float(np.median(centers))
        else:
            subject_x = 0.5

        cx, cy, cw, ch = calculate_crop_box(sw, sh, aspect_ratio_num, subject_x)
        crop_box = (cx, cy, cw, ch)
        filtergraph = f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={tw}:{th}[out_v]"

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
        cmd.extend(["-map", "0:a", "-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.append("-an")

    cmd.append(str(resolved_output))
    run_ffmpeg(cmd, config)

    final_info = verify_render(
        resolved_output,
        config,
        Expectations(
            aspect_ratio=target_aspect,
            requires_video=True,
            requires_audio=source_info.has_audio,
        ),
    )

    return ReframeResult(
        output=resolved_output,
        media=final_info,
        target_aspect=target_aspect,
        mode=mode,
        crop_box=crop_box,
    )
