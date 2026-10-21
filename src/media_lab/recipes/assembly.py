"""Recipe: Multi-clip timeline assembly with cinematic video and audio transitions.

Concatenates multiple video clips sequentially with smooth xfade video transitions
(fade, wipeleft, wiperight, dissolve, fadeblack, slideleft, slideright) and
equal-power acrossfade audio transitions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..errors import ValidationError
from ..ffmpeg import run_ffmpeg
from ..paths import ensure_readable_source, ensure_writable_output
from ..probe import MediaInfo, probe
from ..verify import ASPECT_RATIOS, Expectations, verify_render

DEFAULT_ASSEMBLY_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "4:3": (1440, 1080),
    "21:9": (2560, 1080),
}

VALID_TRANSITIONS: frozenset[str] = frozenset(
    {
        "fade",
        "fadeblack",
        "fadewhite",
        "dissolve",
        "wipeleft",
        "wiperight",
        "wipeup",
        "wipedown",
        "slideleft",
        "slideright",
        "circlecrop",
        "rectcrop",
        "pixelize",
        "horzopen",
        "vertopen",
    }
)


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    """Outcome of multi-clip timeline assembly."""

    output: Path
    media: MediaInfo
    clips_count: int
    transition: str
    total_duration_s: float


def assemble_clips(
    sources: Sequence[Path | str],
    output: Path | str,
    config: Config,
    *,
    transition: str = "fade",
    transition_duration_s: float = 0.75,
    aspect: str = "16:9",
    force: bool = False,
) -> AssemblyResult:
    """Assemble multiple clips sequentially with seamless transitions.

    Args:
        sources: Sequence of at least 2 media files.
        output: Destination video file.
        config: Project configuration.
        transition: Name of the xfade transition (e.g. 'fade', 'dissolve', 'wipeleft').
        transition_duration_s: Overlap transition duration in seconds (default 0.75s).
        aspect: Target aspect ratio ('16:9', '9:16', '1:1', '4:5', '4:3').
        force: Overwrite existing destination.

    Returns:
        AssemblyResult with rendered metadata.
    """
    if len(sources) < 2:
        raise ValidationError(f"assembly requires at least 2 input clips, got {len(sources)}")
    if transition not in VALID_TRANSITIONS:
        raise ValidationError(
            f"unsupported transition {transition!r}, supported: {sorted(VALID_TRANSITIONS)}"
        )
    if transition_duration_s <= 0:
        raise ValidationError(f"transition_duration_s must be > 0, got {transition_duration_s}")
    if aspect not in ASPECT_RATIOS:
        raise ValidationError(f"unsupported aspect {aspect!r}, supported: {sorted(ASPECT_RATIOS)}")

    resolved_sources = [ensure_readable_source(s) for s in sources]
    resolved_output = ensure_writable_output(output, config, force=force)

    # Probe all inputs to ensure they have video streams and check durations
    probed_infos: list[MediaInfo] = []
    for s in resolved_sources:
        info = probe(s, config)
        if not info.has_video:
            raise ValidationError(f"clip {s} has no video stream")
        if info.duration_s <= transition_duration_s:
            raise ValidationError(
                f"clip {s} duration ({info.duration_s}s) must be longer than "
                f"transition ({transition_duration_s}s)"
            )
        probed_infos.append(info)

    tw, th = DEFAULT_ASSEMBLY_RESOLUTIONS.get(aspect, (1920, 1080))
    all_have_audio = all(info.has_audio for info in probed_infos)

    cmd: list[str] = []
    for s in resolved_sources:
        cmd.extend(["-i", str(s)])

    filter_chains: list[str] = []

    # 1. Normalize each video stream (scale + pad to target aspect, fps=30, sar=1, yuv420p)
    for i in range(len(resolved_sources)):
        filter_chains.append(
            f"[{i}:v]scale={tw}:{th}:force_original_aspect_ratio=decrease,"
            f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[v_norm_{i}]"
        )
        if all_have_audio:
            filter_chains.append(
                f"[{i}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                f"aresample=async=1[a_norm_{i}]"
            )

    # 2. Chain xfade transitions
    # offset_1 = d_0 - delta
    # offset_k = offset_{k-1} + d_{k-1} - delta
    current_v = "v_norm_0"
    current_a = "a_norm_0" if all_have_audio else None

    # Cumulative offset tracker
    cum_offset = probed_infos[0].duration_s - transition_duration_s

    for k in range(1, len(resolved_sources)):
        next_v = f"v_norm_{k}"
        out_v = f"v_xfade_{k}"
        filter_chains.append(
            f"[{current_v}][{next_v}]xfade=transition={transition}:"
            f"duration={transition_duration_s:.3f}:offset={cum_offset:.3f}[{out_v}]"
        )
        current_v = out_v

        if all_have_audio and current_a is not None:
            next_a = f"a_norm_{k}"
            out_a = f"a_xfade_{k}"
            filter_chains.append(
                f"[{current_a}][{next_a}]acrossfade=d={transition_duration_s:.3f}:"
                f"c1=qsin:c2=qsin[{out_a}]"
            )
            current_a = out_a

        if k < len(resolved_sources) - 1:
            cum_offset += probed_infos[k].duration_s - transition_duration_s

    filtergraph = ";".join(filter_chains)
    cmd.extend(
        [
            "-filter_complex",
            filtergraph,
            "-map",
            f"[{current_v}]",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
    )

    if current_a is not None:
        cmd.extend(["-map", f"[{current_a}]", "-c:a", "aac", "-b:a", "256k"])
    else:
        cmd.append("-an")

    cmd.append(str(resolved_output))
    run_ffmpeg(cmd, config)

    # Compute expected total duration: sum(d_i) - (N-1)*delta
    expected_duration = (
        sum(info.duration_s for info in probed_infos)
        - (len(resolved_sources) - 1) * transition_duration_s
    )

    final_info = verify_render(
        resolved_output,
        config,
        Expectations(
            duration_s=round(expected_duration, 2),
            width=tw,
            height=th,
            requires_video=True,
            requires_audio=all_have_audio,
        ),
    )

    return AssemblyResult(
        output=resolved_output,
        media=final_info,
        clips_count=len(resolved_sources),
        transition=transition,
        total_duration_s=round(final_info.duration_s, 2),
    )
