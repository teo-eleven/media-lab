"""Recipe: B-Roll cutaway overlay preserving primary speaker dialogue track.

Allows seamless L-cut / J-cut cutaway insertion of secondary video footage or
stills over a primary talking-head timeline without interrupting the speech audio.
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

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


@dataclass(frozen=True, slots=True)
class BrollCut:
    """A single B-roll cutaway overlay definition."""

    path: Path | str
    start_s: float
    duration_s: float
    transition: str = "cut"  # "cut" or "fade"
    fade_duration_s: float = 0.25
    volume: float = 0.0  # 0.0 = muted cutaway, >0.0 = mixed audio under primary voice


@dataclass(frozen=True, slots=True)
class BrollResult:
    """Outcome of inserting B-roll cutaways."""

    output: Path
    media: MediaInfo
    cuts_count: int


def insert_broll(
    source: Path | str,
    output: Path | str,
    cuts: list[BrollCut],
    config: Config,
    *,
    force: bool = False,
) -> BrollResult:
    """Overlay B-Roll footage or images over a primary video timeline.

    Args:
        source: Base primary video.
        output: Destination video path.
        cuts: List of BrollCut specifications.
        config: System configuration.
        force: Overwrite existing output.

    Returns:
        BrollResult with verified media info.
    """
    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    if not cuts:
        raise ValidationError("at least one BrollCut must be provided")

    source_info = probe(resolved_source, config)
    if not source_info.has_video or source_info.width is None or source_info.height is None:
        raise ValidationError(f"source {resolved_source} has no valid video stream")

    sw = source_info.width
    sh = source_info.height

    # Resolve and validate each cut
    resolved_cuts: list[tuple[Path, float, float, str, float, float]] = []
    for i, cut in enumerate(cuts):
        p = ensure_readable_source(cut.path)
        if cut.start_s < 0:
            raise ValidationError(f"cut {i} start_s must be >= 0, got {cut.start_s}")
        if cut.duration_s <= 0:
            raise ValidationError(f"cut {i} duration_s must be > 0, got {cut.duration_s}")
        if cut.transition not in {"cut", "fade"}:
            raise ValidationError(
                f"cut {i} transition must be 'cut' or 'fade', got {cut.transition!r}"
            )
        if not (0.0 <= cut.volume <= 2.0):
            raise ValidationError(f"cut {i} volume must be between 0.0 and 2.0, got {cut.volume}")
        resolved_cuts.append(
            (p, cut.start_s, cut.duration_s, cut.transition, cut.fade_duration_s, cut.volume)
        )

    cmd: list[str] = ["-i", str(resolved_source)]

    # Add each B-roll asset input
    for p, _start, dur, _trans, _fade, _vol in resolved_cuts:
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            cmd.extend(["-loop", "1", "-t", str(dur + 1.0), "-i", str(p)])
        else:
            cmd.extend(["-i", str(p)])

    # Construct filtergraph
    filter_chains: list[str] = []
    current_v = "0:v"

    for idx, (_p, start_s, dur_s, trans, fade_dur, _vol) in enumerate(resolved_cuts, start=1):
        broll_in = f"{idx}:v"
        broll_proc = f"broll_proc_{idx}"
        overlay_out = f"v_layer_{idx}"

        proc_steps = [
            f"scale={sw}:{sh}:force_original_aspect_ratio=increase",
            f"crop={sw}:{sh}",
            f"trim=duration={dur_s}",
            f"setpts=PTS-STARTPTS+{start_s}/TB",
        ]

        if trans == "fade":
            fd = min(fade_dur, dur_s / 2)
            out_st = max(start_s, start_s + dur_s - fd)
            proc_steps.extend(
                [
                    "format=yuva420p",
                    f"fade=t=in:st={start_s}:d={fd}:alpha=1",
                    f"fade=t=out:st={out_st}:d={fd}:alpha=1",
                ]
            )

        filter_chains.append(f"[{broll_in}]{','.join(proc_steps)}[{broll_proc}]")

        end_s = start_s + dur_s
        filter_chains.append(
            f"[{current_v}][{broll_proc}]overlay=0:0:enable='between(t,{start_s},{end_s})':eof_action=pass[{overlay_out}]"
        )
        current_v = overlay_out

    # Construct audio mixing if needed
    has_broll_audio = any(v > 0.0 for _, _, _, _, _, v in resolved_cuts)
    current_a = "0:a" if source_info.has_audio else None

    if has_broll_audio and source_info.has_audio:
        # Build audio mix
        audio_streams: list[str] = ["[0:a]"]
        for idx, (_p, start_s, dur_s, _trans, _fade, vol) in enumerate(resolved_cuts, start=1):
            if vol > 0.0:
                broll_a = f"{idx}:a"
                broll_adelay = f"adelay_{idx}"
                delay_ms = int(start_s * 1000)
                filter_chains.append(
                    f"[{broll_a}]atrim=0:{dur_s},adelay={delay_ms}|{delay_ms},volume={vol}[{broll_adelay}]"
                )
                audio_streams.append(f"[{broll_adelay}]")

        mix_inputs = "".join(audio_streams)
        filter_chains.append(
            f"{mix_inputs}amix=inputs={len(audio_streams)}:duration=first:dropout_transition=0[mixed_a]"
        )
        current_a = "mixed_a"

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
        cmd.extend(
            [
                "-map",
                f"[{current_a}]" if current_a.startswith("mixed") else "0:a",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
            ]
        )
    else:
        cmd.append("-an")

    cmd.append(str(resolved_output))
    run_ffmpeg(cmd, config)

    final_info = verify_render(
        resolved_output,
        config,
        Expectations(
            duration_s=source_info.duration_s,
            width=sw,
            height=sh,
            requires_video=True,
            requires_audio=source_info.has_audio,
        ),
    )

    return BrollResult(
        output=resolved_output,
        media=final_info,
        cuts_count=len(resolved_cuts),
    )
