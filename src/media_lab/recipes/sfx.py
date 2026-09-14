"""Sound effects (SFX) synthesis, library, and timeline synchronization recipe.

Provides procedural sound effects (whoosh, pop, ding, impact, click) that generate
pure audio with zero external asset dependencies, as well as timeline placement
and mixing of custom SFX files at exact timestamps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..errors import ValidationError
from ..ffmpeg import run_ffmpeg
from ..paths import ensure_readable_source, ensure_writable_output, work_path
from ..probe import MediaInfo, probe
from ..validation import check_range
from ..verify import Expectations, verify_render

SFX_KINDS = ("whoosh", "pop", "ding", "impact", "click")


@dataclass(frozen=True, slots=True)
class SfxCue:
    """A sound effect event on the timeline."""

    kind: str | Path
    at_s: float
    volume: float = 1.0


@dataclass(frozen=True, slots=True)
class SfxResult:
    """Outcome of SFX mixing."""

    output: Path
    media: MediaInfo
    cues_count: int


def generate_procedural_sfx(
    kind: str,
    output: Path | str,
    config: Config,
    *,
    force: bool = False,
) -> Path:
    """Generate a procedural sound effect using mathematical audio synthesis.

    Args:
        kind: One of 'whoosh', 'pop', 'ding', 'impact', 'click'.
        output: Destination WAV file.
        config: Project configuration.
        force: Allow overwrite.
    """
    if kind not in SFX_KINDS:
        raise ValidationError(f"unsupported procedural SFX kind: {kind!r}, choices: {SFX_KINDS}")

    resolved_output = ensure_writable_output(output, config, force=force)

    # ffmpeg mathematical synthesis filtergraphs for each SFX
    if kind == "whoosh":
        # Fast sweeping air whoosh
        graph = (
            "anoisesrc=d=0.35:c=pink:r=44100,"
            "highpass=f=400,lowpass=f=3500,"
            "volume='sin(t/0.35*PI)^2':eval=frame"
        )
    elif kind == "pop":
        # Snappy tactile bubble pop
        graph = "sine=f=800:d=0.08:r=44100,asetrate=44100*0.8,volume='exp(-30*t)':eval=frame"
    elif kind == "ding":
        # High clear chime / notification ding
        graph = "sine=f=1760:d=0.5:r=44100,volume='exp(-6*t)':eval=frame"
    elif kind == "impact":
        # Heavy cinematic bass thump
        graph = "sine=f=65:d=0.6:r=44100,lowpass=f=120,volume='exp(-5*t)':eval=frame"
    else:  # click
        # Mechanical subtle click
        graph = "sine=f=2400:d=0.02:r=44100,volume='exp(-100*t)':eval=frame"

    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            graph,
            "-c:a",
            "pcm_s16le",
            str(resolved_output),
        ],
        config,
    )
    return resolved_output


def add_sfx(
    source: Path | str,
    output: Path | str,
    config: Config,
    cues: Sequence[SfxCue],
    *,
    force: bool = False,
) -> SfxResult:
    """Mix sound effects onto an audio or video timeline at specified timestamps.

    Args:
        source: Base audio or video file.
        output: Destination media path.
        config: Project configuration.
        cues: Sequence of SfxCue objects (kind/path, at_s, volume).
        force: Allow overwrite.
    """
    if not cues:
        raise ValidationError("at least one SfxCue must be specified")

    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)
    source_info = probe(resolved_source, config)

    # Prepare SFX files (either generated procedural SFX or external files)
    sfx_paths: list[Path] = []
    for i, cue in enumerate(cues):
        check_range(cue.at_s, 0.0, max(0.1, source_info.duration_s), f"cue[{i}].at_s")
        check_range(cue.volume, 0.0, 3.0, f"cue[{i}].volume")

        if isinstance(cue.kind, str) and cue.kind in SFX_KINDS:
            sfx_file = work_path(config, f"sfx_{cue.kind}_{i}", ".wav")
            generate_procedural_sfx(cue.kind, sfx_file, config, force=True)
            sfx_paths.append(sfx_file)
        else:
            sfx_paths.append(ensure_readable_source(cue.kind))

    # Build multi-input ffmpeg filtergraph
    # Input 0: base video/audio
    # Inputs 1..N: SFX files
    cmd = ["-i", str(resolved_source)]
    for sp in sfx_paths:
        cmd.extend(["-i", str(sp)])

    filter_chains: list[str] = []
    mix_inputs: list[str] = []

    # 1. Base audio stream
    if source_info.has_audio:
        mix_inputs.append("[0:a]")
    else:
        # Generate silence base track if source had no audio
        filter_chains.append(f"anullsrc=r=44100:cl=stereo:d={source_info.duration_s}[base_silence]")
        mix_inputs.append("[base_silence]")

    # 2. Delayed & volume-adjusted SFX streams
    for i, (cue, _path) in enumerate(zip(cues, sfx_paths, strict=True)):
        input_idx = i + 1
        delay_ms = max(0, int(cue.at_s * 1000))
        label = f"[sfx_{i}]"
        filter_chains.append(
            f"[{input_idx}:a]adelay={delay_ms}|{delay_ms},volume={cue.volume:.2f}{label}"
        )
        mix_inputs.append(label)

    # 3. Mix all streams together
    mix_chain = (
        f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:"
        "duration=first:dropout_transition=0[out_a]"
    )
    filter_chains.append(mix_chain)
    filtergraph = ";".join(filter_chains)

    if source_info.has_video:
        cmd.extend(
            [
                "-filter_complex",
                filtergraph,
                "-map",
                "0:v",
                "-map",
                "[out_a]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
            ]
        )
    else:
        cmd.extend(
            [
                "-filter_complex",
                filtergraph,
                "-map",
                "[out_a]",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
            ]
        )

    cmd.append(str(resolved_output))
    run_ffmpeg(cmd, config)

    final_info = verify_render(
        resolved_output,
        config,
        Expectations(requires_audio=True, requires_video=source_info.has_video),
    )

    return SfxResult(
        output=resolved_output,
        media=final_info,
        cues_count=len(cues),
    )
