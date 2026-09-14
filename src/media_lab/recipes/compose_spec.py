"""Render a shot from a compose-spec YAML.

Loads and validates the spec (`compose_spec.load_spec`), resolves every input
path, checks the subject frames are already placed on the full canvas, writes
the filtergraph to an inspectable sidecar, runs the single ffmpeg pass, and
verifies the result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ..compose_spec import build_filtergraph, load_spec
from ..config import Config
from ..errors import SpecError
from ..ffmpeg import run_filtergraph
from ..paths import ensure_readable_source, ensure_writable_output
from ..probe import MediaInfo, probe
from ..verify import Expectations, verify_render


@dataclass(frozen=True, slots=True)
class ComposeResult:
    media: MediaInfo
    spec_path: Path
    filtergraph_path: Path


def _resolve(config: Config, raw: str) -> Path:
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (config.root / candidate)


def _first_frame(pattern: str) -> str:
    """`f-%04d.png` -> `f-0001.png`. Raises SpecError on an unusable pattern."""
    try:
        return pattern % 1
    except (TypeError, ValueError) as exc:
        raise SpecError(f"subject.frames pattern is not usable: {pattern!r} ({exc})") from exc


def compose(
    spec_path: Path | str,
    output: Path | str,
    config: Config,
    *,
    force: bool = False,
) -> ComposeResult:
    """Render `spec_path` to `output`. Raises SpecError / PathSafetyError /
    FFmpegError / VerificationError."""
    resolved_spec = ensure_readable_source(spec_path)
    spec = load_spec(resolved_spec)
    resolved_output = ensure_writable_output(output, config, force=force)

    background = ensure_readable_source(_resolve(config, spec.background.path))
    first_frame = ensure_readable_source(_resolve(config, _first_frame(spec.subject.frames)))

    frame_info = probe(first_frame, config)
    canvas = (spec.background.width, spec.background.height)
    if (frame_info.width, frame_info.height) != canvas:
        raise SpecError(
            f"subject frame is {frame_info.width}x{frame_info.height}, canvas is "
            f"{canvas[0]}x{canvas[1]}; compose-spec overlays pre-placed full-canvas frames"
        )

    graph = build_filtergraph(spec)
    # build_filtergraph keeps the spec's (relative) paths; ffmpeg runs with an
    # unknown cwd, so swap in the resolved absolute ones.
    absolute_inputs = (
        replace(graph.inputs[0], path=str(background)),
        replace(graph.inputs[1], path=str(_resolve(config, spec.subject.frames))),
    )

    filtergraph_path = ensure_writable_output(
        resolved_output.parent / f"{resolved_output.stem}.filtergraph.txt", config, force=force
    )
    filtergraph_path.write_text(graph.filter_complex + "\n", encoding="utf-8")

    run_filtergraph(
        [(inp.lead_args, inp.path) for inp in absolute_inputs],
        graph.filter_complex,
        graph.map_target,
        graph.encode_args,
        resolved_output,
        config,
        frames=graph.frames,
    )

    media = verify_render(
        resolved_output,
        config,
        Expectations(
            width=spec.background.width,
            height=spec.background.height,
            duration_s=spec.background.frames / spec.background.fps,
            requires_audio=False,
        ),
    )
    return ComposeResult(media=media, spec_path=resolved_spec, filtergraph_path=filtergraph_path)
