"""Turn a YAML shot spec into the ffmpeg filtergraph that renders it.

Pure: this module parses and validates a spec and builds strings. It never
touches the filesystem or ffmpeg - `recipes/compose_spec.py` does that.

The graph reproduces the v23 `docs/video-agent/pipeline/compose_pipeline.sh`
in one pass (verified by spike S1): background prep -> overlay the pre-placed,
full-canvas subject frames at 0:0 -> re-overlay a feathered bottom strip of
the background on top (depth occlusion) -> a fixed "v23" grade -> encode.
The subject is never moved or scaled here; that stays upstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import SpecError

GRADE_PROFILES = ("v23", "none")
MAX_CRF = 51
# `vignette` lands verbatim in the ffmpeg -filter_complex, so keep it to a plain
# angle expression - digits, PI, and the arithmetic operators. No commas, no
# filter names, nothing that could open a second filter or a `movie=` source.
_VIGNETTE_RE = re.compile(r"^[0-9PI.\s*/+()-]+$")
# The v23 grade chain, constants lifted verbatim from compose_pipeline.sh:27.
# Only atmosphere opacity, grain and vignette are exposed as spec knobs.
_V23_ATMOSPHERE = "scale=iw/6:ih/6,gblur=sigma=7,scale={w}:{h}:flags=bilinear,eq=brightness=0.03"
_V23_COLOR = (
    "eq=contrast=1.04:saturation=1.07:brightness=0.008:gamma=0.99,"
    "colorbalance=rs=0.008:bs=-0.012:rm=0.008:bm=-0.008,"
    "curves=master='0/0.01 0.5/0.5 1/0.993',unsharp=5:5:0.24"
)


@dataclass(frozen=True, slots=True)
class BackgroundSpec:
    path: str
    width: int
    height: int
    fps: int
    frames: int
    start_s: float = 0.0


@dataclass(frozen=True, slots=True)
class SubjectSpec:
    frames: str  # ffmpeg image2 pattern, e.g. work/.../placed/f-%04d.png
    fps: int


@dataclass(frozen=True, slots=True)
class OcclusionSpec:
    height: int
    feather: int
    y: int


@dataclass(frozen=True, slots=True)
class GradeSpec:
    profile: str = "v23"
    atmosphere: float = 0.07
    grain: int = 4
    vignette: str = "PI/5.8"


@dataclass(frozen=True, slots=True)
class OutputSpec:
    crf: int = 18
    preset: str = "medium"


@dataclass(frozen=True, slots=True)
class ComposeSpec:
    background: BackgroundSpec
    subject: SubjectSpec
    grade: GradeSpec = field(default_factory=GradeSpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    occlusion: OcclusionSpec | None = None


@dataclass(frozen=True, slots=True)
class FfmpegInput:
    """One ffmpeg `-i`, with the flags that must precede it."""

    path: str
    lead_args: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FilterGraph:
    """Everything `recipes/compose_spec.py` needs for a single ffmpeg run."""

    inputs: tuple[FfmpegInput, ...]
    filter_complex: str
    map_target: str
    frames: int
    encode_args: tuple[str, ...]


# --- parsing -----------------------------------------------------------------


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise SpecError(f"{where}: missing '{key}'")
    return mapping[key]


def _as_section(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SpecError(f"{where}: expected a positive integer, got {value!r}")
    return value


def _non_negative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpecError(f"{where}: expected a non-negative integer, got {value!r}")
    return value


def load_spec(path: Path | str) -> ComposeSpec:
    """Parse and validate a compose-spec YAML file. Raises SpecError."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"{path}: not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise SpecError(f"{path}: top level must be a mapping")

    bg = _as_section(_require(raw, "background", "background"), "background")
    bg_path = _require(bg, "path", "background")
    if not isinstance(bg_path, str) or not bg_path.strip():
        raise SpecError("background.path: must be a non-empty string")
    background = BackgroundSpec(
        path=bg_path,
        width=_positive_int(_require(bg, "width", "background"), "background.width"),
        height=_positive_int(_require(bg, "height", "background"), "background.height"),
        fps=_positive_int(_require(bg, "fps", "background"), "background.fps"),
        frames=_positive_int(_require(bg, "frames", "background"), "background.frames"),
        start_s=float(bg.get("start_s", 0.0)),
    )

    subj = _as_section(_require(raw, "subject", "subject"), "subject")
    subj_frames = _require(subj, "frames", "subject")
    if not isinstance(subj_frames, str) or "%" not in subj_frames:
        raise SpecError("subject.frames: must be an ffmpeg pattern like 'f-%04d.png'")
    subject = SubjectSpec(
        frames=subj_frames,
        fps=_positive_int(subj.get("fps", background.fps), "subject.fps"),
    )

    occlusion = None
    if "occlusion" in raw and raw["occlusion"] is not None:
        occ = _as_section(raw["occlusion"], "occlusion")
        occlusion = OcclusionSpec(
            height=_positive_int(_require(occ, "height", "occlusion"), "occlusion.height"),
            feather=_non_negative_int(_require(occ, "feather", "occlusion"), "occlusion.feather"),
            y=_non_negative_int(_require(occ, "y", "occlusion"), "occlusion.y"),
        )
        if occlusion.feather > occlusion.height:
            raise SpecError("occlusion.feather must not exceed occlusion.height")
        if occlusion.y + occlusion.height > background.height:
            raise SpecError("occlusion strip runs past the bottom of the canvas")

    grade = GradeSpec()
    if "grade" in raw and raw["grade"] is not None:
        g = _as_section(raw["grade"], "grade")
        profile = g.get("profile", "v23")
        if profile not in GRADE_PROFILES:
            raise SpecError(
                f"grade.profile: must be one of {list(GRADE_PROFILES)}, got {profile!r}"
            )
        atmosphere = float(g.get("atmosphere", 0.07))
        if not 0.0 <= atmosphere <= 1.0:
            raise SpecError(f"grade.atmosphere: must be within [0, 1], got {atmosphere}")
        vignette = str(g.get("vignette", "PI/5.8"))
        if not _VIGNETTE_RE.match(vignette):
            raise SpecError(
                f"grade.vignette: only an angle expression is allowed "
                f"(digits, PI, + - * / . parens), got {vignette!r}"
            )
        grade = GradeSpec(
            profile=profile,
            atmosphere=atmosphere,
            grain=_non_negative_int(g.get("grain", 4), "grade.grain"),
            vignette=vignette,
        )

    output = OutputSpec()
    if "output" in raw and raw["output"] is not None:
        o = _as_section(raw["output"], "output")
        crf = _non_negative_int(o.get("crf", 18), "output.crf")
        if crf > MAX_CRF:
            raise SpecError(f"output.crf: must be within [0, {MAX_CRF}], got {crf}")
        output = OutputSpec(crf=crf, preset=str(o.get("preset", "medium")))

    return ComposeSpec(
        background=background,
        subject=subject,
        grade=grade,
        output=output,
        occlusion=occlusion,
    )


# --- building --------------------------------------------------------------


def _occlusion_branch(spec: ComposeSpec) -> tuple[str, str, str]:
    """(strip-producing chain, subject-overlay chain, strip-overlay chain)."""
    occ = spec.occlusion
    if occ is None:
        return "", "[bgA][1:v]overlay=0:0:format=auto[out_pre]", ""
    w = spec.background.width
    ramp = f"if(lt(Y,{occ.feather}),255*Y/{occ.feather},255)" if occ.feather > 0 else "255"
    strip = (
        f"[bgB]crop={w}:{occ.height}:0:{occ.y},format=yuva444p,"
        f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='{ramp}'[fg];"
    )
    subject_overlay = "[bgA][1:v]overlay=0:0:format=auto[mid];"
    strip_overlay = f"[mid][fg]overlay=0:{occ.y}:format=auto[out_pre]"
    return strip, subject_overlay, strip_overlay


def _grade_chain(spec: ComposeSpec) -> str:
    """From [out_pre] to [out]."""
    if spec.grade.profile == "none":
        return "[out_pre]format=yuv420p[out]"
    w, h = spec.background.width, spec.background.height
    atmo = _V23_ATMOSPHERE.format(w=w, h=h)
    return (
        "[out_pre]format=gbrp,split[base][atm];"
        f"[atm]{atmo}[atmb];"
        f"[base][atmb]blend=all_mode=screen:all_opacity={spec.grade.atmosphere:g}[lit];"
        f"[lit]{_V23_COLOR},vignette={spec.grade.vignette},"
        f"noise=alls={spec.grade.grain}:allf=t,format=yuv420p[out]"
    )


def build_filtergraph(spec: ComposeSpec) -> FilterGraph:
    """Build the single-pass ffmpeg filtergraph for `spec`. No I/O."""
    bg, subj = spec.background, spec.subject
    strip, subject_overlay, strip_overlay = _occlusion_branch(spec)

    scale = f"[0:v]fps={bg.fps},scale={bg.width}:{bg.height}:flags=lanczos"
    prep = f"{scale},split[bgA][bgB];" if spec.occlusion is not None else f"{scale}[bgA];"

    filter_complex = prep + strip + subject_overlay + strip_overlay + ";" + _grade_chain(spec)

    return FilterGraph(
        inputs=(
            FfmpegInput(bg.path, ("-ss", f"{bg.start_s:g}")),
            FfmpegInput(subj.frames, ("-framerate", str(subj.fps))),
        ),
        filter_complex=filter_complex,
        map_target="[out]",
        frames=bg.frames,
        encode_args=(
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-preset",
            spec.output.preset,
            "-crf",
            str(spec.output.crf),
            "-movflags",
            "+faststart",
        ),
    )
