"""Configuration, read from the environment and validated at startup.

The project must fail loudly here rather than three stages into a render.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError, MlEnvError

DEFAULT_FFMPEG_DIR = "./bin"
DEFAULT_IN_DIR = "./in"
DEFAULT_OUT_DIR = "./out"
DEFAULT_WORK_DIR = "./work"
DEFAULT_KINO_TIMEOUT_S = 1800
MIN_KINO_TIMEOUT_S = 1
REQUIRED_BINARIES = ("ffmpeg", "ffprobe")
# RVM (Robust Video Matting) is a GPL-3 source checkout, never committed, and
# its weights are large binary files. Neither is validated at startup - only
DEFAULT_RVM_REPO = "./tools/RobustVideoMatting"
DEFAULT_WEIGHTS_DIR = "./work/punto-edit/gen/weights"
RVM_WEIGHT_FILES = {"resnet50": "rvm_resnet50.pth", "mobilenetv3": "rvm_mobilenetv3.pth"}
REALESRGAN_WEIGHT_FILES = {2: "RealESRGAN_x2plus.pth", 4: "RealESRGAN_x4plus.pth"}


@dataclass(frozen=True, slots=True)
class Config:
    """Resolved, validated settings. Immutable once built."""

    root: Path
    ffmpeg_dir: Path
    in_dir: Path
    out_dir: Path
    work_dir: Path
    kino_timeout_s: int
    hyperframes_command: Path | None
    rvm_repo: Path
    weights_dir: Path

    @property
    def ffmpeg(self) -> Path:
        return self.ffmpeg_dir / "ffmpeg"

    @property
    def ffprobe(self) -> Path:
        return self.ffmpeg_dir / "ffprobe"

    @property
    def rvm_ready(self) -> bool:
        """True when the RVM checkout looks importable and the weights dir exists."""
        return (self.rvm_repo / "model" / "__init__.py").is_file() and self.weights_dir.is_dir()


def read_env_file(path: Path) -> dict[str, str]:
    """Parse a KEY=VALUE .env file. Missing file yields an empty mapping."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _resolve_dir(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def _parse_timeout(raw: str) -> int:
    try:
        timeout = int(raw)
    except ValueError as exc:
        raise ConfigError(
            f"MEDIA_LAB_KINO_TIMEOUT_S must be a whole number of seconds, got {raw!r}"
        ) from exc
    if timeout < MIN_KINO_TIMEOUT_S:
        raise ConfigError(
            f"MEDIA_LAB_KINO_TIMEOUT_S must be at least {MIN_KINO_TIMEOUT_S}, got {timeout}"
        )
    return timeout


def _check_binaries(ffmpeg_dir: Path) -> None:
    missing = [
        name
        for name in REQUIRED_BINARIES
        if not (ffmpeg_dir / name).is_file() or not os.access(ffmpeg_dir / name, os.X_OK)
    ]
    if missing:
        raise ConfigError(
            f"missing or non-executable in {ffmpeg_dir}: {', '.join(missing)}. "
            "Run ./scripts/fetch-ffmpeg.sh to install them."
        )


def _check_directories_are_distinct(in_dir: Path, out_dir: Path, work_dir: Path) -> None:
    """Overlapping directories only fail much later, on the first write."""
    named = (
        ("MEDIA_LAB_IN_DIR", in_dir),
        ("MEDIA_LAB_OUT_DIR", out_dir),
        ("MEDIA_LAB_WORK_DIR", work_dir),
    )
    for first_name, first in named:
        for second_name, second in named:
            if first_name >= second_name:
                continue
            if first == second:
                raise ConfigError(f"{first_name} and {second_name} point at the same path: {first}")
            if first in second.parents or second in first.parents:
                raise ConfigError(
                    f"{first_name} ({first}) and {second_name} ({second}) are nested; "
                    "they must be separate directories"
                )


def load_config(
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Build a validated Config. Raises ConfigError on anything unusable."""
    project_root = (root or Path.cwd()).resolve()
    merged: dict[str, str] = {}
    merged.update(read_env_file(project_root / ".env"))
    merged.update(os.environ if env is None else env)

    ffmpeg_dir = _resolve_dir(project_root, merged.get("MEDIA_LAB_FFMPEG_DIR", DEFAULT_FFMPEG_DIR))
    _check_binaries(ffmpeg_dir)

    in_dir = _resolve_dir(project_root, merged.get("MEDIA_LAB_IN_DIR", DEFAULT_IN_DIR))
    if not in_dir.is_dir():
        raise ConfigError(f"source directory does not exist: {in_dir}")

    out_dir = _resolve_dir(project_root, merged.get("MEDIA_LAB_OUT_DIR", DEFAULT_OUT_DIR))
    work_dir = _resolve_dir(project_root, merged.get("MEDIA_LAB_WORK_DIR", DEFAULT_WORK_DIR))
    _check_directories_are_distinct(in_dir, out_dir, work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    raw_hyperframes = merged.get("MCP_VIDEO_HYPERFRAMES_COMMAND", "").strip()
    hyperframes = Path(raw_hyperframes) if raw_hyperframes else None
    if hyperframes is not None and not hyperframes.is_absolute():
        hyperframes = (project_root / hyperframes).resolve()

    # ML paths are resolved but NOT checked here - see require_ml().
    rvm_repo = _resolve_dir(project_root, merged.get("MEDIA_LAB_RVM_REPO", DEFAULT_RVM_REPO))
    weights_dir = _resolve_dir(
        project_root, merged.get("MEDIA_LAB_WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR)
    )

    return Config(
        root=project_root,
        ffmpeg_dir=ffmpeg_dir,
        in_dir=in_dir,
        out_dir=out_dir,
        work_dir=work_dir,
        kino_timeout_s=_parse_timeout(
            merged.get("MEDIA_LAB_KINO_TIMEOUT_S", str(DEFAULT_KINO_TIMEOUT_S))
        ),
        hyperframes_command=hyperframes,
        rvm_repo=rvm_repo,
        weights_dir=weights_dir,
    )


def require_ml(config: Config, *, model: str | None = None) -> None:
    """Assert the RVM checkout and weights are in place, or raise MlEnvError.

    Recipes (matte-video) call this; `load_config` deliberately does not, so
    `doctor` and every non-ML command work on a machine that has never run
    `scripts/fetch-rvm.sh`. Pass `model` to also require that backend's weight
    file.
    """
    problems: list[str] = []
    if not (config.rvm_repo / "model" / "__init__.py").is_file():
        problems.append(f"RVM checkout not found at {config.rvm_repo}")
    if not config.weights_dir.is_dir():
        problems.append(f"weights directory not found at {config.weights_dir}")
    elif model is not None:
        if model not in RVM_WEIGHT_FILES:
            allowed = sorted(RVM_WEIGHT_FILES)
            raise MlEnvError(f"unknown RVM model {model!r}; expected one of {allowed}")
        weight = config.weights_dir / RVM_WEIGHT_FILES[model]
        if not weight.is_file():
            problems.append(f"missing weight file {weight}")

    if problems:
        listed = "\n".join(f"  - {p}" for p in problems)
        raise MlEnvError(
            "RVM is not set up:\n"
            f"{listed}\n"
            "Run ./scripts/fetch-rvm.sh, then download the weights it prints."
        )


def require_realesrgan(config: Config, *, scale: int = 2) -> Path:
    """Assert Real-ESRGAN weight file exists, or raise MlEnvError.

    Returns the resolved path to the model weights file.
    """
    if scale not in REALESRGAN_WEIGHT_FILES:
        allowed = sorted(REALESRGAN_WEIGHT_FILES)
        raise MlEnvError(f"unsupported upscale scale {scale}; expected one of {allowed}")
    if not config.weights_dir.is_dir():
        raise MlEnvError(
            f"weights directory not found at {config.weights_dir}.\n"
            "Create it and download the Real-ESRGAN weights."
        )
    weight_file = config.weights_dir / REALESRGAN_WEIGHT_FILES[scale]
    if not weight_file.is_file():
        raise MlEnvError(
            f"Real-ESRGAN weight file not found at {weight_file}.\n"
            f"Expected {REALESRGAN_WEIGHT_FILES[scale]} under {config.weights_dir}."
        )
    return weight_file
