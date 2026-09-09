"""The single place that runs an ML task as a child process.

RVM (Robust Video Matting) is GPL-3.0 and `torch` is a heavy import; neither
belongs in `media_lab`'s own process. Matting runs as a child of the project
interpreter, driven through here, with data crossing as files. This mirrors
`kino.py` (kinocut) and `ffmpeg.py` (direct ffmpeg): one bounded, typed exit.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .errors import MlEnvError

# The Real-ESRGAN pass the punto runner drives through here is ~24 min for the
# 193-frame clip, so the default ceiling is generous.
DEFAULT_ML_TIMEOUT_S = 1800


@dataclass(frozen=True, slots=True)
class MlResult:
    """Outcome of one successful ML subprocess."""

    command: tuple[str, ...]
    stdout: str
    stderr: str
    duration_s: float


@dataclass(frozen=True, slots=True)
class MlRunner:
    """Runs a Python driver script as a child of the project interpreter."""

    config: Config
    python: Path

    @classmethod
    def from_config(cls, config: Config) -> MlRunner:
        return cls(config=config, python=Path(sys.executable))

    def _environment(self) -> dict[str, str]:
        """Inherit the parent environment, with the project ffmpeg on PATH."""
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([str(self.config.ffmpeg_dir), env.get("PATH", "")])
        return env

    def run(
        self,
        script: Path | str,
        args: Sequence[str] = (),
        *,
        timeout_s: int = DEFAULT_ML_TIMEOUT_S,
    ) -> MlResult:
        """Run `script args...` with the project interpreter.

        Raises MlEnvError if the script is missing, exits non-zero, or times out.
        """
        script_path = Path(script)
        if not script_path.is_file():
            raise MlEnvError(f"ML driver script not found: {script_path}")

        command = [str(self.python), str(script_path), *(str(a) for a in args)]
        started = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603 - fixed interpreter, no shell
                command,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=self._environment(),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MlEnvError(
                f"ML subprocess timed out after {timeout_s}s: {script_path.name}"
            ) from exc

        if completed.returncode != 0:
            tail = "\n".join(completed.stderr.splitlines()[-5:]) or "(no stderr)"
            raise MlEnvError(
                f"ML subprocess failed (exit {completed.returncode}): {script_path.name}\n{tail}"
            )
        return MlResult(
            command=tuple(command),
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_s=time.monotonic() - started,
        )
