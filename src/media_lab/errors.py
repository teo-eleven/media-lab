"""Typed errors for media-lab.

Every failure path in this package raises one of these, so callers never have
to parse a message string to know what went wrong.
"""

from __future__ import annotations


class MediaLabError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(MediaLabError):
    """Configuration is missing, malformed, or points at something absent."""


class ValidationError(MediaLabError):
    """A caller-supplied argument is outside the range this project accepts."""


class PathSafetyError(MediaLabError):
    """A path would violate the non-destructive contract of the project."""


class KinoError(MediaLabError):
    """A `kino` invocation failed."""

    def __init__(
        self, args: tuple[str, ...], returncode: int, stderr: str, stdout: str = ""
    ) -> None:
        self.args_used = args
        self.returncode = returncode
        self.stderr = stderr.strip()
        self.stdout = stdout.strip()
        command = " ".join(args)
        detail = self.stderr or self.stdout or "(no output)"
        super().__init__(f"kino command failed (exit {returncode}): {command}\n{detail}")


class KinoTimeoutError(MediaLabError):
    """A `kino` invocation exceeded its allotted time."""

    def __init__(self, args: tuple[str, ...], timeout_s: int) -> None:
        self.args_used = args
        self.timeout_s = timeout_s
        super().__init__(f"kino command timed out after {timeout_s}s: {' '.join(args)}")


class FFmpegError(MediaLabError):
    """A direct ffmpeg invocation failed."""

    def __init__(self, args: tuple[str, ...], returncode: int, stderr: str) -> None:
        self.args_used = args
        self.returncode = returncode
        self.stderr = stderr.strip()
        tail = "\n".join(self.stderr.splitlines()[-5:]) or "(no stderr)"
        super().__init__(f"ffmpeg failed (exit {returncode}):\n{tail}")


class ProbeError(MediaLabError):
    """ffprobe could not read a file, or returned something unusable."""


class MlEnvError(MediaLabError):
    """The ML environment (the RVM checkout or its weights) is missing or incomplete."""


class VerificationError(MediaLabError):
    """A render completed but does not match what was asked for."""

    def __init__(self, path: str, problems: tuple[str, ...]) -> None:
        self.path = path
        self.problems = problems
        listed = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"render did not match expectations: {path}\n{listed}")
