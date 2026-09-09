"""The ML subprocess boundary. Real child processes of tiny scripts - no mocks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from media_lab.config import Config
from media_lab.errors import MlEnvError
from media_lab.ml_runner import MlRunner


@pytest.fixture
def runner(config: Config) -> MlRunner:
    return MlRunner.from_config(config)


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "driver.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_from_config_uses_the_project_interpreter(config: Config) -> None:
    assert MlRunner.from_config(config).python == Path(sys.executable)


def test_runs_a_script_and_captures_stdout(runner: MlRunner, tmp_path: Path) -> None:
    script = _script(tmp_path, "print('matte done')")

    result = runner.run(script)

    assert result.stdout.strip() == "matte done"
    assert result.duration_s >= 0
    assert result.command[0] == sys.executable


def test_passes_arguments_through_to_the_script(runner: MlRunner, tmp_path: Path) -> None:
    script = _script(tmp_path, "import sys; print('|'.join(sys.argv[1:]))")

    result = runner.run(script, ["resnet50", "0.375", str(tmp_path)])

    assert result.stdout.strip() == f"resnet50|0.375|{tmp_path}"


def test_child_sees_the_project_ffmpeg_on_path(
    runner: MlRunner, tmp_path: Path, bin_dir: Path
) -> None:
    script = _script(tmp_path, "import os; print(os.environ['PATH'].split(os.pathsep)[0])")

    result = runner.run(script)

    assert result.stdout.strip() == str(bin_dir)


def test_missing_script_is_reported(runner: MlRunner, tmp_path: Path) -> None:
    with pytest.raises(MlEnvError, match="driver script not found"):
        runner.run(tmp_path / "absent.py")


def test_non_zero_exit_raises_with_a_stderr_tail(runner: MlRunner, tmp_path: Path) -> None:
    script = _script(
        tmp_path,
        "import sys; print('loading weights', file=sys.stderr); "
        "print('CUDA not available', file=sys.stderr); sys.exit(3)",
    )

    with pytest.raises(MlEnvError, match=r"(?s)exit 3.*CUDA not available"):
        runner.run(script)


def test_timeout_raises(runner: MlRunner, tmp_path: Path) -> None:
    script = _script(tmp_path, "import time; time.sleep(5)")

    with pytest.raises(MlEnvError, match="timed out after 1s"):
        runner.run(script, timeout_s=1)
