"""Config is validated at startup, loudly."""

from __future__ import annotations

from pathlib import Path

import pytest

from media_lab.config import load_config, read_env_file, require_ml
from media_lab.errors import ConfigError, MlEnvError

REPO_BIN = Path(__file__).resolve().parent.parent / "bin"


def _env(**overrides: str) -> dict[str, str]:
    base = {"MEDIA_LAB_FFMPEG_DIR": str(REPO_BIN)}
    base.update(overrides)
    return base


def test_loads_valid_configuration(tmp_path: Path, bin_dir: Path) -> None:
    (tmp_path / "in").mkdir()
    config = load_config(root=tmp_path, env=_env())

    assert config.ffmpeg_dir == bin_dir
    assert config.out_dir.is_dir()
    assert config.work_dir.is_dir()
    assert config.kino_timeout_s == 1800


def test_rejects_missing_ffmpeg(tmp_path: Path) -> None:
    (tmp_path / "in").mkdir()
    empty = tmp_path / "nowhere"
    empty.mkdir()

    with pytest.raises(ConfigError, match="fetch-ffmpeg"):
        load_config(root=tmp_path, env=_env(MEDIA_LAB_FFMPEG_DIR=str(empty)))


def test_rejects_missing_source_directory(tmp_path: Path, bin_dir: Path) -> None:
    with pytest.raises(ConfigError, match="source directory does not exist"):
        load_config(root=tmp_path, env=_env())


def test_rejects_non_numeric_timeout(tmp_path: Path, bin_dir: Path) -> None:
    (tmp_path / "in").mkdir()
    with pytest.raises(ConfigError, match="whole number"):
        load_config(root=tmp_path, env=_env(MEDIA_LAB_KINO_TIMEOUT_S="soon"))


def test_rejects_zero_timeout(tmp_path: Path, bin_dir: Path) -> None:
    (tmp_path / "in").mkdir()
    with pytest.raises(ConfigError, match="at least 1"):
        load_config(root=tmp_path, env=_env(MEDIA_LAB_KINO_TIMEOUT_S="0"))


def test_reads_env_file_and_ignores_comments(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "# a comment\n\nMEDIA_LAB_IN_DIR='./sources'\nBROKEN\nX=1\n", encoding="utf-8"
    )
    values = read_env_file(tmp_path / ".env")

    assert values == {"MEDIA_LAB_IN_DIR": "./sources", "X": "1"}


def test_missing_env_file_is_not_an_error(tmp_path: Path) -> None:
    assert read_env_file(tmp_path / ".env") == {}


def test_config_is_immutable(tmp_path: Path, bin_dir: Path) -> None:
    (tmp_path / "in").mkdir()
    config = load_config(root=tmp_path, env=_env())

    with pytest.raises(AttributeError):
        config.kino_timeout_s = 5  # type: ignore[misc]


def test_rejects_nested_media_directories(tmp_path: Path, bin_dir: Path) -> None:
    """Overlapping directories must fail at startup, not on the first write."""
    (tmp_path / "in").mkdir()
    with pytest.raises(ConfigError, match="nested"):
        load_config(root=tmp_path, env=_env(MEDIA_LAB_OUT_DIR="./in/out"))


def test_rejects_identical_media_directories(tmp_path: Path, bin_dir: Path) -> None:
    (tmp_path / "in").mkdir()
    with pytest.raises(ConfigError, match="same path"):
        load_config(root=tmp_path, env=_env(MEDIA_LAB_OUT_DIR="./in"))


# --- ML paths (RVM checkout + weights), validated lazily via require_ml -------


def _stub_rvm(root: Path, *, models: tuple[str, ...] = ("resnet50", "mobilenetv3")) -> None:
    """Create the minimal RVM checkout + weight files require_ml looks for."""
    (root / "tools" / "RobustVideoMatting" / "model").mkdir(parents=True)
    (root / "tools" / "RobustVideoMatting" / "model" / "__init__.py").write_text(
        "from .model import MattingNetwork\n", encoding="utf-8"
    )
    weights = root / "work" / "punto-edit" / "gen" / "weights"
    weights.mkdir(parents=True)
    for model in models:
        (weights / f"rvm_{model}.pth").write_bytes(b"stub")


def test_ml_paths_default_relative_to_root(tmp_path: Path, bin_dir: Path) -> None:
    (tmp_path / "in").mkdir()
    config = load_config(root=tmp_path, env=_env())

    assert config.rvm_repo == tmp_path / "tools" / "RobustVideoMatting"
    assert config.weights_dir == tmp_path / "work" / "punto-edit" / "gen" / "weights"


def test_ml_paths_read_from_env(tmp_path: Path, bin_dir: Path) -> None:
    (tmp_path / "in").mkdir()
    config = load_config(
        root=tmp_path,
        env=_env(MEDIA_LAB_RVM_REPO="./vendor/rvm", MEDIA_LAB_WEIGHTS_DIR="./w"),
    )

    assert config.rvm_repo == tmp_path / "vendor" / "rvm"
    assert config.weights_dir == tmp_path / "w"


def test_load_config_does_not_touch_the_ml_environment(tmp_path: Path, bin_dir: Path) -> None:
    """load_config must succeed with no RVM checkout - only `matte` needs it."""
    (tmp_path / "in").mkdir()
    config = load_config(root=tmp_path, env=_env())

    assert not config.rvm_repo.exists()
    assert config.rvm_ready is False


def test_require_ml_reports_a_missing_checkout(tmp_path: Path, bin_dir: Path) -> None:
    (tmp_path / "in").mkdir()
    config = load_config(root=tmp_path, env=_env())

    with pytest.raises(MlEnvError, match=r"(?s)RVM checkout not found.*fetch-rvm"):
        require_ml(config)


def test_require_ml_reports_a_missing_weight_file(tmp_path: Path, bin_dir: Path) -> None:
    (tmp_path / "in").mkdir()
    _stub_rvm(tmp_path, models=("mobilenetv3",))
    config = load_config(root=tmp_path, env=_env())

    with pytest.raises(MlEnvError, match=r"(?s)missing weight file.*rvm_resnet50\.pth"):
        require_ml(config, model="resnet50")


def test_require_ml_rejects_an_unknown_model(tmp_path: Path, bin_dir: Path) -> None:
    (tmp_path / "in").mkdir()
    _stub_rvm(tmp_path)
    config = load_config(root=tmp_path, env=_env())

    with pytest.raises(MlEnvError, match="unknown RVM model"):
        require_ml(config, model="birefnet")


def test_require_ml_passes_when_checkout_and_weights_are_present(
    tmp_path: Path, bin_dir: Path
) -> None:
    (tmp_path / "in").mkdir()
    _stub_rvm(tmp_path)
    config = load_config(root=tmp_path, env=_env())

    require_ml(config, model="resnet50")  # no raise
    assert config.rvm_ready is True
