"""The command line surface."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from media_lab import cli


def test_doctor_reports_the_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], bin_dir: Path
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)

    assert cli.main(["doctor"]) == 0

    printed = capsys.readouterr().out
    assert "media-lab environment" in printed
    assert "kino doctor" in printed
    assert "rvm repo" in printed
    assert "weights" in printed
    # every ML line is marked configured or not - never left blank
    for line in printed.splitlines():
        if line.strip().startswith(("rvm repo", "weights")):
            assert "(ok)" in line or "(not configured)" in line


def test_reports_configuration_failures_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    assert cli.main(["doctor"]) == 1
    assert "error:" in capsys.readouterr().err


def test_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_parser_exposes_the_doctor_command() -> None:
    assert cli.build_parser().prog == "media-lab"


def test_every_recipe_has_a_handler() -> None:
    """A subcommand without a handler would fail only at runtime."""
    parser = cli.build_parser()
    actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    names = set(actions[0].choices)

    assert names == set(cli.HANDLERS) | {"doctor"}


def test_pipeline_command_accepts_the_full_option_set() -> None:
    args = cli.build_parser().parse_args(
        [
            "pipeline",
            "clip.mp4",
            "-o",
            "out.mp4",
            "--bg",
            "bg.png",
            "--look",
            "noir",
            "--track",
            "song.mp3",
            "--aspect",
            "9:16",
        ]
    )

    assert args.command == "pipeline"
    assert (args.bg, args.look, args.track) == ("bg.png", "noir", "song.mp3")


def test_filter_command_rejects_an_unknown_look() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["filter", "c.mp4", "-o", "o.mp4", "--look", "disco"])


def test_music_command_defaults_to_looping_the_bed() -> None:
    args = cli.build_parser().parse_args(["music", "c.mp4", "-o", "o.mp4", "--track", "s.mp3"])
    assert args.loop is True

    args = cli.build_parser().parse_args(
        ["music", "c.mp4", "-o", "o.mp4", "--track", "s.mp3", "--no-loop"]
    )
    assert args.loop is False


def test_cutout_reports_a_failure_as_a_message_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)

    assert cli.main(["cutout", "in/definitely-absent.mp4", "-o", "out/x.webm"]) == 1
    assert "does not exist" in capsys.readouterr().err


@pytest.fixture
def cli_project(tmp_path: Path, bin_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway project directory the CLI can be driven inside."""
    import subprocess

    (tmp_path / "in").mkdir()
    clip = tmp_path / "in" / "clip.mp4"
    subprocess.run(
        [
            str(bin_dir / "ffmpeg"),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=6:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(clip),
        ],
        check=True,
        capture_output=True,
    )
    monkeypatch.setenv("MEDIA_LAB_FFMPEG_DIR", str(bin_dir))
    monkeypatch.setenv(
        "MCP_VIDEO_HYPERFRAMES_COMMAND",
        str(Path(__file__).resolve().parent.parent / "node_modules" / ".bin" / "hyperframes"),
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_filter_handler_renders_and_reports(
    cli_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["filter", "in/clip.mp4", "-o", "out/graded.mp4", "--look", "noir"])

    assert exit_code == 0
    assert "look written to" in capsys.readouterr().out
    assert (cli_project / "out" / "graded.mp4").is_file()


def test_filter_handler_chains_two_looks(
    cli_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(
        ["filter", "in/clip.mp4", "-o", "out/g.mp4", "--look", "warm", "--then", "grain"]
    )

    assert exit_code == 0
    assert (cli_project / "out" / "g.mp4").is_file()


def test_short_handler_reports_the_quality_gate(
    cli_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["short", "in/clip.mp4", "-o", "out/vertical.mp4"])

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "short written to" in printed
    assert "quality score" in printed


def test_music_handler_reports_measured_loudness(
    cli_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import subprocess

    music = cli_project / "in" / "song.m4a"
    subprocess.run(
        [
            str(Path(__file__).resolve().parent.parent / "bin" / "ffmpeg"),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=2",
            "-c:a",
            "aac",
            str(music),
        ],
        check=True,
        capture_output=True,
    )

    exit_code = cli.main(["music", "in/clip.mp4", "-o", "out/mix.mp4", "--track", "in/song.m4a"])

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "measured loudness" in printed
    assert "ducking engaged: True" in printed


def test_handler_refuses_to_write_outside_the_project(
    cli_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["filter", "in/clip.mp4", "-o", "/tmp/escaped.mp4", "--look", "noir"])

    assert exit_code == 1
    assert "outside the project" in capsys.readouterr().err


def test_pipeline_accepts_the_quality_gate_flag() -> None:
    args = cli.build_parser().parse_args(["pipeline", "c.mp4", "-o", "o.mp4", "--fail-on-warning"])
    assert args.fail_on_warning is True


def test_clean_reports_an_already_empty_directory(
    cli_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["clean"]) == 0
    assert "already empty" in capsys.readouterr().out


def test_clean_removes_work_directory_contents(
    cli_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work = cli_project / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "leftover.mp4").write_bytes(b"scratch data")

    exit_code = cli.main(["clean"])

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "removed 1 item(s)" in printed
    assert "leftover.mp4" in printed
    assert list(work.iterdir()) == []


def test_clean_dry_run_lists_without_deleting(
    cli_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work = cli_project / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "leftover.mp4").write_bytes(b"scratch data")

    exit_code = cli.main(["clean", "--dry-run"])

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "would remove 1 item(s)" in printed
    assert (work / "leftover.mp4").is_file()


def test_clean_never_touches_in_or_out(
    cli_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = cli_project / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "final.mp4").write_bytes(b"keep me")
    work = cli_project / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "scratch.tmp").write_bytes(b"delete me")

    cli.main(["clean"])

    assert (out / "final.mp4").read_bytes() == b"keep me"
    assert (cli_project / "in" / "clip.mp4").is_file()
