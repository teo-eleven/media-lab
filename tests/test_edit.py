"""Tests for edit spec parser, recipe, and declarative CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from media_lab.cli import main
from media_lab.config import Config
from media_lab.edit_spec import EditSpec, parse_edit_spec
from media_lab.errors import ValidationError
from media_lab.kino import KinoRunner
from media_lab.ml_runner import MlRunner
from media_lab.recipes.edit import run_edit_spec


def _generate_synthetic_video(path: Path, config: Config, duration: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(config.ffmpeg),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size=320x240:rate=24",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )
    return path


def _generate_synthetic_image(path: Path, width: int = 400, height: int = 400) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (width, height), (120, 180, 240))
    im.save(path)
    return path


def test_parse_edit_spec_from_yaml_string() -> None:
    yaml_text = """
    source: in/video.mp4
    output: out/reel.mp4
    video:
      aspect: "9:16"
      look: "cinematic"
      subtitles:
        enabled: true
        style: "tiktok"
    audio:
      clean_speech: true
      target_lufs: -14.0
    """
    spec = parse_edit_spec(yaml_text)
    assert spec.source == "in/video.mp4"
    assert spec.output == "out/reel.mp4"
    assert spec.video.aspect == "9:16"
    assert spec.video.look == "cinematic"
    assert spec.video.subtitles.enabled is True
    assert spec.video.subtitles.style == "tiktok"
    assert spec.audio.clean_speech is True
    assert spec.audio.target_lufs == -14.0


def test_parse_edit_spec_validation_errors() -> None:
    with pytest.raises(ValidationError, match="requires a non-empty 'source'"):
        parse_edit_spec({"output": "out.mp4"})

    with pytest.raises(ValidationError, match="requires a non-empty 'output'"):
        parse_edit_spec({"source": "in.mp4"})

    with pytest.raises(ValidationError, match="unsupported aspect ratio"):
        parse_edit_spec({"source": "in.mp4", "output": "out.mp4", "video": {"aspect": "invalid"}})

    with pytest.raises(ValidationError, match="unsupported look"):
        parse_edit_spec({"source": "in.mp4", "output": "out.mp4", "video": {"look": "invalid"}})

    with pytest.raises(ValidationError, match="unsupported subtitle style"):
        parse_edit_spec(
            {"source": "in.mp4", "output": "out.mp4", "video": {"subtitles": {"style": "invalid"}}}
        )


def test_edit_photo_workflow(config: Config) -> None:
    runner = KinoRunner.from_config(config)
    ml_runner = MlRunner.from_config(config)
    img = _generate_synthetic_image(config.in_dir / "edit_src.jpg", 500, 300)
    out = config.out_dir / "edited_photo.jpg"

    spec = EditSpec(
        source=str(img),
        output=str(out),
    )
    res = run_edit_spec(spec, config, runner, ml_runner)
    assert res.output.is_file()
    assert "photo_edit" in res.steps_executed


def test_edit_video_workflow(config: Config) -> None:
    runner = KinoRunner.from_config(config)
    ml_runner = MlRunner.from_config(config)
    video = _generate_synthetic_video(config.in_dir / "edit_video.mp4", config)
    out = config.out_dir / "edited_video.mp4"

    spec = EditSpec(
        source=str(video),
        output=str(out),
    )
    res = run_edit_spec(spec, config, runner, ml_runner)
    assert res.output.is_file()
    assert "finalize" in res.steps_executed


def test_edit_video_multi_step_workflow(config: Config) -> None:
    runner = KinoRunner.from_config(config)
    ml_runner = MlRunner.from_config(config)
    video = _generate_synthetic_video(config.in_dir / "edit_multi.mp4", config)
    music = _generate_synthetic_video(config.in_dir / "music.mp4", config)
    out = config.out_dir / "edited_multi.mp4"

    config.work_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = config.work_dir / "spec.yaml"
    yaml_path.write_text(
        f"""
        source: {video}
        output: {out}
        video:
          aspect: "9:16"
          look: "warm"
          second_look: "cool"
        audio:
          music_track: {music}
          music_volume: 0.8
        """
    )

    res = run_edit_spec(yaml_path, config, runner, ml_runner)
    assert res.output.is_file()
    assert "look_warm" in res.steps_executed
    assert "reframe_9:16" in res.steps_executed
    assert "music_bed" in res.steps_executed


def test_edit_video_with_stems_and_subs(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = KinoRunner.from_config(config)
    ml_runner = MlRunner.from_config(config)
    video = _generate_synthetic_video(config.in_dir / "edit_stems_subs.mp4", config)
    out = config.out_dir / "edited_stems_subs.mp4"

    from media_lab.recipes.stems import StemsResult
    from media_lab.recipes.subtitles import SubtitlesResult

    # Mock stems and subtitles to avoid invoking real ML neural nets
    def fake_stems(
        source: Path | str,
        out_dir: Path | str,
        cfg: Config,
        ml_run: MlRunner,
        clean_speech: bool = False,
        output_video: Path | str | None = None,
        force: bool = False,
    ) -> StemsResult:
        import shutil

        src_path = Path(source)
        out_d = Path(out_dir)
        cleaned: Path | None = None
        if output_video:
            cleaned = Path(output_video)
            shutil.copy2(src_path, cleaned)
        return StemsResult(
            stems={"vocals": src_path, "no_vocals": src_path},
            output_dir=out_d,
            cleaned_video=cleaned,
        )

    def fake_subs(
        source: Path | str,
        out_path: Path | str,
        cfg: Config,
        ml_run: MlRunner,
        style: str = "tiktok",
        language: str | None = None,
        burn: bool = True,
        force: bool = False,
    ) -> SubtitlesResult:
        import shutil

        dest = Path(out_path)
        shutil.copy2(Path(source), dest)
        return SubtitlesResult(subtitle_file=dest, video_file=dest, segment_count=1, language="en")

    monkeypatch.setattr("media_lab.recipes.edit.separate_stems", fake_stems)
    monkeypatch.setattr("media_lab.recipes.edit.generate_subtitles", fake_subs)

    spec = parse_edit_spec(
        {
            "source": str(video),
            "output": str(out),
            "video": {
                "subtitles": {"enabled": True, "style": "tiktok"},
            },
            "audio": {
                "clean_speech": True,
            },
        }
    )
    res = run_edit_spec(spec, config, runner, ml_runner)
    assert res.output.is_file()
    assert "clean_speech" in res.steps_executed
    assert "subtitles_tiktok" in res.steps_executed


def test_cli_edit(
    config: Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("media_lab.cli.load_config", lambda: config)
    img = _generate_synthetic_image(config.in_dir / "cli_edit.jpg", 300, 300)
    out = config.out_dir / "cli_edited.jpg"

    rc = main(["edit", str(img), "-o", str(out), "--target", "1:1", "--look", "warm"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "edit finished:" in captured


def test_cli_edit_with_spec_file(
    config: Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("media_lab.cli.load_config", lambda: config)
    img = _generate_synthetic_image(config.in_dir / "cli_spec_img.jpg", 300, 300)
    out = config.out_dir / "cli_spec_out.jpg"

    config.work_dir.mkdir(parents=True, exist_ok=True)
    spec_file = config.work_dir / "my_spec.yaml"
    spec_file.write_text(
        f"""
        source: {img}
        output: {out}
        video:
          aspect: "4:5"
        """
    )

    rc = main(["edit", "--spec", str(spec_file)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "edit finished:" in captured
    assert out.is_file()
