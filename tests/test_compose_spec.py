"""The pure compose-spec parser + filtergraph builder. No I/O beyond reading YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from media_lab.compose_spec import build_filtergraph, load_spec
from media_lab.errors import SpecError


def _raw(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "background": {
            "path": "in/backgrounds/nyc-wallst.mp4",
            "width": 2160,
            "height": 3840,
            "fps": 30,
            "frames": 193,
            "start_s": 1.0,
        },
        "subject": {"frames": "work/punto-edit/isnet/placed/f-%04d.png", "fps": 30},
        "occlusion": {"height": 130, "feather": 70, "y": 3710},
        "grade": {"profile": "v23", "atmosphere": 0.07, "grain": 4, "vignette": "PI/5.8"},
        "output": {"crf": 18, "preset": "medium"},
    }
    base.update(overrides)
    return base


def _write(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


# --- parsing ---------------------------------------------------------------


def test_loads_a_full_spec_from_yaml(tmp_path: Path) -> None:
    spec = load_spec(_write(tmp_path, _raw()))

    assert spec.background.width == 2160
    assert spec.background.start_s == 1.0
    assert spec.subject.frames.endswith("f-%04d.png")
    assert spec.occlusion is not None and spec.occlusion.y == 3710
    assert spec.grade.profile == "v23"
    assert spec.output.crf == 18


def test_optional_sections_fall_back_to_defaults(tmp_path: Path) -> None:
    raw = _raw()
    del raw["grade"]
    del raw["output"]
    del raw["occlusion"]
    spec = load_spec(_write(tmp_path, raw))

    assert spec.occlusion is None
    assert spec.grade.profile == "v23"
    assert spec.grade.atmosphere == 0.07
    assert spec.output.crf == 18
    assert spec.subject.fps == spec.background.fps  # inherited


def test_rejects_a_non_mapping_top_level(tmp_path: Path) -> None:
    path = tmp_path / "spec.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(SpecError, match="top level must be a mapping"):
        load_spec(path)


def test_rejects_a_missing_background_path(tmp_path: Path) -> None:
    raw = _raw()
    del raw["background"]["path"]
    with pytest.raises(SpecError, match="background: missing 'path'"):
        load_spec(_write(tmp_path, raw))


def test_rejects_a_non_positive_dimension(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="background.width: expected a positive integer"):
        load_spec(_write(tmp_path, _raw(background={**_raw()["background"], "width": 0})))


def test_rejects_an_unknown_grade_profile(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="grade.profile: must be one of"):
        load_spec(_write(tmp_path, _raw(grade={"profile": "teal-orange"})))


def test_rejects_atmosphere_out_of_range(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match=r"grade.atmosphere: must be within \[0, 1\]"):
        load_spec(_write(tmp_path, _raw(grade={"profile": "v23", "atmosphere": 1.5})))


def test_rejects_feather_larger_than_the_strip(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="feather must not exceed"):
        load_spec(_write(tmp_path, _raw(occlusion={"height": 100, "feather": 200, "y": 3000})))


def test_rejects_an_occlusion_strip_past_the_canvas(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="past the bottom of the canvas"):
        load_spec(_write(tmp_path, _raw(occlusion={"height": 200, "feather": 50, "y": 3700})))


def test_rejects_a_subject_without_a_frame_pattern(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="ffmpeg pattern"):
        load_spec(_write(tmp_path, _raw(subject={"frames": "placed/frame.png", "fps": 30})))


def test_rejects_a_crf_above_the_ceiling(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match=r"output.crf: must be within \[0, 51\]"):
        load_spec(_write(tmp_path, _raw(output={"crf": 80})))


def test_rejects_a_vignette_that_smuggles_a_filter(tmp_path: Path) -> None:
    raw = _raw(grade={"profile": "v23", "vignette": "PI/5,movie=/etc/passwd[x]"})
    with pytest.raises(SpecError, match="grade.vignette: only an angle expression"):
        load_spec(_write(tmp_path, raw))


# --- building ------------------------------------------------------------


def test_v23_filtergraph_has_every_stage(tmp_path: Path) -> None:
    graph = build_filtergraph(load_spec(_write(tmp_path, _raw())))

    fc = graph.filter_complex
    assert "scale=2160:3840:flags=lanczos,split[bgA][bgB]" in fc
    assert "crop=2160:130:0:3710" in fc
    assert "a='if(lt(Y,70),255*Y/70,255)'" in fc
    assert "[bgA][1:v]overlay=0:0:format=auto[mid]" in fc
    assert "[mid][fg]overlay=0:3710:format=auto[out_pre]" in fc
    assert "blend=all_mode=screen:all_opacity=0.07" in fc
    assert "noise=alls=4:allf=t" in fc
    assert fc.endswith("[out]")
    assert graph.map_target == "[out]"
    assert graph.frames == 193


def test_inputs_carry_their_lead_flags(tmp_path: Path) -> None:
    graph = build_filtergraph(load_spec(_write(tmp_path, _raw())))

    assert graph.inputs[0].lead_args == ("-ss", "1")
    assert graph.inputs[0].path == "in/backgrounds/nyc-wallst.mp4"
    assert graph.inputs[1].lead_args == ("-framerate", "30")


def test_encode_args_reflect_the_output_section(tmp_path: Path) -> None:
    raw = _raw(output={"crf": 20, "preset": "slow"})
    graph = build_filtergraph(load_spec(_write(tmp_path, raw)))

    assert "-crf" in graph.encode_args
    assert graph.encode_args[graph.encode_args.index("-crf") + 1] == "20"
    assert graph.encode_args[graph.encode_args.index("-preset") + 1] == "slow"


def test_grade_none_drops_the_whole_chain(tmp_path: Path) -> None:
    graph = build_filtergraph(load_spec(_write(tmp_path, _raw(grade={"profile": "none"}))))

    assert graph.filter_complex.endswith("[out_pre]format=yuv420p[out]")
    assert "blend=" not in graph.filter_complex
    assert "colorbalance" not in graph.filter_complex


def test_no_occlusion_is_a_plain_overlay(tmp_path: Path) -> None:
    raw = _raw()
    del raw["occlusion"]
    graph = build_filtergraph(load_spec(_write(tmp_path, raw)))

    fc = graph.filter_complex
    assert "split[bgA][bgB]" not in fc
    assert "crop=" not in fc
    assert "[bgA][1:v]overlay=0:0:format=auto[out_pre]" in fc


def test_zero_feather_gives_a_solid_strip(tmp_path: Path) -> None:
    graph = build_filtergraph(
        load_spec(_write(tmp_path, _raw(occlusion={"height": 130, "feather": 0, "y": 3710})))
    )
    assert "a='255'" in graph.filter_complex
