"""Declarative end-to-end editing recipe for video and photo assets.

Takes a high-level EditSpec and orchestrates the complete chain of native recipes:
speech cleaning -> visual looks -> reframing -> subtitles -> music bed mixing.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config
from ..edit_spec import EditSpec, parse_edit_spec
from ..ffmpeg import run_ffmpeg
from ..kino import KinoRunner
from ..ml_runner import MlRunner
from ..paths import ensure_readable_source, ensure_writable_output, work_directory
from ..probe import MediaInfo, probe
from ..recipes.audio_bed import add_music_bed
from ..recipes.filters import apply_look, apply_look_chain
from ..recipes.photo import IMAGE_EXTENSIONS, edit_photo
from ..recipes.stems import separate_stems
from ..recipes.subtitles import generate_subtitles
from ..recipes.to_short import to_short
from ..verify import Expectations, verify_render


@dataclass(frozen=True, slots=True)
class EditResult:
    """Outcome of declarative edit execution."""

    output: Path
    media: MediaInfo
    steps_executed: tuple[str, ...]


def run_edit_spec(
    spec: EditSpec | str | Path | dict[str, Any],
    config: Config,
    runner: KinoRunner,
    ml_runner: MlRunner,
    *,
    force: bool = False,
) -> EditResult:
    """Execute a declarative edit specification.

    Args:
        spec: EditSpec instance, path to YAML file, or raw mapping.
        config: Configuration object.
        runner: Kino CLI runner.
        ml_runner: ML subprocess runner.
        force: Allow overwriting existing output.
    """
    if not isinstance(spec, EditSpec):
        spec = parse_edit_spec(spec)

    resolved_source = ensure_readable_source(spec.source)
    resolved_output = ensure_writable_output(spec.output, config, force=force)

    is_image = resolved_source.suffix.lower() in IMAGE_EXTENSIONS

    if is_image:
        photo_res = edit_photo(
            resolved_source,
            resolved_output,
            config,
            runner,
            aspect=spec.video.aspect,
            bg=spec.video.backdrop,
            cutout=spec.video.cutout,
            look=spec.video.look,
            force=force,
        )
        media_info = probe(photo_res.output_path, config)
        return EditResult(
            output=photo_res.output_path,
            media=media_info,
            steps_executed=("photo_edit",),
        )

    # Video workflow orchestration
    work = work_directory(config, "edit_orchestrator")
    current_clip = resolved_source
    steps_executed: list[str] = []

    # 1. Clean speech / audio stems
    if spec.audio.clean_speech:
        stems_out_dir = work / "stems"
        stems_video = work / "step1_clean_speech.mp4"
        separate_stems(
            current_clip,
            stems_out_dir,
            config,
            ml_runner,
            clean_speech=True,
            output_video=stems_video,
            force=True,
        )
        current_clip = stems_video
        steps_executed.append("clean_speech")

    # 2. Visual looks & color grading
    if spec.video.look:
        look_video = work / "step2_graded.mp4"
        if spec.video.second_look:
            apply_look_chain(
                current_clip,
                look_video,
                spec.video.look,
                spec.video.second_look,
                config,
                runner,
                force=True,
            )
        else:
            apply_look(current_clip, look_video, spec.video.look, config, runner, force=True)
        current_clip = look_video
        steps_executed.append(f"look_{spec.video.look}")

    # 3. Reframing / vertical social format
    if spec.video.aspect != "original":
        reframe_video = work / "step3_reframed.mp4"
        to_short(
            current_clip,
            reframe_video,
            config,
            runner,
            aspect_ratio=spec.video.aspect,
            thumbnail=False,
            force=True,
        )
        current_clip = reframe_video
        steps_executed.append(f"reframe_{spec.video.aspect}")

    # 4. Whisper subtitles
    if spec.video.subtitles.enabled:
        subs_video = work / "step4_subtitled.mp4"
        generate_subtitles(
            current_clip,
            subs_video,
            config,
            ml_runner,
            style=spec.video.subtitles.style,
            language=spec.video.subtitles.language,
            burn=True,
            force=True,
        )
        current_clip = subs_video
        steps_executed.append(f"subtitles_{spec.video.subtitles.style}")

    # 5. Background music mixing with ducking
    if spec.audio.music_track:
        add_music_bed(
            current_clip,
            spec.audio.music_track,
            resolved_output,
            config,
            target_lufs=spec.audio.target_lufs,
            music_volume=spec.audio.music_volume,
            force=force,
        )
        steps_executed.append("music_bed")
    else:
        # Copy or remux to final destination
        if current_clip.suffix.lower() == resolved_output.suffix.lower():
            shutil.copy2(current_clip, resolved_output)
        else:
            run_ffmpeg(
                ["-i", str(current_clip), "-c", "copy", str(resolved_output)],
                config,
            )
        steps_executed.append("finalize")

    final_info = verify_render(
        resolved_output,
        config,
        Expectations(duration_s=probe(resolved_source, config).duration_s),
    )

    return EditResult(
        output=resolved_output,
        media=final_info,
        steps_executed=tuple(steps_executed),
    )
