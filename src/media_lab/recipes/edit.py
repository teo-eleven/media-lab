"""Declarative end-to-end editing recipe for video and photo assets.

Takes a high-level EditSpec and orchestrates the complete chain of native recipes:
silence trim -> vocal mastering -> visual looks -> smart reframe / short ->
retention punch-zoom -> B-roll cutaways -> typography badges -> subtitles ->
procedural sfx -> background music mixing with ducking.
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
from ..recipes.audio_enhance import enhance_audio
from ..recipes.broll import BrollCut, insert_broll
from ..recipes.face_retouch import retouch_portrait
from ..recipes.filters import apply_look, apply_look_chain
from ..recipes.inpainting import inpaint_image
from ..recipes.photo import IMAGE_EXTENSIONS, edit_photo
from ..recipes.progress_bar import add_progress_bar
from ..recipes.punch_zoom import punch_zoom
from ..recipes.sfx import SfxCue, add_sfx
from ..recipes.silence_trim import trim_silence
from ..recipes.smart_reframe import smart_reframe
from ..recipes.speed import change_speed
from ..recipes.stems import separate_stems
from ..recipes.subtitles import generate_subtitles
from ..recipes.to_short import to_short
from ..recipes.typography import TypographyStyle, apply_typography
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
        work = work_directory(config, "edit_photo_orchestrator")
        current_img = resolved_source
        photo_steps: list[str] = []

        # 1. Inpainting if requested
        if spec.photo.inpaint_bbox is not None:
            inpainted_path = work / "step1_inpainted.png"
            inpaint_image(
                current_img,
                inpainted_path,
                config,
                bbox=spec.photo.inpaint_bbox,
                force=True,
            )
            current_img = inpainted_path
            photo_steps.append("inpainting")

        # 2. Retouching / skin smoothing / depth blur if requested
        if spec.photo.retouch:
            retouched_path = work / "step2_retouched.png"
            retouch_portrait(
                current_img,
                retouched_path,
                config,
                smooth_skin=True,
                skin_strength=spec.photo.skin_strength,
                depth_blur=spec.photo.depth_blur,
                blur_sigma=spec.photo.bokeh_sigma,
                radiance=spec.photo.radiance,
                force=True,
            )
            current_img = retouched_path
            photo_steps.append("retouch")

        # 3. Base photo edit (framing, backdrop, looks)
        base_edit_path = work / "step3_base_edit.png"
        photo_res = edit_photo(
            current_img,
            base_edit_path,
            config,
            runner,
            aspect=spec.video.aspect,
            bg=spec.video.backdrop,
            cutout=spec.video.cutout,
            look=spec.video.look,
            force=True,
        )
        current_img = photo_res.output_path
        photo_steps.append("photo_edit")

        # 4. Typography badge overlay if requested
        if spec.video.typography is not None:
            typo_spec = spec.video.typography
            t_style = TypographyStyle(
                position=typo_spec.position,
                font_size=typo_spec.font_size,
                text_color=typo_spec.text_color,
                badge=typo_spec.badge,
            )
            apply_typography(
                current_img,
                resolved_output,
                typo_spec.text,
                config,
                style=t_style,
                force=force,
            )
            photo_steps.append("typography")
        else:
            if current_img.suffix.lower() == resolved_output.suffix.lower():
                shutil.copy2(current_img, resolved_output)
            else:
                from PIL import Image

                with Image.open(current_img) as pil_img:
                    pil_img.save(resolved_output)

        media_info = probe(resolved_output, config)
        return EditResult(
            output=resolved_output,
            media=media_info,
            steps_executed=tuple(photo_steps),
        )

    # Video workflow orchestration
    work = work_directory(config, "edit_orchestrator")
    current_clip = resolved_source
    steps_executed: list[str] = []

    # 1. Silence Jump-Cutting
    if spec.audio.silence_trim:
        silence_video = work / "step1_silence_cut.mp4"
        trim_silence(current_clip, silence_video, config, force=True)
        current_clip = silence_video
        steps_executed.append("silence_trim")

    # 2. Clean speech / Demucs stems
    if spec.audio.clean_speech:
        stems_out_dir = work / "stems"
        stems_video = work / "step2_clean_speech.mp4"
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

    # 3. Vocal Mastering EQ / Compressor / Denoise
    if spec.audio.master_profile:
        mastered_video = work / "step3_audio_enhance.mp4"
        enhance_audio(
            current_clip,
            mastered_video,
            config,
            profile=spec.audio.master_profile,
            target_lufs=spec.audio.target_lufs,
            force=True,
        )
        current_clip = mastered_video
        steps_executed.append(f"audio_master_{spec.audio.master_profile}")

    # 4. Visual looks & color grading
    if spec.video.look:
        look_video = work / "step4_graded.mp4"
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

    # 5. Reframing / vertical social format
    if spec.video.aspect != "original":
        reframe_video = work / "step5_reframed.mp4"
        if spec.video.smart_reframe:
            smart_reframe(
                current_clip,
                reframe_video,
                config,
                target_aspect=spec.video.aspect,
                mode=spec.video.reframe_mode,
                force=True,
            )
            steps_executed.append(f"smart_reframe_{spec.video.aspect}")
        else:
            to_short(
                current_clip,
                reframe_video,
                config,
                runner,
                aspect_ratio=spec.video.aspect,
                thumbnail=False,
                force=True,
            )
            steps_executed.append(f"reframe_{spec.video.aspect}")
        current_clip = reframe_video

    # 6. Retention Punch-in Zooms
    if spec.video.punch_zoom:
        zoom_video = work / "step6_zoomed.mp4"
        punch_zoom(
            current_clip,
            zoom_video,
            config,
            auto_interval_s=spec.video.zoom_interval or 5.0,
            force=True,
        )
        current_clip = zoom_video
        steps_executed.append("punch_zoom")

    # 7. B-roll cutaway inserts
    if spec.video.broll_cuts:
        broll_video = work / "step7_broll.mp4"
        cuts = [
            BrollCut(
                path=c["path"],
                start_s=float(c.get("start_s", 0.0)),
                duration_s=float(c.get("duration_s", 3.0)),
                transition=str(c.get("transition", "cut")),
                volume=float(c.get("volume", 0.0)),
            )
            for c in spec.video.broll_cuts
            if "path" in c
        ]
        if cuts:
            insert_broll(current_clip, broll_video, cuts, config, force=True)
            current_clip = broll_video
            steps_executed.append(f"broll_{len(cuts)}_cuts")

    # 8. Typography overlay badges
    if spec.video.typography is not None:
        typo_video = work / "step8_typography.mp4"
        typo_spec = spec.video.typography
        t_style = TypographyStyle(
            position=typo_spec.position,
            font_size=typo_spec.font_size,
            text_color=typo_spec.text_color,
            badge=typo_spec.badge,
        )
        apply_typography(
            current_clip,
            typo_video,
            typo_spec.text,
            config,
            style=t_style,
            start_s=typo_spec.start_s,
            duration_s=typo_spec.duration_s,
            force=True,
        )
        current_clip = typo_video
        steps_executed.append("typography")

    # 9. Whisper Subtitles
    if spec.video.subtitles.enabled:
        subs_video = work / "step9_subtitled.mp4"
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

    # 10. Procedural / custom SFX
    if spec.audio.sfx_cues:
        sfx_video = work / "step10_sfx.mp4"
        sfx_cues = [
            SfxCue(
                kind=str(c.get("kind", "whoosh")),
                at_s=float(c.get("at", 0.0)),
                volume=float(c.get("volume", 1.0)),
            )
            for c in spec.audio.sfx_cues
        ]
        add_sfx(current_clip, sfx_video, config, sfx_cues, force=True)
        current_clip = sfx_video
        steps_executed.append(f"sfx_{len(sfx_cues)}_cues")

    # 11. Speed ramping
    if spec.video.speed != 1.0:
        speed_video = work / "step11_speed.mp4"
        change_speed(current_clip, speed_video, config, speed=spec.video.speed, force=True)
        current_clip = speed_video
        steps_executed.append(f"speed_{spec.video.speed}x")

    # 12. Social retention progress bar
    if spec.video.progress_bar:
        pb_video = work / "step12_progress_bar.mp4"
        add_progress_bar(
            current_clip,
            pb_video,
            config,
            position=spec.video.progress_bar_position,
            color=spec.video.progress_bar_color,
            force=True,
        )
        current_clip = pb_video
        steps_executed.append("progress_bar")

    # 13. Background music mixing with ducking
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
        # Finalize output
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
        Expectations(requires_video=True),
    )

    return EditResult(
        output=resolved_output,
        media=final_info,
        steps_executed=tuple(steps_executed),
    )
