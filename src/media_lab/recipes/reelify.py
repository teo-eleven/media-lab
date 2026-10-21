"""Recipe: Reelify — All-in-One Viral Short Video Transformer.

Synthesizes the best capabilities of Opus Clip, Descript, Captions.ai, and CapCut AI
into a single, unified, local-first short-form video generator:
1. Silence Purge: Cuts dead pauses (>0.4s) while preserving vocal rhythm.
2. Voice Studio Mastering: 80Hz high-pass, presence EQ, de-esser 6.5kHz, compressor, gate.
3. Smart 9:16 Reframe: Subject/face-tracking vertical reframe.
4. Kinetic Subtitles: Whisper speech-to-text with styled ASS subtitles in the social safe zone.
5. Retention Progress Bar: Dynamic animated progress indicator along the bottom.
6. Optional Retention Punch-Zoom: Dynamic 1.15x zooms on key moments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..errors import ValidationError
from ..ml_runner import MlRunner
from ..paths import ensure_readable_source, ensure_writable_output, work_directory
from ..probe import MediaInfo, probe
from .audio_enhance import enhance_audio
from .progress_bar import add_progress_bar
from .punch_zoom import punch_zoom
from .silence_trim import trim_silence
from .smart_reframe import smart_reframe
from .subtitles import generate_subtitles


@dataclass(frozen=True, slots=True)
class ReelifyResult:
    """Outcome of the complete Reelify pipeline."""

    output: Path
    media: MediaInfo
    stages_executed: tuple[str, ...]
    subtitles_burned: bool


def reelify(
    source: Path | str,
    output: Path | str,
    config: Config,
    ml_runner: MlRunner,
    *,
    subtitles_style: str = "tiktok",
    burn_subtitles: bool = True,
    whisper_model: str = "base",
    enable_punch_zoom: bool = True,
    enable_progress_bar: bool = True,
    progress_bar_color: str = "yellow",
    silence_threshold_db: float = -32.0,
    min_silence_s: float = 0.4,
    force: bool = False,
) -> ReelifyResult:
    """Transform raw footage into a polished, high-retention 9:16 social short.

    Args:
        source: Input raw media file.
        output: Destination vertical video file.
        config: System configuration.
        ml_runner: Subprocess runner for Whisper ML.
        subtitles_style: ASS subtitle preset ('tiktok', 'clean', 'box').
        burn_subtitles: Whether to transcribe and burn subtitles.
        whisper_model: Whisper model size ('tiny', 'base', 'small').
        enable_punch_zoom: Whether to add rhythmic retention punch zooms.
        enable_progress_bar: Whether to overlay an animated retention progress bar.
        progress_bar_color: Color for the progress bar ('yellow', 'tiktok', 'green', 'white').
        silence_threshold_db: Silence detection dB threshold.
        min_silence_s: Minimum pause duration to cut out.
        force: Overwrite output if it already exists.

    Returns:
        ReelifyResult with the final path, media information, and applied stages.
    """
    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    source_info = probe(resolved_source, config)
    if not source_info.has_video:
        raise ValidationError(f"Source {resolved_source} has no video stream to reelify.")

    work = work_directory(config, "reelify")
    stages: list[str] = []
    current_clip = resolved_source

    # Stage 1: Silence Purge (if audio exists)
    if source_info.has_audio:
        silence_out = work / f"{resolved_source.stem}_s1_cut.mp4"
        try:
            silence_res = trim_silence(
                current_clip,
                silence_out,
                config,
                min_silence_s=min_silence_s,
                noise_db=silence_threshold_db,
                force=True,
            )
            current_clip = silence_res.output
            stages.append("silence_purge")
        except Exception:
            # Fallback gracefully if silence detection fails or video has no pauses
            pass

        # Stage 2: Voice Studio Mastering
        audio_out = work / f"{resolved_source.stem}_s2_enhanced.mp4"
        try:
            enhanced_res = enhance_audio(
                current_clip,
                audio_out,
                config,
                de_ess=True,
                noise_gate=True,
                force=True,
            )
            current_clip = enhanced_res.output
            stages.append("audio_enhance")
        except Exception:
            pass

    # Stage 3: Smart Reframe to 9:16
    reframe_out = work / f"{resolved_source.stem}_s3_reframe.mp4"
    reframe_res = smart_reframe(
        current_clip,
        reframe_out,
        config,
        target_aspect="9:16",
        mode="smart",
        force=True,
    )
    current_clip = reframe_res.output
    stages.append("smart_reframe_9:16")

    # Stage 4: Retention Punch-Zoom
    if enable_punch_zoom:
        zoom_out = work / f"{resolved_source.stem}_s4_zoom.mp4"
        try:
            zoom_res = punch_zoom(
                current_clip,
                zoom_out,
                config,
                auto_interval_s=4.0,
                auto_scale=1.15,
                force=True,
            )
            current_clip = zoom_res.output
            stages.append("punch_zoom")
        except Exception:
            pass

    # Stage 5: Kinetic Subtitles (if enabled & audio present)
    subtitles_applied = False
    if burn_subtitles and source_info.has_audio:
        sub_out = work / f"{resolved_source.stem}_s5_subbed.mp4"
        try:
            sub_res = generate_subtitles(
                current_clip,
                sub_out,
                config,
                ml_runner,
                style=subtitles_style,
                model=whisper_model,
                burn=True,
                kinetic=True,
                emojis=True,
                force=True,
            )
            if sub_res.video_file and sub_res.video_file.exists():
                current_clip = sub_res.video_file
                subtitles_applied = True
                stages.append("subtitles")
        except Exception:
            pass

    # Stage 6: Progress Bar (Final stage directly into resolved_output)
    if enable_progress_bar:
        prog_res = add_progress_bar(
            current_clip,
            resolved_output,
            config,
            position="bottom",
            height=8,
            color=progress_bar_color,
            force=force,
        )
        final_clip = prog_res.output
        stages.append("progress_bar")
    else:
        # Move or copy current_clip to resolved_output
        import shutil

        shutil.copy2(current_clip, resolved_output)
        final_clip = resolved_output

    final_info = probe(final_clip, config)

    return ReelifyResult(
        output=final_clip,
        media=final_info,
        stages_executed=tuple(stages),
        subtitles_burned=subtitles_applied,
    )
