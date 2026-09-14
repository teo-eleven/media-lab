"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .colour_transfer import RelightParams
from .config import Config, load_config
from .edit_spec import AudioEditSpec, EditSpec, SubtitleEditSpec, VideoEditSpec, parse_edit_spec
from .errors import MediaLabError
from .inspect import inspect_media
from .kino import KinoRunner
from .mcp_server import run_mcp_server
from .ml_runner import MlRunner
from .paths import clear_work_directory
from .pipeline import run_pipeline
from .prompt_agent import execute_prompt, plan_prompt
from .recipes.audio_bed import add_music_bed
from .recipes.audio_enhance import VOICE_PROFILES, enhance_audio
from .recipes.backdrop import place_on_backdrop
from .recipes.broll import BrollCut, insert_broll
from .recipes.colour_match import colour_match
from .recipes.compose_spec import compose
from .recipes.cutout import DEVICE_CHOICES, QUALITY_CHOICES, cut_out_person
from .recipes.edit import run_edit_spec
from .recipes.face_retouch import retouch_portrait
from .recipes.filters import LOOKS, apply_look, apply_look_chain
from .recipes.inpainting import inpaint_image
from .recipes.matte_video import MODEL_CHOICES, matte_video
from .recipes.photo import PHOTO_ASPECTS, PHOTO_LOOKS, edit_photo, process_photo_batch
from .recipes.proxy_preview import proxy_preview
from .recipes.punch_zoom import punch_zoom
from .recipes.punto_v23 import run_punto
from .recipes.scale_plate import estimate_plate_scale
from .recipes.sfx import SFX_KINDS, SfxCue, add_sfx
from .recipes.silence_trim import trim_silence
from .recipes.smart_reframe import VALID_RENAME_MODES, smart_reframe
from .recipes.stems import AUDIO_FORMAT_CHOICES, TWO_STEMS_CHOICES, separate_stems
from .recipes.subject_ground import ground_subject
from .recipes.subtitles import generate_subtitles
from .recipes.to_short import to_short
from .recipes.typography import VALID_POSITIONS, TypographyStyle, apply_typography
from .recipes.upscale import upscale
from .subtitles import STYLE_CHOICES
from .verify import ASPECT_RATIOS

RESIZE_QUALITY_CHOICES = ("low", "medium", "high", "ultra")


def _add_io_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Input media file")
    parser.add_argument("-o", "--output", required=True, help="Output path")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="media-lab",
        description="Local video and photo editing pipeline built on Kinocut.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("doctor", help="Report the state of the local environment")

    clean = subcommands.add_parser("clean", help="Empty work/, the pipeline scratch space")
    clean.add_argument(
        "--dry-run", action="store_true", help="List what would be removed, without removing it"
    )

    cutout = subcommands.add_parser("cutout", help="Cut a person out of a still or video")
    _add_io_arguments(cutout)
    cutout.add_argument("--quality", choices=QUALITY_CHOICES, default="balanced")
    cutout.add_argument("--device", choices=DEVICE_CHOICES, default="auto")

    matte = subcommands.add_parser("matte", help="Matte a person out of a video with RVM")
    _add_io_arguments(matte)
    matte.add_argument("--model", choices=MODEL_CHOICES, default="resnet50")

    upscale_cmd = subcommands.add_parser(
        "upscale", help="Upscale a video or PNG frame sequence with Real-ESRGAN"
    )
    _add_io_arguments(upscale_cmd)
    upscale_cmd.add_argument("--scale", type=int, choices=[2, 4], default=2, help="Upscale factor")
    upscale_cmd.add_argument("--tile", type=int, default=512, help="Tile size (0 for no tiling)")
    upscale_cmd.add_argument("--fps", type=float, help="Override framerate")

    ground_cmd = subcommands.add_parser(
        "ground",
        help="Ground and position subject onto canvas with foot-locking and shadow",
    )
    _add_io_arguments(ground_cmd)
    ground_cmd.add_argument("--canvas-width", type=int, default=2160, help="Canvas width")
    ground_cmd.add_argument("--canvas-height", type=int, default=3840, help="Canvas height")
    ground_cmd.add_argument("--ground-y", type=int, default=3560, help="Ground contact line Y")
    ground_cmd.add_argument("--dx", type=int, default=0, help="Horizontal offset from center")
    ground_cmd.add_argument("--scale", type=float, default=1.0, help="Base subject scale")
    ground_cmd.add_argument(
        "--no-shadow", dest="enable_shadow", action="store_false", help="Disable contact shadow"
    )
    ground_cmd.add_argument(
        "--shadow-opacity", type=float, default=1.0, help="Shadow opacity (0-1)"
    )
    ground_cmd.add_argument(
        "--no-zoom-norm",
        dest="zoom_normalise",
        action="store_false",
        help="Disable zoom normalisation",
    )
    ground_cmd.add_argument("--fps", type=float, help="Framerate override")

    scale_cmd = subcommands.add_parser(
        "scale-plate",
        help="Estimate scale and ground line for subject from background plate reference",
    )
    scale_cmd.add_argument("input", help="Cutout image or directory of cutout frames")
    scale_cmd.add_argument(
        "--ref-height", type=int, required=True, help="Reference person height in px"
    )
    scale_cmd.add_argument(
        "--ref-ground", type=int, required=True, help="Reference ground contact line Y in px"
    )
    scale_cmd.add_argument(
        "--ratio", type=float, default=1.0, help="Real-world height ratio (subject / ref)"
    )

    colour_cmd = subcommands.add_parser(
        "colour-match",
        help="Transfer background colour mood and apply scene relighting to subject",
    )
    _add_io_arguments(colour_cmd)
    colour_cmd.add_argument("--bg", help="Reference background image or plate")
    colour_cmd.add_argument(
        "--strength", type=float, default=1.0, help="Colour transfer strength (0-1)"
    )
    colour_cmd.add_argument("--bright", type=float, default=1.0, help="Brightness multiplier")
    colour_cmd.add_argument("--gamma", type=float, default=1.0, help="Gamma exponent")
    colour_cmd.add_argument("--sat", type=float, default=1.0, help="Saturation multiplier")
    colour_cmd.add_argument("--contrast", type=float, default=1.0, help="Contrast multiplier")
    colour_cmd.add_argument("--fps", type=float, help="Framerate override")

    stems_cmd = subcommands.add_parser(
        "stems",
        help="Separate audio tracks into stems and isolate speech using Demucs",
    )
    stems_cmd.add_argument("input", help="Source audio or video file")
    stems_cmd.add_argument(
        "-o",
        "--output",
        "--out-dir",
        dest="out_dir",
        required=True,
        help="Directory to save separated stems",
    )
    stems_cmd.add_argument(
        "--two-stems",
        default="vocals",
        choices=list(TWO_STEMS_CHOICES),
        help="Separate selected stem vs remainder (default: vocals)",
    )
    stems_cmd.add_argument(
        "--model", default="htdemucs", help="Demucs model name (default: htdemucs)"
    )
    stems_cmd.add_argument(
        "--format",
        dest="audio_format",
        default="wav",
        choices=list(AUDIO_FORMAT_CHOICES),
        help="Audio format for stems (default: wav)",
    )
    stems_cmd.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cpu", "cuda"],
        help="Device to run inference on (auto, mps, cpu, cuda)",
    )
    stems_cmd.add_argument(
        "--clean-speech",
        action="store_true",
        help="Mux clean vocals back onto source video",
    )
    stems_cmd.add_argument(
        "--clean-video-out",
        help="Custom destination for cleaned video if --clean-speech is set",
    )
    stems_cmd.add_argument("--force", action="store_true", help="Overwrite existing output files")

    subs_cmd = subcommands.add_parser(
        "subtitles",
        help="Transcribe audio and generate or burn styled subtitles (TikTok, clean, box)",
    )
    subs_cmd.add_argument("input", help="Source audio or video file")
    subs_cmd.add_argument(
        "-o", "--output", required=True, help="Destination subtitle (.ass/.srt) or video (.mp4)"
    )
    subs_cmd.add_argument(
        "--style",
        default="tiktok",
        choices=list(STYLE_CHOICES),
        help="Subtitle styling preset (default: tiktok)",
    )
    subs_cmd.add_argument("--model", default="base", help="Whisper model name (default: base)")
    subs_cmd.add_argument("--language", help="Language code (e.g. en, ro, es)")
    subs_cmd.add_argument(
        "--burn",
        action="store_true",
        help="Burn subtitles into video (automatic if -o ends in .mp4/.mov)",
    )
    subs_cmd.add_argument(
        "--save-subs",
        help="Also save subtitle file (.ass or .srt) when burning to video",
    )
    subs_cmd.add_argument("--force", action="store_true", help="Overwrite existing output files")

    photo_cmd = subcommands.add_parser(
        "photo",
        help="Edit social media photos (framing 1:1/4:5/9:16, backdrop, look, clarity)",
    )
    photo_cmd.add_argument("input", help="Source image or directory of images")
    photo_cmd.add_argument(
        "-o", "--output", required=True, help="Destination image file or directory"
    )
    photo_cmd.add_argument(
        "--aspect",
        default="original",
        choices=list(PHOTO_ASPECTS),
        help="Aspect ratio framing (default: original)",
    )
    photo_cmd.add_argument("--bg", help="New background image")
    photo_cmd.add_argument("--look", choices=list(PHOTO_LOOKS), help="Curated visual look")
    photo_cmd.add_argument(
        "--sharpen", action="store_true", help="Apply clarity sharpening (unsharp mask)"
    )
    photo_cmd.add_argument(
        "--cutout", action="store_true", help="Cut out subject from source before compositing"
    )
    photo_cmd.add_argument(
        "--batch", action="store_true", help="Process directory of images in batch"
    )
    photo_cmd.add_argument("--force", action="store_true", help="Overwrite existing output")

    inspect_cmd = subcommands.add_parser(
        "inspect",
        help="Deep inspection of media returning structured ground-truth metrics",
    )
    inspect_cmd.add_argument("input", help="Media file to inspect")
    inspect_cmd.add_argument(
        "--json", action="store_true", help="Output raw JSON (for LLM agents and scripts)"
    )

    edit_cmd = subcommands.add_parser(
        "edit",
        help="Declarative end-to-end editing from prompt parameters or edit-spec YAML",
    )
    edit_cmd.add_argument(
        "input", nargs="?", default=None, help="Source media file or edit-spec YAML file"
    )
    edit_cmd.add_argument("-o", "--output", help="Destination output path")
    edit_cmd.add_argument("--spec", help="Path to edit-spec YAML file")
    edit_cmd.add_argument(
        "--target", default="original", help="Target aspect ratio (9:16, 1:1, 4:5, 16:9)"
    )
    edit_cmd.add_argument(
        "--clean-speech", action="store_true", help="Clean voice and remove noise"
    )
    edit_cmd.add_argument("--look", help="Visual look (warm, cool, cinematic, etc.)")
    edit_cmd.add_argument("--subtitles", action="store_true", help="Generate and burn subtitles")
    edit_cmd.add_argument(
        "--sub-style", default="tiktok", help="Subtitle style (tiktok, clean, box)"
    )
    edit_cmd.add_argument("--music", help="Background music track to mix with ducking")
    edit_cmd.add_argument("--force", action="store_true", help="Overwrite existing output")

    backdrop = subcommands.add_parser("backdrop", help="Composite a cutout onto a backdrop")
    _add_io_arguments(backdrop)
    backdrop.add_argument("--bg", required=True, help="Backdrop image or video")
    backdrop.add_argument("--width", type=int, help="Canvas width (defaults to the backdrop)")
    backdrop.add_argument("--height", type=int, help="Canvas height (defaults to the backdrop)")
    backdrop.add_argument("--scale", type=float, default=1.0, help="Subject scale")
    backdrop.add_argument("--opacity", type=float, default=1.0, help="Subject opacity")

    look = subcommands.add_parser("filter", help="Apply a named look")
    _add_io_arguments(look)
    look.add_argument("--look", required=True, choices=sorted(LOOKS), help="Look to apply")
    look.add_argument("--then", dest="second_look", choices=sorted(LOOKS), help="A second look")

    music = subcommands.add_parser("music", help="Mix a music bed under the voice")
    _add_io_arguments(music)
    music.add_argument("--track", required=True, help="Music file")
    music.add_argument("--target-lufs", type=float, default=-16.0)
    music.add_argument("--music-volume", type=float, default=1.0)
    music.add_argument("--no-loop", dest="loop", action="store_false", help="Do not loop the bed")

    enhance_cmd = subcommands.add_parser(
        "audio-enhance",
        help="Studio vocal mastering (rumble high-pass, presence EQ, de-esser, compressor, LUFS)",
    )
    _add_io_arguments(enhance_cmd)
    enhance_cmd.add_argument(
        "--profile", default="podcast", choices=list(VOICE_PROFILES), help="Vocal EQ character"
    )
    enhance_cmd.add_argument(
        "--target-lufs", type=float, default=-14.0, help="Target loudness (default: -14.0)"
    )
    enhance_cmd.add_argument("--no-de-ess", action="store_true", help="Disable de-esser")
    enhance_cmd.add_argument("--no-gate", action="store_true", help="Disable noise gate")

    silence_cmd = subcommands.add_parser(
        "cut-silence",
        help="Automatic dead-time and pause trimming (jump-cut video/audio)",
    )
    _add_io_arguments(silence_cmd)
    silence_cmd.add_argument(
        "--min-silence", type=float, default=0.4, help="Minimum silence duration to cut (seconds)"
    )
    silence_cmd.add_argument(
        "--noise-db", type=float, default=-38.0, help="Silence noise floor threshold in dB"
    )
    silence_cmd.add_argument(
        "--padding", type=float, default=0.08, help="Speech buffer padding (seconds)"
    )

    sfx_cmd = subcommands.add_parser(
        "sfx",
        help="Synthesize and mix sound effects (whoosh, pop, ding, impact, click) onto timeline",
    )
    _add_io_arguments(sfx_cmd)
    sfx_cmd.add_argument(
        "--kind", default="whoosh", choices=list(SFX_KINDS), help="Procedural SFX or sound file"
    )
    sfx_cmd.add_argument(
        "--at", type=float, default=0.0, help="Timestamp in seconds to play the SFX"
    )
    sfx_cmd.add_argument("--volume", type=float, default=1.0, help="Sound effect volume multiplier")

    reframe_cmd = subcommands.add_parser(
        "smart-reframe",
        help="Intelligent vertical reframe with face/salience tracking or split-blur",
    )
    _add_io_arguments(reframe_cmd)
    reframe_cmd.add_argument(
        "--aspect", default="9:16", choices=list(ASPECT_RATIOS), help="Target aspect ratio"
    )
    reframe_cmd.add_argument(
        "--mode", default="smart", choices=list(VALID_RENAME_MODES), help="Reframe mode"
    )

    zoom_cmd = subcommands.add_parser(
        "punch-zoom",
        help="Dynamic retention punch-in zoom (1.1x–1.25x) for video engagement",
    )
    _add_io_arguments(zoom_cmd)
    zoom_cmd.add_argument(
        "--auto-interval",
        type=float,
        default=5.0,
        help="Rhythmic zoom interval in seconds (0 to disable auto)",
    )
    zoom_cmd.add_argument("--duration", type=float, default=2.0, help="Zoom duration in seconds")
    zoom_cmd.add_argument(
        "--scale", type=float, default=1.15, help="Zoom scale factor (e.g. 1.15 for 115%)"
    )

    broll_cmd = subcommands.add_parser(
        "broll",
        help="Overlay B-roll footage or stills preserving primary dialogue track (L/J-cut)",
    )
    _add_io_arguments(broll_cmd)
    broll_cmd.add_argument("--clip", required=True, help="Path to B-roll video or image")
    broll_cmd.add_argument("--start", type=float, default=0.0, help="Start timestamp in seconds")
    broll_cmd.add_argument("--duration", type=float, default=3.0, help="B-roll duration in seconds")
    broll_cmd.add_argument(
        "--transition", default="cut", choices=["cut", "fade"], help="Cutaway transition style"
    )
    broll_cmd.add_argument(
        "--volume", type=float, default=0.0, help="B-roll audio mix volume (0.0 = muted)"
    )

    typo_cmd = subcommands.add_parser(
        "text-overlay",
        help="Render styled typography title badge / lower-third onto photo or video",
    )
    _add_io_arguments(typo_cmd)
    typo_cmd.add_argument("--text", required=True, help="Text to render")
    typo_cmd.add_argument(
        "--position", choices=sorted(VALID_POSITIONS), default="bottom", help="Screen position"
    )
    typo_cmd.add_argument("--font-size", type=int, default=44, help="Font size in pixels")
    typo_cmd.add_argument("--text-color", default="#FFFFFF", help="Hex color for text")
    typo_cmd.add_argument(
        "--no-badge", dest="badge", action="store_false", help="Disable pill badge"
    )
    typo_cmd.add_argument(
        "--start", type=float, default=0.0, help="Video overlay start time in seconds"
    )
    typo_cmd.add_argument(
        "--duration", type=float, default=None, help="Video overlay duration in seconds"
    )

    inpaint_cmd = subcommands.add_parser(
        "inpaint",
        help="Content-aware object erasing and background reconstruction",
    )
    _add_io_arguments(inpaint_cmd)
    inpaint_cmd.add_argument("--bbox", help="Comma-separated x,y,w,h bounding box to erase")
    inpaint_cmd.add_argument("--mask", help="Path to binary mask image")
    inpaint_cmd.add_argument(
        "--method", choices=["telea", "ns"], default="telea", help="Inpainting algorithm"
    )
    inpaint_cmd.add_argument("--radius", type=int, default=5, help="Inpaint neighborhood radius")

    retouch_cmd = subcommands.add_parser(
        "retouch",
        help="Portrait retouching (edge-preserving skin smoothing and depth-of-field bokeh)",
    )
    _add_io_arguments(retouch_cmd)
    retouch_cmd.add_argument(
        "--no-smooth", dest="smooth", action="store_false", help="Disable skin smoothing"
    )
    retouch_cmd.add_argument(
        "--skin-strength", type=float, default=0.5, help="Skin smoothing strength [0.0, 1.0]"
    )
    retouch_cmd.add_argument(
        "--bokeh", type=float, default=0.0, help="Background depth blur sigma (0 to disable)"
    )
    retouch_cmd.add_argument(
        "--radiance", type=float, default=0.0, help="Skin radiance warmth [0.0, 1.0]"
    )
    retouch_cmd.add_argument("--mask", help="Optional silhouette mask for depth bokeh")

    prompt_cmd = subcommands.add_parser(
        "prompt",
        help="Natural language autonomous editor (Romanian and English prompt-to-edit)",
    )
    prompt_cmd.add_argument("prompt", help="Natural language editing instructions")
    prompt_cmd.add_argument("-i", "--input", required=True, help="Input media file path")
    prompt_cmd.add_argument("-o", "--output", required=True, help="Output destination path")
    prompt_cmd.add_argument(
        "--plan-only", action="store_true", help="Print derived action plan without executing"
    )
    prompt_cmd.add_argument("--force", action="store_true", help="Overwrite existing output")

    subcommands.add_parser(
        "mcp",
        help="Run native Model Context Protocol (MCP) server over stdio for AI integration",
    )

    short = subcommands.add_parser("short", help="Export a vertical social clip")
    _add_io_arguments(short)
    short.add_argument("--aspect", choices=sorted(ASPECT_RATIOS), default="9:16")
    short.add_argument("--quality", choices=RESIZE_QUALITY_CHOICES, default="high")
    short.add_argument("--no-thumbnail", dest="thumbnail", action="store_false")
    short.add_argument("--fail-on-warning", action="store_true")

    compose_cmd = subcommands.add_parser(
        "compose", help="Render a shot from a compose-spec YAML (bg, subject, occlusion, grade)"
    )
    compose_cmd.add_argument("spec", help="compose-spec YAML file")
    compose_cmd.add_argument("-o", "--output", required=True, help="Output path")
    compose_cmd.add_argument("--force", action="store_true", help="Overwrite an existing output")

    proxy = subcommands.add_parser(
        "proxy", help="Fast low-res proxy + contact sheet of a render (into work/)"
    )
    proxy.add_argument("input", help="Rendered clip to preview")
    proxy.add_argument("--compare", help="A second file to place side by side")
    proxy.add_argument("--height", type=int, default=540, help="Proxy height in px")
    proxy.add_argument("--frames", type=int, default=8, help="Frames on the contact sheet")
    proxy.add_argument("--force", action="store_true", help="Overwrite existing previews")

    punto = subcommands.add_parser(
        "punto", help="Reproduce the punto v23 render through matte/compose/proxy"
    )
    punto.add_argument("-o", "--output", required=True, help="Output path")
    punto.add_argument(
        "--proxy", action="store_true", help="Fast path: mobilenetv3 matte, skip the upscale"
    )
    punto.add_argument("--force", action="store_true", help="Overwrite an existing output")

    pipeline = subcommands.add_parser(
        "pipeline", help="Run the whole edit: cutout, backdrop, look, music, vertical export"
    )
    _add_io_arguments(pipeline)
    pipeline.add_argument("--bg", help="Backdrop image or video (enables the cutout stage)")
    pipeline.add_argument("--look", choices=sorted(LOOKS), help="Look to grade with")
    pipeline.add_argument("--track", help="Music bed")
    pipeline.add_argument("--aspect", choices=sorted(ASPECT_RATIOS), default="9:16")
    pipeline.add_argument("--cutout-quality", choices=QUALITY_CHOICES, default="balanced")
    pipeline.add_argument("--target-lufs", type=float, default=-16.0)
    pipeline.add_argument(
        "--fail-on-warning", action="store_true", help="Fail if the quality gate warns"
    )

    return parser


def _run_pipeline(args: argparse.Namespace, config: Config, runner: KinoRunner) -> int:
    result = run_pipeline(
        args.input,
        args.output,
        config,
        runner,
        backdrop=args.bg,
        look=args.look,
        music=args.track,
        aspect_ratio=args.aspect,
        cutout_quality=args.cutout_quality,
        target_lufs=args.target_lufs,
        force=args.force,
    )
    print("pipeline complete")
    for stage in result.stages:
        print(f"  {stage.name:<10} {stage.output}")
    print(f"final {result.final} ({result.media.width}x{result.media.height})")
    if result.thumbnail_path is not None:
        print(f"  thumbnail {result.thumbnail_path}")
    print(f"  quality score {result.quality_report.overall_score:.0f}")
    for check in result.quality_report.checks:
        if not check.passed:
            print(f"  warning: {check.name}: {check.message}")
    return 0


def _report_doctor(config: Config, runner: KinoRunner) -> int:
    print("media-lab environment")
    print(f"  root            {config.root}")
    print(f"  ffmpeg          {config.ffmpeg}")
    print(f"  ffprobe         {config.ffprobe}")
    print(f"  sources (in)    {config.in_dir}")
    print(f"  renders (out)   {config.out_dir}")
    print(f"  work            {config.work_dir}")
    print(f"  kino            {runner.executable}")
    print(f"  kino timeout    {config.kino_timeout_s}s")
    print(f"  hyperframes     {config.hyperframes_command or '(not configured)'}")
    repo_ok = (config.rvm_repo / "model" / "__init__.py").is_file()
    weights_ok = config.weights_dir.is_dir()
    print(f"  rvm repo        {config.rvm_repo} {'(ok)' if repo_ok else '(not configured)'}")
    print(f"  weights         {config.weights_dir} {'(ok)' if weights_ok else '(not configured)'}")
    print()
    print("kino doctor")
    print(runner.run(["doctor"]).stdout.rstrip())
    return 0


def _run_cutout(args: argparse.Namespace, config: Config, runner: KinoRunner) -> int:
    result = cut_out_person(
        args.input,
        args.output,
        config,
        runner,
        quality=args.quality,
        device=args.device,
        force=args.force,
    )
    print(f"cutout written to {args.output}")
    print(f"  model {result.model} on {result.provider}")
    print(f"  {result.frames_processed} frames at {result.ms_per_frame:.0f} ms/frame")
    print(f"  alpha spread {result.alpha_spread}/255 (0 would mean nothing was cut)")
    return 0


def _run_matte(args: argparse.Namespace, config: Config, _runner: KinoRunner) -> int:
    result = matte_video(
        args.input,
        args.output,
        config,
        MlRunner.from_config(config),
        model=args.model,
        force=args.force,
    )
    print(f"matte written to {args.output}")
    print(f"  model {result.model}, {result.frames} frames")
    print(f"  alpha spread {result.alpha_spread}/255 (0 would mean nothing was separated)")
    print(f"  stability score {result.stability_score:.2f} (lower = less flicker)")
    return 0


def _run_upscale(args: argparse.Namespace, config: Config, _runner: KinoRunner) -> int:
    result = upscale(
        args.input,
        args.output,
        config,
        MlRunner.from_config(config),
        scale=args.scale,
        tile=args.tile,
        fps=args.fps,
        force=args.force,
    )
    print(f"upscale written to {result.output}")
    print(f"  scale {result.scale}x, {result.frames} frames")
    return 0


def _run_ground(args: argparse.Namespace, config: Config, _runner: KinoRunner) -> int:
    result = ground_subject(
        args.input,
        args.output,
        config,
        canvas=(args.canvas_width, args.canvas_height),
        ground_y=args.ground_y,
        dx=args.dx,
        base_scale=args.scale,
        enable_shadow=args.enable_shadow,
        shadow_opacity=args.shadow_opacity,
        zoom_normalise=args.zoom_normalise,
        fps=args.fps,
        force=args.force,
    )
    print(f"grounded subject written to {result.output}")
    print(
        f"  canvas {result.canvas[0]}x{result.canvas[1]}, "
        f"ground Y {result.ground_y}, {result.frames} frames"
    )
    return 0


def _run_scale_plate(args: argparse.Namespace, config: Config, _runner: KinoRunner) -> int:
    result = estimate_plate_scale(
        args.input,
        config,
        ref_height=args.ref_height,
        ref_ground_y=args.ref_ground,
        height_ratio=args.ratio,
    )
    print(f"scale estimate for {args.input}:")
    print(f"  measured subject height: {result.subject_height_px}px")
    print(f"  target height:           {result.target_height_px}px")
    print(f"  recommended scale:       {result.base_scale:.4f}")
    print(f"  recommended ground-y:    {result.ground_y}")
    print(
        f"  hint: pass --scale {result.base_scale:.4f} "
        f"--ground-y {result.ground_y} to media-lab ground"
    )
    return 0


def _run_colour_match(args: argparse.Namespace, config: Config, _runner: KinoRunner) -> int:
    params = RelightParams(
        bright=args.bright,
        gamma=args.gamma,
        sat=args.sat,
        contrast=args.contrast,
    )
    result = colour_match(
        args.input,
        args.output,
        config,
        bg_ref=args.bg,
        transfer_strength=args.strength,
        params=params,
        fps=args.fps,
        force=args.force,
    )
    print(f"colour-matched subject written to {result.output} ({result.frames} frames)")
    return 0


def _run_stems(args: argparse.Namespace, config: Config, _runner: KinoRunner) -> int:
    result = separate_stems(
        args.input,
        args.out_dir,
        config,
        MlRunner.from_config(config),
        two_stems=args.two_stems,
        model=args.model,
        audio_format=args.audio_format,
        device=args.device,
        clean_speech=args.clean_speech,
        output_video=args.clean_video_out,
        force=args.force,
    )
    print(f"separated stems saved to {result.output_dir}:")
    for name, path in sorted(result.stems.items()):
        print(f"  {name}: {path}")
    if result.cleaned_video is not None:
        print(f"clean speech video: {result.cleaned_video}")
    return 0


def _run_subtitles(args: argparse.Namespace, config: Config, _runner: KinoRunner) -> int:
    result = generate_subtitles(
        args.input,
        args.output,
        config,
        MlRunner.from_config(config),
        style=args.style,
        model=args.model,
        language=args.language,
        burn=args.burn,
        output_subs=args.save_subs,
        force=args.force,
    )
    print(f"subtitles generated ({result.language}, {result.segment_count} segments):")
    print(f"  subtitle file: {result.subtitle_file}")
    if result.video_file is not None:
        print(f"  burned video:  {result.video_file}")
    return 0


def _run_photo(args: argparse.Namespace, config: Config, runner: KinoRunner) -> int:
    input_path = Path(args.input)
    if args.batch or input_path.is_dir():
        batch_res = process_photo_batch(
            args.input,
            args.output,
            config,
            runner,
            aspect=args.aspect,
            bg=args.bg,
            look=args.look,
            sharpen=args.sharpen,
            force=args.force,
        )
        print(f"batch photo processed {len(batch_res.results)} images to {batch_res.output_dir}")
        for r in batch_res.results:
            print(f"  {r.output_path.name} ({r.width}x{r.height})")
    else:
        res = edit_photo(
            args.input,
            args.output,
            config,
            runner,
            aspect=args.aspect,
            bg=args.bg,
            cutout=args.cutout,
            look=args.look,
            sharpen=args.sharpen,
            force=args.force,
        )
        print(f"photo written to {res.output_path} ({res.width}x{res.height}, aspect {res.aspect})")
    return 0


def _run_inspect(args: argparse.Namespace, config: Config, _runner: KinoRunner) -> int:
    report = inspect_media(args.input, config)
    if args.json:
        print(report.to_json())
    else:
        print(f"media inspection: {report.file_path}")
        print(f"  type:         {report.media_type}")
        print(f"  dimensions:   {report.width}x{report.height} ({report.aspect_ratio})")
        print(f"  duration:     {report.duration_s:.2f}s @ {report.fps:.1f} fps")
        if report.visual is not None:
            print(
                f"  brightness:   {report.visual.brightness_mean:.1f}, "
                f"contrast {report.visual.contrast_std:.1f}"
            )
            print(f"  palette:      {', '.join(report.visual.dominant_colors)}")
            print(f"  silhouette:   {report.visual.has_human_silhouette}")
        if report.audio is not None:
            print(f"  speech:       {report.audio.has_speech}")
            if report.audio.rms_db is not None:
                print(f"  rms level:    {report.audio.rms_db:.1f} dB")
    return 0


def _run_edit(args: argparse.Namespace, config: Config, runner: KinoRunner) -> int:
    ml_runner = MlRunner.from_config(config)
    if args.spec:
        spec = parse_edit_spec(Path(args.spec))
    elif args.input and str(args.input).endswith((".yaml", ".yml", ".json")):
        spec = parse_edit_spec(Path(args.input))
    elif args.input:
        input_str = str(args.input)
        if not args.output:
            print(
                "error: --output (-o) is required when specifying direct edit flags",
                file=sys.stderr,
            )
            return 1
        video_spec = VideoEditSpec(
            aspect=args.target,
            look=args.look,
            subtitles=SubtitleEditSpec(
                enabled=args.subtitles,
                style=args.sub_style,
            ),
        )
        audio_spec = AudioEditSpec(
            clean_speech=args.clean_speech,
            music_track=args.music,
        )
        spec = EditSpec(
            source=input_str,
            output=str(args.output),
            video=video_spec,
            audio=audio_spec,
        )
    else:
        print("error: must provide either an input file or --spec <path>", file=sys.stderr)
        return 1

    result = run_edit_spec(spec, config, runner, ml_runner, force=args.force)
    print(f"edit finished: {result.output}")
    print(f"  steps executed: {', '.join(result.steps_executed)}")
    print(f"  output: {result.media.width}x{result.media.height}, {result.media.duration_s:.2f}s")
    return 0


def _run_audio_enhance(args: argparse.Namespace, config: Config, _runner: KinoRunner) -> int:
    res = enhance_audio(
        args.input,
        args.output,
        config,
        profile=args.profile,
        target_lufs=args.target_lufs,
        de_ess=not args.no_de_ess,
        noise_gate=not args.no_gate,
        force=args.force,
    )
    print(f"audio enhanced: {res.output} ({res.profile} profile, {res.target_lufs} LUFS)")
    return 0


def _run_cut_silence(args: argparse.Namespace, config: Config, _runner: KinoRunner) -> int:
    res = trim_silence(
        args.input,
        args.output,
        config,
        min_silence_s=args.min_silence,
        noise_db=args.noise_db,
        padding_s=args.padding,
        force=args.force,
    )
    print(f"silence trimmed: {res.output}")
    print(f"  original: {res.original_duration_s:.2f}s -> new: {res.new_duration_s:.2f}s")
    print(f"  removed:  {res.removed_duration_s:.2f}s across {res.cuts_count} pauses")
    return 0


def _run_sfx(args: argparse.Namespace, config: Config, _runner: KinoRunner) -> int:
    cue = SfxCue(kind=args.kind, at_s=args.at, volume=args.volume)
    res = add_sfx(
        args.input,
        args.output,
        config,
        [cue],
        force=args.force,
    )
    print(f"sfx mixed: {res.output} ({args.kind} at {args.at:.2f}s, vol {args.volume})")
    return 0


def _run_backdrop(args: argparse.Namespace, config: Config, runner: KinoRunner) -> int:
    result = place_on_backdrop(
        args.input,
        args.bg,
        args.output,
        config,
        runner,
        width=args.width,
        height=args.height,
        scale=args.scale,
        opacity=args.opacity,
        force=args.force,
    )
    print(f"composite written to {args.output}")
    print(f"  canvas {result.canvas_width}x{result.canvas_height}")
    print(f"  spec kept at {result.spec_path}")
    if result.fps_was_clamped:
        print(
            f"  note: rendered at {result.canvas_fps:.0f} fps "
            f"(source is {result.source_fps:.0f}); kinocut's compositor caps there"
        )
    if result.subject_overflows_canvas:
        print(
            f"  warning: subject is {result.subject_width}x{result.subject_height}, "
            f"larger than the {result.canvas_width}x{result.canvas_height} canvas; it is cropped"
        )
    if result.subject_aspect_differs:
        print("  warning: subject and canvas have different shapes; framing shifts")
    if result.backdrop_was_shorter:
        print("  warning: the backdrop video is shorter than the subject")
    return 0


def _run_compose(args: argparse.Namespace, config: Config, _runner: KinoRunner) -> int:
    result = compose(args.spec, args.output, config, force=args.force)
    print(
        f"composed {args.output} "
        f"({result.media.width}x{result.media.height}, {result.media.duration_s:.2f}s)"
    )
    print(f"  filtergraph kept at {result.filtergraph_path}")
    return 0


def _run_proxy(args: argparse.Namespace, config: Config, _runner: KinoRunner) -> int:
    result = proxy_preview(
        args.input,
        config,
        height=args.height,
        frames=args.frames,
        compare=args.compare,
        force=args.force,
    )
    print(f"proxy   {result.proxy}")
    print(f"sheet   {result.sheet}")
    if result.comparison is not None:
        print(f"vs      {result.comparison}")
    return 0


def _run_punto(args: argparse.Namespace, config: Config, _runner: KinoRunner) -> int:
    result = run_punto(
        config, MlRunner.from_config(config), args.output, proxy=args.proxy, force=args.force
    )
    mode = "proxy" if result.proxy_mode else "full"
    print(f"punto ({mode}) written to {result.output}")
    print(f"  matte {result.matte.model}, stability score {result.matte.stability_score:.2f}")
    print(
        f"  composed {result.composed.media.width}x{result.composed.media.height}, "
        f"{result.composed.media.duration_s:.2f}s"
    )
    print(f"  contact sheet {result.preview.sheet}")
    if result.preview.comparison is not None:
        print(f"  vs reference   {result.preview.comparison}")
    if result.proxy_mode:
        print("  note: --proxy skips the upscale, so the subject is ~half v23 scale")
    return 0


def _run_filter(args: argparse.Namespace, config: Config, runner: KinoRunner) -> int:
    if args.second_look is None:
        info = apply_look(args.input, args.output, args.look, config, runner, force=args.force)
    else:
        info = apply_look_chain(
            args.input, args.output, args.look, args.second_look, config, runner, force=args.force
        )
    print(f"look written to {args.output} ({info.width}x{info.height})")
    return 0


def _format_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _run_clean(args: argparse.Namespace, config: Config, _runner: KinoRunner) -> int:
    entries, total_bytes = clear_work_directory(config, dry_run=args.dry_run)
    if not entries:
        print(f"{config.work_dir} is already empty")
        return 0

    verb = "would remove" if args.dry_run else "removed"
    print(f"{verb} {len(entries)} item(s) from {config.work_dir} ({_format_bytes(total_bytes)})")
    for entry in entries:
        print(f"  {entry.name}")
    return 0


def _run_music(args: argparse.Namespace, config: Config, _runner: KinoRunner) -> int:
    result = add_music_bed(
        args.input,
        args.track,
        args.output,
        config,
        target_lufs=args.target_lufs,
        music_volume=args.music_volume,
        loop=args.loop,
        force=args.force,
    )
    print(f"mix written to {args.output}")
    print(f"  measured loudness {result.measured_lufs:.1f} LUFS (target {args.target_lufs})")
    print(f"  ducking engaged: {result.ducking_engaged}")
    return 0


def _run_short(args: argparse.Namespace, config: Config, runner: KinoRunner) -> int:
    result = to_short(
        args.input,
        args.output,
        config,
        runner,
        aspect_ratio=args.aspect,
        quality=args.quality,
        thumbnail=args.thumbnail,
        fail_on_warning=args.fail_on_warning,
        force=args.force,
    )
    print(f"short written to {args.output} ({result.info.width}x{result.info.height})")
    if result.thumbnail_path is not None:
        print(f"  thumbnail {result.thumbnail_path}")
    print(f"  quality score {result.quality_report.overall_score:.0f}")
    for check in result.quality_report.checks:
        if not check.passed:
            print(f"  warning: {check.name}: {check.message}")
    return 0


def _run_smart_reframe(args: argparse.Namespace, config: Config, runner: KinoRunner) -> int:
    result = smart_reframe(
        args.input,
        args.output,
        config,
        target_aspect=args.aspect,
        mode=args.mode,
        force=args.force,
    )
    print(f"smart-reframe written to {args.output}")
    print(f"  aspect {result.target_aspect}, mode {result.mode}")
    if result.crop_box:
        print(f"  crop box: {result.crop_box}")
    return 0


def _run_punch_zoom(args: argparse.Namespace, config: Config, runner: KinoRunner) -> int:
    result = punch_zoom(
        args.input,
        args.output,
        config,
        auto_interval_s=args.auto_interval if args.auto_interval > 0 else None,
        auto_zoom_duration_s=args.duration,
        auto_scale=args.scale,
        force=args.force,
    )
    print(f"punch-zoom written to {args.output}")
    print(f"  cues applied: {result.cues_applied}")
    return 0


def _run_broll(args: argparse.Namespace, config: Config, runner: KinoRunner) -> int:
    cuts = [
        BrollCut(
            path=args.clip,
            start_s=args.start,
            duration_s=args.duration,
            transition=args.transition,
            volume=args.volume,
        )
    ]
    result = insert_broll(
        args.input,
        args.output,
        cuts,
        config,
        force=args.force,
    )
    print(f"broll written to {args.output}")
    print(f"  cuts applied: {result.cuts_count}")
    return 0


def _run_text_overlay(args: argparse.Namespace, config: Config, runner: KinoRunner) -> int:
    style = TypographyStyle(
        position=args.position,
        font_size=args.font_size,
        text_color=args.text_color,
        badge=args.badge,
    )
    result = apply_typography(
        args.input,
        args.output,
        args.text,
        config,
        style=style,
        start_s=args.start,
        duration_s=args.duration,
        force=args.force,
    )
    media_type = "video" if result.is_video else "image"
    print(f"text-overlay written to {args.output} ({media_type}, {result.width}x{result.height})")
    print(f"  lines rendered: {result.lines_rendered}")
    return 0


def _run_inpaint(args: argparse.Namespace, config: Config, runner: KinoRunner) -> int:
    bbox: tuple[int, int, int, int] | None = None
    if args.bbox:
        parts = [int(p.strip()) for p in args.bbox.split(",")]
        if len(parts) != 4:
            raise MediaLabError("--bbox must be 'x,y,w,h' with 4 integer values")
        bbox = (parts[0], parts[1], parts[2], parts[3])

    result = inpaint_image(
        args.input,
        args.output,
        config,
        bbox=bbox,
        mask_path=args.mask,
        method=args.method,
        inpaint_radius=args.radius,
        force=args.force,
    )
    print(f"inpaint written to {args.output} ({result.width}x{result.height})")
    print(f"  method: {result.method}, erased pixels: {result.erased_pixels}")
    return 0


def _run_retouch(args: argparse.Namespace, config: Config, runner: KinoRunner) -> int:
    result = retouch_portrait(
        args.input,
        args.output,
        config,
        smooth_skin=args.smooth,
        skin_strength=args.skin_strength,
        depth_blur=args.bokeh > 0.0,
        blur_sigma=args.bokeh,
        mask_path=args.mask,
        radiance=args.radiance,
        force=args.force,
    )
    print(f"retouch written to {args.output} ({result.width}x{result.height})")
    print(f"  skin smoothed: {result.skin_smoothed}, depth blur: {result.depth_blur_applied}")
    return 0


def _run_prompt(args: argparse.Namespace, config: Config, runner: KinoRunner) -> int:
    ml_runner = MlRunner.from_config(config)
    if args.plan_only:
        plan = plan_prompt(args.prompt, args.input, args.output)
        print(f"Plan derived from: {args.prompt!r}")
        for i, op in enumerate(plan.operations, start=1):
            print(f"  {i}. {op}")
        return 0

    result = execute_prompt(
        args.prompt,
        args.input,
        args.output,
        config,
        runner,
        ml_runner,
        force=args.force,
    )
    print(f"prompt executed -> {args.output}")
    print(f"  steps executed: {', '.join(result.steps_executed)}")
    return 0


def _run_mcp(args: argparse.Namespace, config: Config, runner: KinoRunner) -> int:
    run_mcp_server(config)
    return 0


HANDLERS = {
    "clean": _run_clean,
    "cutout": _run_cutout,
    "matte": _run_matte,
    "upscale": _run_upscale,
    "ground": _run_ground,
    "scale-plate": _run_scale_plate,
    "colour-match": _run_colour_match,
    "stems": _run_stems,
    "subtitles": _run_subtitles,
    "photo": _run_photo,
    "inspect": _run_inspect,
    "edit": _run_edit,
    "backdrop": _run_backdrop,
    "filter": _run_filter,
    "compose": _run_compose,
    "proxy": _run_proxy,
    "punto": _run_punto,
    "music": _run_music,
    "short": _run_short,
    "pipeline": _run_pipeline,
    "audio-enhance": _run_audio_enhance,
    "cut-silence": _run_cut_silence,
    "sfx": _run_sfx,
    "smart-reframe": _run_smart_reframe,
    "punch-zoom": _run_punch_zoom,
    "broll": _run_broll,
    "text-overlay": _run_text_overlay,
    "inpaint": _run_inpaint,
    "retouch": _run_retouch,
    "prompt": _run_prompt,
    "mcp": _run_mcp,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config()
        runner = KinoRunner.from_config(config)
        if args.command == "doctor":
            return _report_doctor(config, runner)
        handler = HANDLERS.get(args.command)
        if handler is None:
            parser.error(f"unhandled command: {args.command}")
        return handler(args, config, runner)
    except MediaLabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
