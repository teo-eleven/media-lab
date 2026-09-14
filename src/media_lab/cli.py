"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .config import Config, load_config
from .errors import MediaLabError
from .kino import KinoRunner
from .ml_runner import MlRunner
from .paths import clear_work_directory
from .pipeline import run_pipeline
from .recipes.audio_bed import add_music_bed
from .recipes.backdrop import place_on_backdrop
from .recipes.compose_spec import compose
from .recipes.cutout import DEVICE_CHOICES, QUALITY_CHOICES, cut_out_person
from .recipes.filters import LOOKS, apply_look, apply_look_chain
from .recipes.matte_video import MODEL_CHOICES, matte_video
from .recipes.proxy_preview import proxy_preview
from .recipes.punto_v23 import run_punto
from .recipes.subject_ground import ground_subject
from .recipes.to_short import to_short
from .recipes.upscale import upscale
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


HANDLERS = {
    "clean": _run_clean,
    "cutout": _run_cutout,
    "matte": _run_matte,
    "upscale": _run_upscale,
    "ground": _run_ground,
    "backdrop": _run_backdrop,
    "filter": _run_filter,
    "compose": _run_compose,
    "proxy": _run_proxy,
    "punto": _run_punto,
    "music": _run_music,
    "short": _run_short,
    "pipeline": _run_pipeline,
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
