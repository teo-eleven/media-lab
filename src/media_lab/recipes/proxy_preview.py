"""Fast low-res proxy of a render, plus a contact sheet - and optionally a
side-by-side sheet against a reference file.

Every version used to be an ~8-minute encode followed by a hand-built contact
sheet; this makes that one command.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..contact_sheet import extract_frames, side_by_side, tile
from ..ffmpeg import run_ffmpeg
from ..paths import ensure_readable_source, ensure_writable_output, work_directory, work_path
from ..verify import verify_render

DEFAULT_HEIGHT = 540
DEFAULT_FRAMES = 8
DEFAULT_COLS = 4
MIN_FRAMES = 2


@dataclass(frozen=True, slots=True)
class ProxyResult:
    proxy: Path
    sheet: Path
    comparison: Path | None


def _encode_proxy(source: Path, height: int, output: Path, config: Config) -> None:
    run_ffmpeg(
        ["-i", str(source), "-vf", f"scale=-2:{height}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
         "-an", "-movflags", "+faststart", str(output)],
        config,
    )


def proxy_preview(
    source: Path | str,
    config: Config,
    *,
    height: int = DEFAULT_HEIGHT,
    frames: int = DEFAULT_FRAMES,
    cols: int = DEFAULT_COLS,
    compare: Path | str | None = None,
    force: bool = False,
) -> ProxyResult:
    """Render a proxy of `source` and a contact sheet, both under `work/`.

    With `compare`, also proxies that file and writes a side-by-side sheet
    (`compare` on the left, `source` on the right).
    """
    if frames < MIN_FRAMES:
        raise ValueError(f"frames must be at least {MIN_FRAMES}, got {frames}")
    resolved_source = ensure_readable_source(source)
    stem = resolved_source.stem

    proxy = ensure_writable_output(work_path(config, f"{stem}-proxy", ".mp4"), config, force=force)
    _encode_proxy(resolved_source, height, proxy, config)
    verify_render(proxy, config)

    sheet = ensure_writable_output(work_path(config, f"{stem}-sheet", ".jpg"), config, force=force)
    tile(extract_frames(proxy, frames, config, work_directory(config, f"{stem}-cs")),
         sheet, config, cols=cols)

    comparison: Path | None = None
    if compare is not None:
        reference = ensure_readable_source(compare)
        ref_proxy = work_path(config, f"{stem}-cmp-proxy", ".mp4")
        _encode_proxy(reference, height, ref_proxy, config)
        ref_sheet = work_path(config, f"{stem}-cmp-sheet", ".jpg")
        tile(extract_frames(ref_proxy, frames, config, work_directory(config, f"{stem}-cs-cmp")),
             ref_sheet, config, cols=cols)
        comparison = ensure_writable_output(
            work_path(config, f"{stem}-vs", ".jpg"), config, force=force
        )
        side_by_side(ref_sheet, sheet, comparison, config)

    return ProxyResult(proxy=proxy, sheet=sheet, comparison=comparison)
