"""Recipe: NLE Timeline Exporter (FCPXML & CMX 3600 EDL).

Exports editing projects, multi-clip assemblies, and EditSpecs to industry-standard
exchange formats natively supported by:
- DaVinci Resolve (Blackmagic Design)
- Apple Final Cut Pro (v1.10 XML)
- Adobe Premiere Pro (FCPXML & CMX 3600 EDL)
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..errors import ValidationError
from ..paths import ensure_readable_source, ensure_writable_output
from ..probe import probe


@dataclass(frozen=True, slots=True)
class TimelineClip:
    """Represents a video/audio cut within the timeline."""

    name: str
    source_path: Path
    start_s: float
    duration_s: float
    timeline_start_s: float
    has_video: bool = True
    has_audio: bool = True


@dataclass(frozen=True, slots=True)
class TimelineData:
    """Complete project sequence structure."""

    project_name: str
    fps: float
    width: int
    height: int
    clips: tuple[TimelineClip, ...]


@dataclass(frozen=True, slots=True)
class TimelineExportResult:
    """Outcome of exporting a timeline project file."""

    output: Path
    format: str
    clip_count: int
    total_duration_s: float


def seconds_to_timecode(seconds: float, fps: float = 25.0) -> str:
    """Convert floating-point seconds into SMPTE non-drop frame timecode (HH:MM:SS:FF)."""
    if seconds < 0:
        seconds = 0.0

    total_frames = int(round(seconds * fps))
    frame_rate_int = int(round(fps))

    ff = total_frames % frame_rate_int
    total_seconds = total_frames // frame_rate_int
    ss = total_seconds % 60
    mm = (total_seconds // 60) % 60
    hh = total_seconds // 3600

    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def generate_cmx3600_edl(data: TimelineData) -> str:
    """Generate CMX 3600 formatted Edit Decision List text."""
    lines: list[str] = [
        f"TITLE: {data.project_name.upper().replace(' ', '_')}",
        "FCM: NON-DROP FRAME",
        "",
    ]

    for i, clip in enumerate(data.clips, 1):
        idx_str = f"{i:03d}"
        reel = "AX"

        src_in = seconds_to_timecode(clip.start_s, data.fps)
        src_out = seconds_to_timecode(clip.start_s + clip.duration_s, data.fps)

        rec_in = seconds_to_timecode(clip.timeline_start_s, data.fps)
        rec_out = seconds_to_timecode(clip.timeline_start_s + clip.duration_s, data.fps)

        # Video event
        if clip.has_video:
            lines.append(
                f"{idx_str}  {reel:<8} V     C        {src_in} {src_out} {rec_in} {rec_out}"
            )
            lines.append(f"* FROM CLIP NAME: {clip.name}")

        # Audio event
        if clip.has_audio:
            lines.append(
                f"{idx_str}  {reel:<8} A     C        {src_in} {src_out} {rec_in} {rec_out}"
            )
            lines.append(f"* FROM CLIP NAME: {clip.name}")

    return "\n".join(lines).strip() + "\n"


def generate_fcpxml(data: TimelineData) -> str:
    """Generate Apple Final Cut Pro XML (v1.10) for DaVinci Resolve & FCP."""
    total_dur_s = sum(c.duration_s for c in data.clips)
    fps_int = int(round(data.fps))
    format_id = "r1"

    # XML resources
    resources_xml = [
        f'    <format id="{format_id}" name="FFVideoFormat{data.height}p{fps_int}" '
        f'frameDuration="1/{fps_int}s" width="{data.width}" height="{data.height}"/>'
    ]

    asset_map: dict[str, str] = {}
    for i, clip in enumerate(data.clips, 2):
        asset_id = f"r{i}"
        asset_map[str(clip.source_path)] = asset_id
        src_uri = html.escape(clip.source_path.resolve().as_uri())
        clip_name = html.escape(clip.name)
        has_v = "1" if clip.has_video else "0"
        has_a = "1" if clip.has_audio else "0"
        dur_str = f"{int(round(clip.duration_s * 100))}/100s"

        resources_xml.append(
            f'    <asset id="{asset_id}" name="{clip_name}" src="{src_uri}" '
            f'duration="{dur_str}" hasVideo="{has_v}" hasAudio="{has_a}" '
            f'audioSources="1" audioChannels="2"/>'
        )

    # Sequence spine clips
    spine_xml: list[str] = []
    current_offset_s = 0.0

    for clip in data.clips:
        asset_id = asset_map[str(clip.source_path)]
        clip_name = html.escape(clip.name)
        offset_str = f"{int(round(current_offset_s * 100))}/100s"
        start_str = f"{int(round(clip.start_s * 100))}/100s"
        dur_str = f"{int(round(clip.duration_s * 100))}/100s"

        spine_xml.append(
            f'          <asset-clip ref="{asset_id}" offset="{offset_str}" name="{clip_name}" '
            f'start="{start_str}" duration="{dur_str}"/>'
        )
        current_offset_s += clip.duration_s

    total_dur_str = f"{int(round(total_dur_s * 100))}/100s"
    proj_name = html.escape(data.project_name)

    res_block = "\n".join(resources_xml)
    spine_block = "\n".join(spine_xml)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.10">
  <resources>
{res_block}
  </resources>
  <library>
    <event name="Media Lab Export">
      <project name="{proj_name}">
        <sequence format="{format_id}" duration="{total_dur_str}" tcStart="0s" tcFormat="NDF">
          <spine>
{spine_block}
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
"""


def export_timeline(
    source: Path | str,
    output: Path | str,
    config: Config,
    *,
    format: str = "fcpxml",
    fps: float = 25.0,
    project_name: str | None = None,
    force: bool = False,
) -> TimelineExportResult:
    """Export source clip or assembly sequence into DaVinci / Premiere timeline format.

    Args:
        source: Input media file or project source.
        output: Destination .fcpxml or .edl file.
        config: System configuration.
        format: 'fcpxml' (Final Cut Pro / DaVinci XML) or 'edl' (CMX 3600 EDL).
        fps: Timeline frame rate (default: 25.0).
        project_name: Optional name for the sequence/project.
        force: Overwrite existing file.
    """
    resolved_source = ensure_readable_source(source)
    resolved_output = ensure_writable_output(output, config, force=force)

    fmt = format.lower().strip()
    if fmt not in ("fcpxml", "edl"):
        raise ValidationError(
            f"unsupported timeline export format: {format}. Use 'fcpxml' or 'edl'."
        )

    src_info = probe(resolved_source, config)
    width = src_info.width if src_info.width > 0 else 1920
    height = src_info.height if src_info.height > 0 else 1080
    clip_fps = src_info.fps if src_info.fps > 0 else fps
    dur = src_info.duration_s if src_info.duration_s > 0 else 5.0

    p_name = project_name or resolved_source.stem

    clip = TimelineClip(
        name=resolved_source.name,
        source_path=resolved_source,
        start_s=0.0,
        duration_s=dur,
        timeline_start_s=0.0,
        has_video=src_info.has_video,
        has_audio=src_info.has_audio,
    )

    timeline_data = TimelineData(
        project_name=p_name,
        fps=clip_fps,
        width=width,
        height=height,
        clips=(clip,),
    )

    if fmt == "fcpxml":
        content = generate_fcpxml(timeline_data)
    else:
        content = generate_cmx3600_edl(timeline_data)

    resolved_output.write_text(content, encoding="utf-8")

    return TimelineExportResult(
        output=resolved_output,
        format=fmt,
        clip_count=len(timeline_data.clips),
        total_duration_s=dur,
    )
