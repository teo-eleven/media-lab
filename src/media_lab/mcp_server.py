"""Native Model Context Protocol (MCP) server for media-lab.

Provides a standard stdio JSON-RPC 2.0 interface exposing all media-lab tools
directly to AI agents (Antigravity, Cursor, Claude Desktop, etc.).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .config import Config, load_config
from .errors import MediaLabError
from .inspect import inspect_media
from .kino import KinoRunner
from .ml_runner import MlRunner
from .prompt_agent import execute_prompt
from .recipes.assembly import assemble_clips
from .recipes.audio_enhance import enhance_audio
from .recipes.auto_hook import extract_viral_hook
from .recipes.beat_sync import detect_beats
from .recipes.brand_kit import apply_brand_watermark
from .recipes.face_retouch import retouch_portrait
from .recipes.footage_scout import scout_library
from .recipes.inpainting import inpaint_image
from .recipes.narrator import generate_narration
from .recipes.omni_export import omni_export
from .recipes.progress_bar import add_progress_bar
from .recipes.punch_zoom import punch_zoom
from .recipes.reelify import reelify
from .recipes.render_doctor import auto_fix_render, diagnose_render
from .recipes.silence_trim import trim_silence
from .recipes.smart_reframe import smart_reframe
from .recipes.speed import change_speed
from .recipes.typography import TypographyStyle, apply_typography

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "media-lab", "version": "0.2.0"}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "inspect_media",
        "description": (
            "Deeply inspect an audio/video/photo asset and return resolution, duration, "
            "audio LUFS, dominant colors, and presence of human subjects."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the media file"}},
            "required": ["path"],
        },
    },
    {
        "name": "execute_prompt",
        "description": (
            "Execute an autonomous photo, video, or audio editing pipeline directly "
            "from a natural language prompt (in Romanian or English)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Editing instructions in natural language",
                },
                "source": {"type": "string", "description": "Input media file path"},
                "output": {"type": "string", "description": "Destination file path"},
                "force": {
                    "type": "boolean",
                    "description": "Allow overwriting existing output",
                    "default": False,
                },
            },
            "required": ["prompt", "source", "output"],
        },
    },
    {
        "name": "enhance_audio",
        "description": (
            "Studio vocal mastering: high-pass rumble filter, presence EQ, de-esser, "
            "broadcast compressor, and target LUFS normalization."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Input audio or video path"},
                "output": {"type": "string", "description": "Destination file path"},
                "profile": {
                    "type": "string",
                    "enum": ["podcast", "warm", "crisp", "radio"],
                    "default": "podcast",
                },
                "target_lufs": {"type": "number", "default": -14.0},
                "force": {"type": "boolean", "default": False},
            },
            "required": ["source", "output"],
        },
    },
    {
        "name": "cut_silence",
        "description": (
            "Automatically detect and trim dead-time, silent gaps, and pauses with "
            "synchronized jump-cuts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Input video or audio path"},
                "output": {"type": "string", "description": "Destination file path"},
                "min_silence_s": {"type": "number", "default": 0.4},
                "noise_db": {"type": "number", "default": -38.0},
                "force": {"type": "boolean", "default": False},
            },
            "required": ["source", "output"],
        },
    },
    {
        "name": "smart_reframe",
        "description": (
            "Intelligent video reframe to vertical 9:16 (or 1:1, 4:5) using subject/face "
            "tracking or split-blur backdrop."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Input widescreen video path"},
                "output": {"type": "string", "description": "Destination vertical video path"},
                "aspect": {"type": "string", "enum": ["9:16", "1:1", "4:5"], "default": "9:16"},
                "mode": {
                    "type": "string",
                    "enum": ["smart", "center", "split"],
                    "default": "smart",
                },
                "force": {"type": "boolean", "default": False},
            },
            "required": ["source", "output"],
        },
    },
    {
        "name": "punch_zoom",
        "description": (
            "Apply dynamic retention punch-in camera zooms (1.1x–1.25x) to boost "
            "audience engagement."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Input video path"},
                "output": {"type": "string", "description": "Destination video path"},
                "auto_interval_s": {"type": "number", "default": 5.0},
                "scale": {"type": "number", "default": 1.15},
                "force": {"type": "boolean", "default": False},
            },
            "required": ["source", "output"],
        },
    },
    {
        "name": "stabilize_video",
        "description": (
            "2-pass camera motion stabilization using VidStab. Neutralizes camera jitter, "
            "compensates for manual zoom-out/zoom-in glitches, and produces steady framing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Input video path"},
                "output": {"type": "string", "description": "Destination video path"},
                "smoothing": {
                    "type": "integer",
                    "default": 20,
                    "description": "Frames for motion smoothing trajectory",
                },
                "shakiness": {
                    "type": "integer",
                    "default": 8,
                    "description": "Detection shakiness 1-10",
                },
                "force": {"type": "boolean", "default": False},
            },
            "required": ["source", "output"],
        },
    },
    {
        "name": "apply_camera_motion",
        "description": (
            "Apply dynamic camera motion to video: kinetic subject tracking "
            "(smooth camera follow), cinematic push-in zoom, pull-out zoom, "
            "smooth directional pans, or handheld organic drift."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Input video path"},
                "output": {"type": "string", "description": "Destination video path"},
                "motion": {
                    "type": "string",
                    "enum": [
                        "track",
                        "follow",
                        "push_in",
                        "slow_zoom_in",
                        "pull_out",
                        "slow_zoom_out",
                        "pan_left",
                        "pan_right",
                        "handheld",
                    ],
                    "default": "track",
                    "description": "Camera motion type",
                },
                "zoom_factor": {
                    "type": "number",
                    "default": 1.15,
                    "description": "Scale crop factor (default 1.15)",
                },
                "smoothing": {
                    "type": "number",
                    "default": 1.5,
                    "description": "Temporal smoothing inertia in seconds (default 1.5)",
                },
                "force": {"type": "boolean", "default": False},
            },
            "required": ["source", "output"],
        },
    },
    {
        "name": "apply_typography",
        "description": (
            "Overlay styled title badges, capsules, and lower-thirds with drop shadow "
            "onto photos or videos."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Input image or video path"},
                "output": {"type": "string", "description": "Destination file path"},
                "text": {"type": "string", "description": "Title or badge text"},
                "position": {
                    "type": "string",
                    "enum": ["top", "center", "bottom", "top-left", "bottom-left"],
                    "default": "bottom",
                },
                "badge": {"type": "boolean", "default": True},
                "force": {"type": "boolean", "default": False},
            },
            "required": ["source", "output", "text"],
        },
    },
    {
        "name": "inpaint_image",
        "description": (
            "Erase unwanted objects, watermarks, or blemishes from photos using "
            "content-aware inpainting."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Input photo path"},
                "output": {"type": "string", "description": "Destination photo path"},
                "bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "[x, y, w, h] bounding box to erase",
                },
                "method": {"type": "string", "enum": ["telea", "ns"], "default": "telea"},
                "force": {"type": "boolean", "default": False},
            },
            "required": ["source", "output", "bbox"],
        },
    },
    {
        "name": "retouch_portrait",
        "description": (
            "Portrait retouching: edge-preserving skin smoothing and depth-of-field "
            "background bokeh."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Input portrait photo path"},
                "output": {"type": "string", "description": "Destination photo path"},
                "skin_strength": {"type": "number", "default": 0.5},
                "depth_blur": {"type": "boolean", "default": False},
                "bokeh_sigma": {"type": "number", "default": 12.0},
                "radiance": {"type": "number", "default": 0.1},
                "force": {"type": "boolean", "default": False},
            },
            "required": ["source", "output"],
        },
    },
    {
        "name": "assemble_clips",
        "description": "Concatenate multiple video clips with smooth xfade transitions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of video clip paths",
                },
                "output": {"type": "string", "description": "Destination path"},
                "transition": {"type": "string", "default": "fade"},
                "duration": {"type": "number", "default": 0.75},
                "aspect": {"type": "string", "default": "16:9"},
                "force": {"type": "boolean", "default": False},
            },
            "required": ["sources", "output"],
        },
    },
    {
        "name": "change_speed",
        "description": "Modify playback speed (slow-mo or timelapse) with pitch-preserved audio.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Input media path"},
                "output": {"type": "string", "description": "Destination path"},
                "speed": {"type": "number", "default": 1.0},
                "force": {"type": "boolean", "default": False},
            },
            "required": ["source", "output", "speed"],
        },
    },
    {
        "name": "add_progress_bar",
        "description": "Add an animated social retention progress bar to a video.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Input video path"},
                "output": {"type": "string", "description": "Destination path"},
                "position": {"type": "string", "enum": ["bottom", "top"], "default": "bottom"},
                "height": {"type": "integer", "default": 6},
                "color": {"type": "string", "default": "yellow"},
                "force": {"type": "boolean", "default": False},
            },
            "required": ["source", "output"],
        },
    },
    {
        "name": "detect_beats",
        "description": "Detect musical beats, tempo (BPM) and rhythm onsets in an audio track.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Input audio/video path"},
            },
            "required": ["source"],
        },
    },
    {
        "name": "generate_narration",
        "description": (
            "Generate local offline text-to-speech voiceover and optionally attach to video."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to speak"},
                "output": {"type": "string", "description": "Destination audio or video path"},
                "voice": {"type": "string", "description": "Voice name (e.g. 'Ioana', 'Daniel')"},
                "rate_wpm": {"type": "integer", "default": 175},
                "video_source": {"type": "string", "description": "Optional video to attach to"},
                "start_s": {"type": "number", "default": 0.0},
                "force": {"type": "boolean", "default": False},
            },
            "required": ["text", "output"],
        },
    },
    {
        "name": "reelify_video",
        "description": (
            "One-click viral transformation of horizontal raw video into high-retention 9:16 "
            "vertical Reel/Short with smart reframing, captions, ducking, and progress bar."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source video file"},
                "output": {"type": "string", "description": "Destination 9:16 video path"},
                "music_track": {
                    "type": "string",
                    "description": "Optional background music audio file",
                },
                "style": {
                    "type": "string",
                    "description": "Subtitle style (tiktok, boxed_yellow, etc.)",
                    "default": "tiktok",
                },
                "profile": {
                    "type": "string",
                    "description": "Audio profile (podcast, radio, etc.)",
                    "default": "podcast",
                },
                "force": {"type": "boolean", "default": False},
            },
            "required": ["source", "output"],
        },
    },
    {
        "name": "omni_export_campaign",
        "description": (
            "Multi-platform omnichannel social campaign exporter. Produces 9:16, 1:1, 16:9, "
            "and 4:5 deliverables from a single master video."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source master video"},
                "output_dir": {
                    "type": "string",
                    "description": "Destination directory for all aspect renders",
                },
                "force": {"type": "boolean", "default": False},
            },
            "required": ["source", "output_dir"],
        },
    },
    {
        "name": "diagnose_and_repair_render",
        "description": (
            "Technical and visual quality assurance auditor (black frames, freezes, clipping, "
            "odd dimensions) with optional auto-fix repair."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Rendered video file to inspect"},
                "frames": {
                    "type": "integer",
                    "default": 6,
                    "description": "Sample frames to inspect",
                },
                "aspect": {
                    "type": "string",
                    "description": "Expected aspect ratio (e.g. 9:16, 16:9)",
                },
                "auto_fix": {
                    "type": "boolean",
                    "default": False,
                    "description": "Automatically repair detected defects",
                },
                "output": {
                    "type": "string",
                    "description": "Repaired video path if auto_fix is true",
                },
                "force": {"type": "boolean", "default": False},
            },
            "required": ["source"],
        },
    },
    {
        "name": "apply_brand_watermark",
        "description": (
            "Overlay brand logo or watermark with custom opacity, position, and scale percentage."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source video file"},
                "logo": {"type": "string", "description": "Logo image path (PNG recommended)"},
                "output": {"type": "string", "description": "Output video file path"},
                "position": {
                    "type": "string",
                    "default": "top-right",
                    "description": (
                        "Corner position (top-left, top-right, bottom-left, bottom-right)"
                    ),
                },
                "opacity": {
                    "type": "number",
                    "default": 0.85,
                    "description": "Logo opacity between 0.05 and 1.0",
                },
                "scale_pct": {
                    "type": "number",
                    "default": 12.0,
                    "description": "Logo width as percentage of video width",
                },
                "force": {"type": "boolean", "default": False},
            },
            "required": ["source", "logo", "output"],
        },
    },
    {
        "name": "extract_viral_hook",
        "description": (
            "Detect acoustic energy peaks and extract the most engaging 30-60 second "
            "hook clip from long-form video or audio."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Input long-form video or audio"},
                "output": {"type": "string", "description": "Destination extracted hook file"},
                "duration": {
                    "type": "number",
                    "default": 30.0,
                    "description": "Target hook duration in seconds",
                },
                "fade": {
                    "type": "number",
                    "default": 0.15,
                    "description": "Audio fade duration in seconds",
                },
                "force": {"type": "boolean", "default": False},
            },
            "required": ["source", "output"],
        },
    },
    {
        "name": "scout_media_library",
        "description": (
            "Catalog and search local asset library for B-Roll cutaways, background plates, or SFX."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "directories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Directories to search (default: in/)",
                },
                "type": {
                    "type": "string",
                    "description": "Media type (video, audio, image)",
                },
                "category": {
                    "type": "string",
                    "description": "Category (sfx, plate_video, broll, plate_still)",
                },
                "query": {"type": "string", "description": "Search keyword"},
                "vertical": {
                    "type": "boolean",
                    "default": False,
                    "description": "Only vertical 9:16 media",
                },
                "horizontal": {
                    "type": "boolean",
                    "default": False,
                    "description": "Only horizontal 16:9 media",
                },
                "min_duration": {
                    "type": "number",
                    "description": "Minimum duration in seconds",
                },
                "max_duration": {
                    "type": "number",
                    "description": "Maximum duration in seconds",
                },
            },
        },
    },
]


def dispatch_tool(name: str, arguments: dict[str, Any], config: Config) -> dict[str, Any]:
    """Execute requested tool and format result for MCP."""
    runner = KinoRunner.from_config(config)
    ml_runner = MlRunner.from_config(config)

    if name == "inspect_media":
        report = inspect_media(arguments["path"], config)
        return {"report": report.to_dict()}

    if name == "execute_prompt":
        res = execute_prompt(
            arguments["prompt"],
            arguments["source"],
            arguments["output"],
            config,
            runner,
            ml_runner,
            force=arguments.get("force", False),
        )
        return {
            "output": str(res.output),
            "steps_executed": list(res.steps_executed),
        }

    if name == "enhance_audio":
        res_a = enhance_audio(
            arguments["source"],
            arguments["output"],
            config,
            profile=arguments.get("profile", "podcast"),
            target_lufs=float(arguments.get("target_lufs", -14.0)),
            force=arguments.get("force", False),
        )
        return {
            "output": str(res_a.output),
            "profile": res_a.profile,
            "target_lufs": res_a.target_lufs,
        }

    if name == "cut_silence":
        res_s = trim_silence(
            arguments["source"],
            arguments["output"],
            config,
            min_silence_s=float(arguments.get("min_silence_s", 0.4)),
            noise_db=float(arguments.get("noise_db", -38.0)),
            force=arguments.get("force", False),
        )
        return {
            "output": str(res_s.output),
            "new_duration_s": res_s.new_duration_s,
            "removed_duration_s": res_s.removed_duration_s,
            "cuts_count": res_s.cuts_count,
        }

    if name == "smart_reframe":
        res_r = smart_reframe(
            arguments["source"],
            arguments["output"],
            config,
            target_aspect=arguments.get("aspect", "9:16"),
            mode=arguments.get("mode", "smart"),
            force=arguments.get("force", False),
        )
        return {"output": str(res_r.output), "aspect": res_r.target_aspect, "mode": res_r.mode}

    if name == "punch_zoom":
        res_z = punch_zoom(
            arguments["source"],
            arguments["output"],
            config,
            auto_interval_s=float(arguments.get("auto_interval_s", 5.0)),
            auto_scale=float(arguments.get("scale", 1.15)),
            force=arguments.get("force", False),
        )
        return {"output": str(res_z.output), "cues_applied": res_z.cues_applied}

    if name == "apply_typography":
        style = TypographyStyle(
            position=arguments.get("position", "bottom"),
            badge=arguments.get("badge", True),
        )
        res_t = apply_typography(
            arguments["source"],
            arguments["output"],
            arguments["text"],
            config,
            style=style,
            force=arguments.get("force", False),
        )
        return {"output": str(res_t.output), "lines_rendered": res_t.lines_rendered}

    if name == "inpaint_image":
        raw_box = arguments["bbox"]
        bbox = (int(raw_box[0]), int(raw_box[1]), int(raw_box[2]), int(raw_box[3]))
        res_i = inpaint_image(
            arguments["source"],
            arguments["output"],
            config,
            bbox=bbox,
            method=arguments.get("method", "telea"),
            force=arguments.get("force", False),
        )
        return {"output": str(res_i.output), "erased_pixels": res_i.erased_pixels}

    if name == "retouch_portrait":
        res_p = retouch_portrait(
            arguments["source"],
            arguments["output"],
            config,
            smooth_skin=True,
            skin_strength=float(arguments.get("skin_strength", 0.5)),
            depth_blur=arguments.get("depth_blur", False),
            blur_sigma=float(arguments.get("bokeh_sigma", 12.0)),
            radiance=float(arguments.get("radiance", 0.1)),
            force=arguments.get("force", False),
        )
        return {"output": str(res_p.output), "skin_smoothed": res_p.skin_smoothed}

    if name == "assemble_clips":
        res_ass = assemble_clips(
            arguments["sources"],
            arguments["output"],
            config,
            transition=arguments.get("transition", "fade"),
            transition_duration_s=float(arguments.get("duration", 0.75)),
            aspect=arguments.get("aspect", "16:9"),
            force=arguments.get("force", False),
        )
        return {
            "output": str(res_ass.output),
            "clips_count": res_ass.clips_count,
            "transition": res_ass.transition,
            "duration_s": res_ass.total_duration_s,
        }

    if name == "change_speed":
        res_spd = change_speed(
            arguments["source"],
            arguments["output"],
            config,
            speed=float(arguments.get("speed", 1.0)),
            force=arguments.get("force", False),
        )
        return {
            "output": str(res_spd.output),
            "speed_factor": res_spd.speed_factor,
            "duration_s": res_spd.new_duration_s,
        }

    if name == "add_progress_bar":
        res_pb = add_progress_bar(
            arguments["source"],
            arguments["output"],
            config,
            position=arguments.get("position", "bottom"),
            height=int(arguments.get("height", 6)),
            color=arguments.get("color", "yellow"),
            force=arguments.get("force", False),
        )
        return {
            "output": str(res_pb.output),
            "position": res_pb.position,
            "height": res_pb.height,
            "color": res_pb.color,
        }

    if name == "detect_beats":
        res_bt = detect_beats(arguments["source"], config)
        return {
            "total_beats": res_bt.total_beats,
            "estimated_bpm": res_bt.estimated_bpm,
            "beats_s": list(res_bt.beats_s),
            "duration_s": res_bt.duration_s,
        }

    if name == "generate_narration":
        res_nr = generate_narration(
            arguments["text"],
            arguments["output"],
            config,
            voice=arguments.get("voice"),
            rate_wpm=int(arguments.get("rate_wpm", 175)),
            video_source=arguments.get("video_source"),
            start_s=float(arguments.get("start_s", 0.0)),
            force=arguments.get("force", False),
        )
        return {
            "output": str(res_nr.output),
            "voice": res_nr.voice,
            "duration_s": res_nr.duration_s,
            "text": res_nr.text,
        }

    if name == "stabilize_video":
        from .recipes.stabilize import stabilize_video

        res_st = stabilize_video(
            arguments["source"],
            arguments["output"],
            config,
            smoothing=int(arguments.get("smoothing", 20)),
            shakiness=int(arguments.get("shakiness", 8)),
            force=bool(arguments.get("force", False)),
        )
        return {
            "output": str(res_st.output),
            "smoothing": res_st.smoothing,
            "media": {
                "duration_s": res_st.media.duration_s,
                "width": res_st.media.width,
                "height": res_st.media.height,
            },
        }

    if name == "apply_camera_motion":
        from .recipes.camera_motion import apply_camera_motion

        res_cm = apply_camera_motion(
            arguments["source"],
            arguments["output"],
            config,
            motion=str(arguments.get("motion", "track")),
            zoom_factor=float(arguments.get("zoom_factor", 1.15)),
            smoothing=float(arguments.get("smoothing", 1.5)),
            force=bool(arguments.get("force", False)),
        )
        return {
            "output": str(res_cm.output),
            "motion": res_cm.motion,
            "media": {
                "duration_s": res_cm.media.duration_s,
                "width": res_cm.media.width,
                "height": res_cm.media.height,
            },
        }

    if name == "reelify_video":
        res_rf = reelify(
            arguments["source"],
            arguments["output"],
            config,
            ml_runner,
            subtitles_style=arguments.get("style", "tiktok"),
            force=bool(arguments.get("force", False)),
        )
        return {
            "output": str(res_rf.output),
            "media": {
                "duration_s": res_rf.media.duration_s,
                "width": res_rf.media.width,
                "height": res_rf.media.height,
            },
            "stages_executed": list(res_rf.stages_executed),
        }

    if name == "omni_export_campaign":
        res_om = omni_export(
            arguments["source"],
            arguments["output_dir"],
            config,
            force=bool(arguments.get("force", False)),
        )
        return {
            "output_dir": str(res_om.output_dir),
            "deliverables": {d.aspect: str(d.path) for d in res_om.deliverables},
        }

    if name == "diagnose_and_repair_render":
        doc_report = diagnose_render(
            arguments["source"],
            config,
            sample_frames=int(arguments.get("frames", 6)),
            expected_aspect=arguments.get("aspect"),
        )
        if arguments.get("auto_fix", False) and arguments.get("output"):
            res_fix = auto_fix_render(
                arguments["source"],
                arguments["output"],
                doc_report,
                config,
                force=bool(arguments.get("force", False)),
            )
            return {
                "fixed_output": str(res_fix.output),
                "fixes_applied": list(res_fix.fixes_applied),
                "initial_report": doc_report.to_dict(),
            }
        return {"report": doc_report.to_dict()}

    if name == "apply_brand_watermark":
        res_br = apply_brand_watermark(
            arguments["source"],
            arguments["logo"],
            arguments["output"],
            config,
            position=arguments.get("position", "top-right"),
            opacity=float(arguments.get("opacity", 0.85)),
            scale_pct=float(arguments.get("scale_pct", 12.0)),
            force=bool(arguments.get("force", False)),
        )
        return {
            "output": str(res_br.output),
            "position": res_br.position,
            "opacity": res_br.opacity,
        }

    if name == "extract_viral_hook":
        res_hk = extract_viral_hook(
            arguments["source"],
            arguments["output"],
            config,
            target_duration_s=float(arguments.get("duration", 30.0)),
            fade_s=float(arguments.get("fade", 0.15)),
            force=bool(arguments.get("force", False)),
        )
        return {
            "output": str(res_hk.output),
            "start_s": res_hk.hook.start_s,
            "end_s": res_hk.hook.end_s,
            "score": res_hk.hook.score,
        }

    if name == "scout_media_library":
        raw_dirs = arguments.get("directories")
        dirs = [Path(d) for d in raw_dirs] if raw_dirs else [config.in_dir]
        orientation = (
            "vertical"
            if arguments.get("vertical")
            else ("horizontal" if arguments.get("horizontal") else None)
        )
        assets = scout_library(
            directories=dirs,
            config=config,
            media_type=arguments.get("type"),
            category=arguments.get("category"),
            query=arguments.get("query"),
            orientation=orientation,
            min_duration=arguments.get("min_duration"),
            max_duration=arguments.get("max_duration"),
        )
        return {"assets": [a.to_dict() for a in assets], "count": len(assets)}

    raise MediaLabError(f"unknown tool {name!r}")


def handle_rpc_request(msg: dict[str, Any], config: Config) -> dict[str, Any] | None:
    """Process a single JSON-RPC 2.0 message."""
    method = msg.get("method")
    msg_id = msg.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        }

    if method == "notifications/initialized":
        return None

    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = msg.get("params", {})
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        try:
            result_data = dispatch_tool(tool_name, tool_args, config)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result_data, indent=2)}],
                    "isError": False,
                },
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error executing {tool_name}: {exc}"}],
                    "isError": True,
                },
            }

    else:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def run_mcp_server(config: Config | None = None) -> None:
    """Run the MCP stdio loop."""
    cfg = config or load_config()
    for line in sys.stdin:
        line_clean = line.strip()
        if not line_clean:
            continue
        try:
            req = json.loads(line_clean)
        except json.JSONDecodeError:
            continue

        resp = handle_rpc_request(req, cfg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
