"""Native Model Context Protocol (MCP) server for media-lab.

Provides a standard stdio JSON-RPC 2.0 interface exposing all media-lab tools
directly to AI agents (Antigravity, Cursor, Claude Desktop, etc.).
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .config import Config, load_config
from .errors import MediaLabError
from .inspect import inspect_media
from .kino import KinoRunner
from .ml_runner import MlRunner
from .prompt_agent import execute_prompt
from .recipes.assembly import assemble_clips
from .recipes.audio_enhance import enhance_audio
from .recipes.beat_sync import detect_beats
from .recipes.face_retouch import retouch_portrait
from .recipes.inpainting import inpaint_image
from .recipes.narrator import generate_narration
from .recipes.progress_bar import add_progress_bar
from .recipes.punch_zoom import punch_zoom
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
