"""Tests for Model Context Protocol (MCP) server implementation."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from media_lab.config import Config
from media_lab.mcp_server import (
    PROTOCOL_VERSION,
    SERVER_INFO,
    TOOLS,
    handle_rpc_request,
    run_mcp_server,
)


def _generate_synthetic_video(path: Path, config: Config, duration: float = 1.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(config.ffmpeg),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size=320x240:rate=25",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=500:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )
    return path


def _generate_synthetic_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    img.save(path)
    return path


def test_mcp_initialize(config: Config) -> None:
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    resp = handle_rpc_request(req, config)

    assert resp is not None
    assert resp["id"] == 1
    assert resp["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert resp["result"]["serverInfo"] == SERVER_INFO


def test_mcp_ping_and_notifications(config: Config) -> None:
    ping_req = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
    ping_resp = handle_rpc_request(ping_req, config)
    assert ping_resp == {"jsonrpc": "2.0", "id": 2, "result": {}}

    notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    assert handle_rpc_request(notif, config) is None


def test_mcp_tools_list(config: Config) -> None:
    req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
    resp = handle_rpc_request(req, config)

    assert resp is not None
    tools = resp["result"]["tools"]
    assert len(tools) == len(TOOLS)
    tool_names = {t["name"] for t in tools}
    assert "inspect_media" in tool_names
    assert "execute_prompt" in tool_names
    assert "enhance_audio" in tool_names
    assert "smart_reframe" in tool_names
    assert "punch_zoom" in tool_names
    assert "inpaint_image" in tool_names


def test_mcp_tools_call_inspect(config: Config) -> None:
    src = _generate_synthetic_image(config.in_dir / "mcp_img.png")
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "inspect_media",
            "arguments": {"path": str(src)},
        },
    }
    resp = handle_rpc_request(req, config)

    assert resp is not None
    assert resp["result"]["isError"] is False
    content_text = resp["result"]["content"][0]["text"]
    data = json.loads(content_text)
    assert "report" in data
    assert data["report"]["width"] == 200


def test_mcp_tools_call_enhance_audio(config: Config) -> None:
    src = _generate_synthetic_video(config.in_dir / "mcp_vid.mp4", config)
    out = config.out_dir / "mcp_enhanced.mp4"
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "enhance_audio",
            "arguments": {"source": str(src), "output": str(out), "profile": "warm", "force": True},
        },
    }
    resp = handle_rpc_request(req, config)

    assert resp is not None
    assert resp["result"]["isError"] is False
    assert out.is_file()


def test_mcp_tools_call_execute_prompt(config: Config) -> None:
    src = _generate_synthetic_video(config.in_dir / "mcp_prompt_src.mp4", config)
    out = config.out_dir / "mcp_prompt_out.mp4"
    req = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "execute_prompt",
            "arguments": {
                "prompt": "Fa un clip vertical cu look cald",
                "source": str(src),
                "output": str(out),
                "force": True,
            },
        },
    }
    resp = handle_rpc_request(req, config)

    assert resp is not None
    assert resp["result"]["isError"] is False
    assert out.is_file()


def test_mcp_unknown_method_and_error(config: Config) -> None:
    # Unknown method
    req_bad = {"jsonrpc": "2.0", "id": 7, "method": "unknown/method"}
    resp_bad = handle_rpc_request(req_bad, config)
    assert resp_bad is not None
    assert resp_bad["error"]["code"] == -32601

    # Tool error handled gracefully
    req_err = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {"name": "inspect_media", "arguments": {"path": "non_existent.mp4"}},
    }
    resp_err = handle_rpc_request(req_err, config)
    assert resp_err is not None
    assert resp_err["result"]["isError"] is True


def test_run_mcp_server_loop(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    input_stream = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n")
    output_stream = io.StringIO()

    monkeypatch.setattr("sys.stdin", input_stream)
    monkeypatch.setattr("sys.stdout", output_stream)

    run_mcp_server(config)

    output_lines = [
        json.loads(line) for line in output_stream.getvalue().splitlines() if line.strip()
    ]
    assert len(output_lines) == 1
    assert output_lines[0]["id"] == 1
    assert output_lines[0]["result"] == {}
