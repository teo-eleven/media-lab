"""Tests for the local Media Lab Studio Web server."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from media_lab.config import Config
from media_lab.ffmpeg import run_ffmpeg
from media_lab.kino import KinoRunner
from media_lab.ml_runner import MlRunner
from media_lab.server import create_server


def _generate_synthetic_video(path: Path, config: Config) -> Path:
    cmd = [
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=160x120:r=25:d=1.0",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1.0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    run_ffmpeg(cmd, config)
    return path


def test_studio_server_endpoints(config: Config) -> None:
    src_file = _generate_synthetic_video(config.in_dir / "test_server_src.mp4", config)

    runner = KinoRunner.from_config(config)
    ml_runner = MlRunner.from_config(config)
    server = create_server(config, runner, ml_runner, host="127.0.0.1", port=0)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. Test GET / (HTML UI)
        with urllib.request.urlopen(f"{base_url}/") as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "<!DOCTYPE html>" in html
            assert "Media Lab Studio" in html

        # 2. Test GET /api/files (JSON file tree)
        with urllib.request.urlopen(f"{base_url}/api/files") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "in" in data
            assert "out" in data
            assert any(f["name"] == "test_server_src.mp4" for f in data["in"])

        # 3. Test GET /media/in/test_server_src.mp4 (media streaming with Range)
        req = urllib.request.Request(f"{base_url}/media/in/test_server_src.mp4")
        req.add_header("Range", "bytes=0-100")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 206
            assert resp.headers.get("Content-Range") is not None
            chunk = resp.read()
            assert len(chunk) == 101

        # 4. Test POST /api/prompt conversational (greeting)
        post_chat = json.dumps({"prompt": "salut"}).encode("utf-8")
        req_chat = urllib.request.Request(
            f"{base_url}/api/prompt",
            data=post_chat,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_chat) as resp:
            assert resp.status == 200
            res_chat = json.loads(resp.read().decode("utf-8"))
            assert res_chat["intent"] == "greeting"
            assert res_chat["executable"] is False
            assert len(res_chat["suggested_prompts"]) > 0

        # 4b. Test POST /api/prompt plan mode
        post_data = json.dumps(
            {
                "prompt": "Fa un short vertical cu subtitrari galbene",
                "input": f"in/{src_file.name}",
                "execute": False,
            }
        ).encode("utf-8")
        req_post = urllib.request.Request(
            f"{base_url}/api/prompt",
            data=post_data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_post) as resp:
            assert resp.status == 200
            res_data = json.loads(resp.read().decode("utf-8"))
            assert "operations" in res_data
            assert len(res_data["operations"]) > 0
            assert res_data["intent"] == "plan"
            assert res_data["executable"] is True

        # 4c. Test POST /api/prompt execute mode
        post_exec = json.dumps(
            {
                "prompt": "Conversie rapida",
                "input": f"in/{src_file.name}",
                "execute": True,
            }
        ).encode("utf-8")
        req_exec = urllib.request.Request(
            f"{base_url}/api/prompt",
            data=post_exec,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_exec) as resp:
            assert resp.status == 200
            res_exec = json.loads(resp.read().decode("utf-8"))
            assert res_exec["status"] == "ok"
            assert "output" in res_exec
            assert (config.out_dir / res_exec["output"]).is_file()

        # 5. Security Test: Path traversal defense
        try:
            req_trav = urllib.request.Request(f"{base_url}/media/in/../../etc/passwd")
            urllib.request.urlopen(req_trav)
            pytest.fail("Should have failed path traversal")
        except urllib.error.HTTPError as err:
            assert err.code in (400, 403, 404)

        # 6. Security Test: RFC 7233 Range Not Satisfiable (416)
        try:
            req_oor = urllib.request.Request(f"{base_url}/media/in/test_server_src.mp4")
            req_oor.add_header("Range", "bytes=999999-1000000")
            urllib.request.urlopen(req_oor)
            pytest.fail("Should have failed with 416")
        except urllib.error.HTTPError as err:
            assert err.code == 416

        # 8. Test GET /api/session
        with urllib.request.urlopen(f"{base_url}/api/session") as resp:
            assert resp.status == 200
            s_data = json.loads(resp.read().decode("utf-8"))
            assert "has_previous_spec" in s_data
            assert "provider" in s_data

        # 9. Test GET /api/settings and POST /api/settings
        with urllib.request.urlopen(f"{base_url}/api/settings") as resp:
            assert resp.status == 200
            cfg_data = json.loads(resp.read().decode("utf-8"))
            assert "provider" in cfg_data
            assert "ollama_url" in cfg_data

        post_set = json.dumps({"provider": "local"}).encode("utf-8")
        req_set = urllib.request.Request(
            f"{base_url}/api/settings",
            data=post_set,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_set) as resp:
            assert resp.status == 200
            upd_data = json.loads(resp.read().decode("utf-8"))
            assert upd_data["status"] == "ok"
            assert upd_data["settings"]["provider"] == "local"

        # 10. Test Continuity & Camera motion handling in chat
        post_iter = json.dumps(
            {
                "prompt": (
                    "este exact la fel ca înainte, scoate acel zoom de pe persoana, "
                    "fiindcă in faza inițială când am filmat, am dat zoom out când "
                    "se apropia de mine si in videoclipul final se vede prost... poți face asta?"
                ),
                "input": f"in/{src_file.name}",
                "execute": False,
            }
        ).encode("utf-8")
        req_iter = urllib.request.Request(
            f"{base_url}/api/prompt",
            data=post_iter,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_iter) as resp:
            assert resp.status == 200
            res_iter = json.loads(resp.read().decode("utf-8"))
            assert res_iter["executable"] is True
            # Check stabilization was recognized
            ops_str = " ".join(res_iter["operations"]).lower()
            assert "stabiliz" in ops_str or "vidstab" in ops_str

        # 11. Test POST /api/session/reset
        req_rst = urllib.request.Request(
            f"{base_url}/api/session/reset",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req_rst) as resp:
            assert resp.status == 200
            rst_res = json.loads(resp.read().decode("utf-8"))
            assert rst_res["status"] == "ok"

    finally:
        server.shutdown()
        server.server_close()
