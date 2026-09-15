"""Local Web Studio server with interactive Chat UI and HTML5 media player.

Runs an offline, pure-Python HTTP server providing a dark-mode studio interface
for chatting with the Media Lab agent, previewing media files with HTTP Range
streaming, and executing automated editing pipelines directly from the browser.
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import Config
from .kino import KinoRunner
from .ml_runner import MlRunner
from .prompt_agent import chat_agent, execute_prompt

STATIC_DIR: Path = Path(__file__).parent / "static"


def get_studio_html() -> str:
    """Load the Studio single page application HTML."""
    html_file = STATIC_DIR / "index.html"
    if html_file.is_file():
        return html_file.read_text(encoding="utf-8")
    return "<h1>Media Lab Studio</h1><p>index.html not found</p>"


class StudioRequestHandler(SimpleHTTPRequestHandler):
    """HTTP Request handler for Media Lab Studio with range requests and API."""

    config: Config
    runner: KinoRunner
    ml_runner: MlRunner

    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path in {"/", "/index.html"}:
            self._send_html_response(get_studio_html())
            return

        if path == "/api/files":
            self._handle_api_files()
            return

        if path.startswith("/media/"):
            self._handle_media_stream(path[len("/media/") :])
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/prompt":
            self._handle_api_prompt()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _send_html_response(self, content: str) -> None:
        data = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json_response(self, data: Any, status: int = 200) -> None:
        content = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _handle_api_files(self) -> None:
        """List files in in/, out/, work/ directories."""
        result: dict[str, list[dict[str, Any]]] = {"in": [], "out": [], "work": []}

        for folder_name, folder_path in [
            ("in", self.config.in_dir),
            ("out", self.config.out_dir),
            ("work", self.config.work_dir),
        ]:
            if not folder_path.exists():
                continue
            sorted_entries = sorted(
                folder_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
            )
            for entry in sorted_entries:
                if entry.is_file() and not entry.name.startswith("."):
                    size_bytes = entry.stat().st_size
                    human_size = (
                        f"{size_bytes / (1024 * 1024):.1f} MB"
                        if size_bytes >= 1024 * 1024
                        else f"{size_bytes / 1024:.0f} KB"
                    )
                    suffix = entry.suffix.lower()
                    mtype = "other"
                    if suffix in {".mp4", ".mov", ".webm", ".mkv", ".m4v"}:
                        mtype = "video"
                    elif suffix in {".mp3", ".wav", ".aac", ".m4a", ".flac"}:
                        mtype = "audio"
                    elif suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                        mtype = "image"

                    result[folder_name].append(
                        {
                            "name": entry.name,
                            "type": mtype,
                            "size_bytes": size_bytes,
                            "size_human": human_size,
                        }
                    )

        self._send_json_response(result)

    def _handle_media_stream(self, subpath: str) -> None:
        """Serve media files supporting HTTP 206 Partial Content for range requests."""
        parts = subpath.split("/", 1)
        if len(parts) != 2:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid media path format")
            return

        folder, filename = parts[0], urllib.parse.unquote(parts[1])
        if folder not in {"in", "out", "work"}:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid media folder")
            return

        base_dir = (
            self.config.in_dir
            if folder == "in"
            else (self.config.out_dir if folder == "out" else self.config.work_dir)
        )
        file_path = (base_dir / filename).resolve()

        # Prevent directory traversal
        if not file_path.is_relative_to(base_dir.resolve()):
            self.send_error(HTTPStatus.FORBIDDEN, "Access denied")
            return
        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        file_size = file_path.stat().st_size
        content_type, _ = mimetypes.guess_type(str(file_path))
        if not content_type:
            content_type = "application/octet-stream"

        range_header = self.headers.get("Range")
        if range_header:
            match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else file_size - 1
                if file_size == 0 or start >= file_size or start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.end_headers()
                    return

                end = min(end, file_size - 1)
                length = end - start + 1

                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                try:
                    with file_path.open("rb") as f:
                        f.seek(start)
                        remaining = length
                        while remaining > 0:
                            chunk = f.read(min(65536, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        try:
            with file_path.open("rb") as f:
                while chunk := f.read(65536):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_api_prompt(self) -> None:
        """Handle prompt analysis, conversational assistance, and execution from UI."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        try:
            payload = json.loads(post_data.decode("utf-8"))
        except Exception as err:
            self._send_json_response({"error": f"Invalid JSON payload: {err}"}, status=400)
            return

        prompt_str = payload.get("prompt", "").strip()
        input_str = payload.get("input")
        execute = bool(payload.get("execute", False))

        if not prompt_str:
            self._send_json_response({"error": "Prompt cannot be empty"}, status=400)
            return

        # Resolve input_str if provided, or fallback to first media file in in/, then out/
        src_path: Path | None = None
        if input_str:
            p = Path(input_str)
            src_path = (
                (self.config.in_dir.parent / p).resolve() if not p.is_absolute() else p.resolve()
            )
        else:
            for entry in sorted(self.config.in_dir.iterdir()):
                if (
                    entry.is_file()
                    and not entry.name.startswith(".")
                    and entry.suffix.lower()
                    in {".mp4", ".mov", ".m4a", ".wav", ".jpg", ".png", ".jpeg"}
                ):
                    src_path = entry.resolve()
                    break
            if src_path is None and self.config.out_dir.exists():
                for entry in sorted(
                    self.config.out_dir.iterdir(),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True,
                ):
                    if (
                        entry.is_file()
                        and not entry.name.startswith(".")
                        and entry.suffix.lower() in {".mp4", ".mov", ".jpg", ".png"}
                    ):
                        src_path = entry.resolve()
                        break

        if src_path is not None:
            valid_roots = (
                self.config.in_dir.resolve(),
                self.config.out_dir.resolve(),
                self.config.work_dir.resolve(),
            )
            if not any(src_path.is_relative_to(root) for root in valid_roots):
                self._send_json_response(
                    {"error": "Input path must reside within in/, out/, or work/"}, status=403
                )
                return

        chat_res = chat_agent(prompt_str, src_path, self.config)
        is_informational = chat_res.intent in ("greeting", "help", "inspect")

        if not execute or is_informational:
            self._send_json_response(
                {
                    "status": "ok",
                    "reply": chat_res.reply,
                    "intent": chat_res.intent,
                    "operations": list(chat_res.plan_operations),
                    "suggested_prompts": list(chat_res.suggested_prompts),
                    "executable": chat_res.executable,
                    "prompt": prompt_str,
                    "input": str(src_path.name) if src_path else None,
                }
            )
            return

        if not src_path:
            self._send_json_response(
                {"error": "No media file selected and 'in/' directory is empty"}, status=400
            )
            return

        stem = src_path.stem
        if stem.startswith("studio_render_"):
            stem = stem[len("studio_render_") :]
        stem = re.sub(r"_v\d+$", "", stem)

        candidate = self.config.out_dir / f"studio_render_{stem}.mp4"
        counter = 1
        while candidate.exists() and candidate.resolve() == src_path.resolve():
            counter += 1
            candidate = self.config.out_dir / f"studio_render_{stem}_v{counter}.mp4"
        out_path = candidate

        try:
            result = execute_prompt(
                prompt_str,
                src_path,
                out_path,
                self.config,
                self.runner,
                self.ml_runner,
                force=True,
            )
            skipped_audio = [s for s in result.steps_executed if "skipped" in s]
            note = ""
            if skipped_audio:
                note = (
                    "\n\n> ℹ️ **Notă:** Fișierul sursă nu are pistă audio. "
                    "Etapele vocale au fost omise, iar efectele video au fost aplicate."
                )
            self._send_json_response(
                {
                    "status": "ok",
                    "reply": (
                        f"🎉 **Randare completată cu succes!**\n\n"
                        f"Am aplicat modificările cerute direct pe `{src_path.name}`.\n"
                        f"Noul fișier salvat: `{result.output.name}` "
                        f"({len(result.steps_executed)} etape executate).{note}"
                    ),
                    "output": str(result.output.name),
                    "steps": list(result.steps_executed),
                }
            )
        except Exception as exc:
            self._send_json_response({"error": str(exc)}, status=500)


def create_server(
    config: Config,
    runner: KinoRunner,
    ml_runner: MlRunner,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create a configured ThreadingHTTPServer instance for Media Lab Studio."""
    handler_cls = type(
        "ConfiguredStudioHandler",
        (StudioRequestHandler,),
        {"config": config, "runner": runner, "ml_runner": ml_runner},
    )
    return ThreadingHTTPServer((host, port), handler_cls)


def run_studio(
    config: Config,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Start local Studio web interface and optionally open in default browser."""
    runner = KinoRunner.from_config(config)
    ml_runner = MlRunner.from_config(config)
    server = create_server(config, runner, ml_runner, host=host, port=port)

    url = f"http://{host}:{port}"
    print(f"🎬 Media Lab Studio running at: {url}")
    print("Press Ctrl+C to stop the studio server.")

    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Media Lab Studio server.")
    finally:
        server.server_close()
