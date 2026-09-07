from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

from research_assistant.config import REPO_ROOT
from research_assistant.ui.settings import SECRETS, load_settings, save_settings, schema, validate
from research_assistant.ui.worker import atomic_json

STATIC = Path(__file__).parent / "static"


class Application:
    def __init__(self, root=REPO_ROOT):
        self.root = root
        self.runs = root / "results" / "ui"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.process = None
        self.active = None

    def settings(self):
        env, configs = load_settings(self.root / ".env")
        return dict(schema=schema(), env={k: v for k, v in env.items() if k not in SECRETS},
                    secrets={k: bool(env[k]) for k in SECRETS}, configs=configs)

    def merge(self, data):
        env, configs = load_settings(self.root / ".env")
        submitted = data.get("env", {})
        if not isinstance(submitted, dict) or not isinstance(data.get("configs", {}), dict):
            raise ValueError("Configuration must be an object")
        for key, value in submitted.items():
            if key in SECRETS and value == "":
                continue  # Blank password inputs preserve existing secrets.
            env[key] = value
        for key in data.get("clear_secrets", []):
            if key not in SECRETS:
                raise ValueError("Invalid API key")
            env[key] = ""
        for stage, values in data.get("configs", {}).items():
            configs[stage] = values
        validate(env, configs)
        return env, configs

    def start(self, data):
        with self.lock:
            if self.process and self.process.poll() is None:
                raise ValueError("A pipeline is already running. Wait for it to finish or stop the current run.")
            topic = data.get("topic", "")
            if not isinstance(topic, str) or not 1 <= len(topic.strip()) <= 2000:
                raise ValueError("Enter a topic between 1 and 2,000 characters.")
            env, configs = self.merge(data)
            validate(env, configs, preflight=True)
            run_id = uuid4().hex
            directory = self.runs / run_id
            directory.mkdir(mode=0o700)
            atomic_json(directory / "status.json", dict(status="running", topic=topic.strip(),
                        stages=["pending"] * 4, summaries=[""] * 4, warnings=[], files=["run.log"], error=None))
            # Config snapshot excludes credentials. Secrets are passed through stdin only.
            atomic_json(directory / "config.json", dict(topic=topic.strip(), configs=configs))
            with (directory / "run.log").open("a", encoding="utf-8") as log_stream:
                process = subprocess.Popen([sys.executable, "-m", "research_assistant.ui.worker", str(directory)],
                                           cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=log_stream,
                                           stderr=subprocess.STDOUT, text=True, start_new_session=True)
            self.process, self.active = process, run_id
            try:
                process.stdin.write(json.dumps(dict(topic=topic.strip(), env=env, configs=configs)))
                process.stdin.close()
            except (BrokenPipeError, OSError):
                process.wait(timeout=5)
            return dict(run_id=run_id)

    def state(self, run_id):
        with self.lock:
            directory = self.directory(run_id)
            state = json.loads((directory / "status.json").read_text(encoding="utf-8"))
            alive = run_id == self.active and self.process and self.process.poll() is None
            if state["status"] == "running" and not alive:
                exit_code = self.process.poll() if run_id == self.active and self.process else None
                self.append_log(run_id, f"UI server detected that the worker stopped unexpectedly | exit_code={exit_code}")
                state["status"] = "failed"
                state["error"] = "The process stopped unexpectedly or the application restarted. You can start a new run using the saved configuration."
                state["stages"] = ["failed" if s == "running" else "skipped" if s == "pending" else s for s in state["stages"]]
                atomic_json(directory / "status.json", state)
            state["run_id"] = run_id
            return state

    def directory(self, run_id):
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise ValueError("Invalid run")
        directory = self.runs / run_id
        if not directory.is_dir():
            raise ValueError("Run not found")
        return directory

    def log_tail(self, run_id, max_bytes=200_000):
        """Read a bounded log tail so live polling does not transfer an unbounded file."""
        path = self.directory(run_id) / "run.log"
        if not path.is_file():
            return ""
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes))
            data = stream.read()
        text = data.decode("utf-8", errors="replace")
        if size > max_bytes:
            text = "[Earlier log entries omitted from live view. Download run.log for the complete log.]\n" + text.partition("\n")[2]
        return text

    def append_log(self, run_id, message):
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with (self.directory(run_id) / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{timestamp} | INFO     | research_assistant.ui.server | {message}\n")

    def cancel(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.append_log(self.active, "Pipeline cancellation requested by the user")
                if os.name == "posix":
                    os.killpg(self.process.pid, signal.SIGTERM)
                else:
                    self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(self.process.pid, signal.SIGKILL)
                    else:
                        self.process.kill()
                    self.process.wait()
                path = self.directory(self.active) / "status.json"
                state = json.loads(path.read_text(encoding="utf-8"))
                state.update(status="cancelled", error="The user stopped the pipeline.")
                state["stages"] = ["cancelled" if s == "running" else "skipped" if s == "pending" else s for s in state["stages"]]
                atomic_json(path, state)
            return {"ok": True}


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value, status=200, content_type="application/json; charset=utf-8"):
            body = json.dumps(value, ensure_ascii=False).encode() if isinstance(value, (dict, list)) else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            host = f"127.0.0.1:{self.server.server_port}"
            return (self.headers.get("Host") == host and
                    self.headers.get("Origin", f"http://{host}") == f"http://{host}" and
                    self.headers.get("Sec-Fetch-Site", "same-origin") != "cross-site")

        def do_GET(self):
            if not self.allowed():
                return self.reply({"error": "Access is restricted to the application on this computer."}, 403)
            path = self.path.split("?", 1)[0]
            try:
                if path == "/api/settings":
                    return self.reply(dict(**app.settings(), token=app.token))
                if path == "/api/runs":
                    dirs = sorted(app.runs.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                    return self.reply([app.state(p.name) for p in dirs if (p / "status.json").is_file()][:30])
                if path.startswith("/api/runs/"):
                    parts = path.split("/")
                    state = app.state(parts[3])
                    if len(parts) == 4:
                        return self.reply(state)
                    if len(parts) == 5 and parts[4] == "log":
                        return self.reply(app.log_tail(parts[3]).encode(), content_type="text/plain; charset=utf-8")
                    if len(parts) == 6 and parts[4] == "files" and parts[5] in state["files"]:
                        content = (app.directory(parts[3]) / parts[5]).read_bytes()
                        return self.reply(content, content_type="text/plain; charset=utf-8")
                    raise ValueError("File not found")
                assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                          "/style.css": ("style.css", "text/css"), "/i18n.css": ("i18n.css", "text/css"),
                          "/i18n.js": ("i18n.js", "text/javascript")}
                if path in assets:
                    name, mime = assets[path]
                    return self.reply((STATIC / name).read_bytes(), content_type=mime + "; charset=utf-8")
                return self.reply({"error": "Not found"}, 404)
            except (ValueError, OSError) as exc:
                self.reply({"error": str(exc)}, 400)

        def do_POST(self):
            if not self.allowed() or not secrets.compare_digest(self.headers.get("X-UI-Token", ""), app.token):
                return self.reply({"error": "Invalid session. Reload the page."}, 403)
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 100000:
                    raise ValueError("Request data is too large or empty")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Invalid request data")
                if self.path == "/api/settings":
                    with app.lock:
                        env, configs = app.merge(data)
                        save_settings(env, configs, app.root / ".env")
                    return self.reply({"ok": True})
                if self.path == "/api/runs":
                    return self.reply(app.start(data), 202)
                if self.path == "/api/cancel":
                    return self.reply(app.cancel())
                return self.reply({"error": "Not found"}, 404)
            except (ValueError, TypeError, OSError) as exc:
                self.reply({"error": str(exc)}, 400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description="Local Research Assistant UI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    app = Application()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Research Assistant: {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.cancel()
        server.server_close()


if __name__ == "__main__":
    main()
