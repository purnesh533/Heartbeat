#!/usr/bin/env python3
"""
Local receiver for Heartbeat ``export_webhook_url`` testing.

  python mock_webhook_server.py

Then set (PowerShell), from ``backend/``:

  $env:HEARTBEAT_EXPORT_ENABLED = "1"
  $env:HEARTBEAT_EXPORT_WEBHOOK_URL = "http://127.0.0.1:9000/ingest"

Run API:  cd backend && python -m heartbeat_ai.run

Open http://127.0.0.1:9000/ for the webhook viewer (last POST + optional image).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

_DIR = Path(__file__).resolve().parent
_VIEWER = _DIR / "webhook_viewer.html"

_lock = threading.Lock()
_last: Dict[str, Any] = {"empty": True}


class Handler(BaseHTTPRequestHandler):
    server_version = "HeartbeatMockWebhook/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        msg = fmt % args
        if "POST /ingest" in msg or "POST /webhook" in msg:
            print(f"[webhook] {msg.strip()}")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        if urlparse(self.path).path in ("/ingest", "/webhook"):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/ingest", "/webhook"):
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send(400, b'{"ok":false,"error":"invalid json"}', "application/json")
            return

        with _lock:
            global _last
            _last = {"empty": False, "payload": payload}

        out = _DIR / "last_webhook.json"
        try:
            # Write compact copy for inspection (can be large if image embedded)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except OSError:
            pass

        self._send(200, b'{"ok":true}', "application/json")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/last":
            with _lock:
                body = json.dumps(_last, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if path in ("/", "/viewer"):
            if _VIEWER.is_file():
                data = _VIEWER.read_bytes()
                self._send(200, data, "text/html; charset=utf-8")
            else:
                self.send_error(404)
            return
        self.send_error(404)


def main() -> None:
    host, port = "127.0.0.1", 9000
    httpd = HTTPServer((host, port), Handler)
    print(f"Mock webhook: POST http://{host}:{port}/ingest")
    print(f"Viewer:       http://{host}:{port}/")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
