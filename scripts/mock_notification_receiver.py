#!/usr/bin/env python3
"""Local mock of the Slack/Discord incoming webhook, for n8n workflow testing.

Records every POSTed payload (append-only JSON Lines) so tests can assert on
exactly what the workflow would have sent to the real webhook, without ever
calling it. Never used by the real, single controlled external-action test.

Usage:
    venv/bin/python scripts/mock_notification_receiver.py [--port 8090] [--log PATH]
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parent.parent
    / "automation"
    / ".n8n-runtime"
    / "mock_notifications.jsonl"
)


class Handler(BaseHTTPRequestHandler):
    log_path: Path = DEFAULT_LOG_PATH

    def log_message(self, fmt: str, *args) -> None:
        print(f"[mock_notification_receiver] {self.address_string()} - {fmt % args}")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            payload = {"_raw": raw_body.decode("utf-8", errors="replace")}

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--log", type=str, default=str(DEFAULT_LOG_PATH))
    args = parser.parse_args()

    Handler.log_path = Path(args.log)
    Handler.log_path.parent.mkdir(parents=True, exist_ok=True)
    Handler.log_path.write_text("")  # start each run with a clean log

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Mock notification receiver listening on http://127.0.0.1:{args.port}")
    print(f"Logging received payloads to {Handler.log_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
