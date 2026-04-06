"""
CLI entry: camera + processing threads + FastAPI (uvicorn).
Production: no window. Use --debug for OpenCV preview.

From repository root (parent of `heartbeat_ai/`):
  python -m heartbeat_ai.run
From inside `heartbeat_ai/`:
  python run.py
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

# Ensure `heartbeat_ai` is importable whether cwd is repo root or package dir
_PKG_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from heartbeat_ai.app.camera import CameraThread  # noqa: E402
from heartbeat_ai.app.config import settings_from_env  # noqa: E402
from heartbeat_ai.app.logger import setup_logging  # noqa: E402
from heartbeat_ai.app.main import create_service  # noqa: E402


def _warn_if_numpy2(log: logging.Logger) -> None:
    """PyTorch wheels often break with NumPy 2.x until versions are aligned."""

    try:
        import numpy as np
    except Exception:
        return
    major_str, _, _ = np.__version__.partition(".")
    try:
        major = int(major_str)
    except ValueError:
        return
    if major >= 2:
        msg = (
            "NumPy %s is installed; Ultralytics/PyTorch usually require NumPy 1.x. "
            'Fix: pip install "numpy>=1.26.2,<2" then restart this app.'
        ) % np.__version__
        print(msg, file=sys.stderr)
        log.error(msg)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Heartbeat AI office monitor")
    p.add_argument("--debug", action="store_true", help="Show OpenCV window alongside HTTP API")
    p.add_argument(
        "--visual",
        action="store_true",
        help="Live tracking window only; do not start FastAPI / HTTP API",
    )
    p.add_argument("--host", default=None, help="API bind host (default: config)")
    p.add_argument("--port", type=int, default=None, help="API port (default: config)")
    p.add_argument(
        "--export",
        action="store_true",
        help="Enable JSON export (heartbeat_ai/exports/) and optional webhook (set HEARTBEAT_EXPORT_WEBHOOK_URL)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    settings = settings_from_env()
    if args.export:
        settings.export_enabled = True
    if args.host:
        settings.api_host = args.host
    if args.port is not None:
        settings.api_port = args.port

    visual_only = bool(args.visual)
    setup_logging(settings.log_path(), console=bool(args.debug or visual_only))
    log = logging.getLogger("heartbeat")
    log.info("Starting Heartbeat AI | debug=%s visual_only=%s", args.debug, visual_only)
    _warn_if_numpy2(log)

    camera_ok: list = [False]
    service = create_service(
        settings,
        debug=args.debug,
        visual_only=visual_only,
        camera_ok_holder=camera_ok,
    )

    cam = CameraThread(
        service.frame_queue,
        camera_index=settings.camera_index,
        width=settings.frame_width,
        height=settings.frame_height,
        retry_initial=settings.camera_open_retry_initial_sec,
        retry_max=settings.camera_open_retry_max_sec,
        read_fail_threshold=settings.camera_read_fail_threshold,
        camera_ok_callback=service.camera_ok_setter,
        shutdown_event=service.shutdown_event,
        daemon=True,
    )

    proc = threading.Thread(target=service.processing_loop, name="ProcessingThread", daemon=True)

    cam.start()
    proc.start()

    if not visual_only:
        import uvicorn  # noqa: E402

        from heartbeat_ai.app.api import create_api_app  # noqa: E402

        api_app = create_api_app(
            service.state,
            camera_ok,
            cors_enabled=settings.api_cors_enabled,
            last_anomaly_getter=service.get_last_anomaly_for_api,
        )

        def run_uvicorn() -> None:
            uvicorn.run(
                api_app,
                host=settings.api_host,
                port=settings.api_port,
                log_level="warning",
                access_log=False,
            )

        api_thread = threading.Thread(target=run_uvicorn, name="UvicornThread", daemon=True)
        api_thread.start()
        log.info(
            "API listening | http://%s:%s/status | http://%s:%s/last_anomaly",
            settings.api_host,
            settings.api_port,
            settings.api_host,
            settings.api_port,
        )
    else:
        log.info("Visual-only mode | live window: heartbeat_live | press Q in window or Ctrl+C to stop")

    stop = threading.Event()

    def handle_stop(*_a: object) -> None:
        stop.set()
        service.shutdown_event.set()

    if sys.platform != "win32":
        signal.signal(signal.SIGINT, handle_stop)
        signal.signal(signal.SIGTERM, handle_stop)

    try:
        while not stop.is_set() and not service.shutdown_event.is_set():
            time.sleep(0.3)
    except KeyboardInterrupt:
        handle_stop()

    service.shutdown_event.set()
    time.sleep(0.4)
    log.info("Shutdown complete")


if __name__ == "__main__":
    main()
