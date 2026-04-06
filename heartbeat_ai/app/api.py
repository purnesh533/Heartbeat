"""FastAPI: GET /status and GET /health for .NET heartbeat integration."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .state_manager import StateManager


def create_api_app(
    state_manager: StateManager,
    camera_ok_ref: List[bool],
    *,
    title: str = "Heartbeat AI",
    cors_enabled: bool = True,
    last_anomaly_getter: Optional[Callable[[], Dict[str, Any]]] = None,
) -> FastAPI:
    app = FastAPI(title=title, version="1.0.0")

    if cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/status")
    def get_status() -> Dict[str, Any]:
        return state_manager.to_status_dict()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "camera_ok": bool(camera_ok_ref[0])}

    @app.get("/last_anomaly")
    def last_anomaly() -> Dict[str, Any]:
        """
        While a phone or multi-user risk is active, returns the latest annotated
        frame as ``image_jpeg_base64`` plus the same detection fields as export JSON.
        When no anomaly: ``{"available": false}``.
        """
        if last_anomaly_getter is None:
            return {"available": False, "reason": "unconfigured"}
        return last_anomaly_getter()

    return app
