"""FastAPI: GET /status, GET /health, POST /ingest/frame (browser JPEG)."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .state_manager import StateManager


def create_api_app(
    state_manager: StateManager,
    camera_ok_ref: List[bool],
    *,
    title: str = "Heartbeat AI",
    cors_enabled: bool = True,
    last_anomaly_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    browser_ingest_handler: Optional[
        Callable[[bytes, Optional[str]], Dict[str, Any]]
    ] = None,
    browser_ingest_max_bytes: int = 6_000_000,
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

    _ingest_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")

    @app.post("/ingest/frame")
    async def ingest_frame(
        file: UploadFile = File(...),
        x_ingest_key: Optional[str] = Header(None, alias="X-Ingest-Key"),
    ) -> Dict[str, Any]:
        """
        Multipart field ``file``: one JPEG from the browser webcam.
        Optional header ``X-Ingest-Key`` when ``HEARTBEAT_BROWSER_INGEST_KEY`` is set.
        """
        if browser_ingest_handler is None:
            raise HTTPException(status_code=503, detail="ingest_not_configured")
        raw = await file.read()
        if len(raw) > browser_ingest_max_bytes:
            raise HTTPException(status_code=413, detail="payload_too_large")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _ingest_pool,
            lambda: browser_ingest_handler(raw, x_ingest_key),
        )
        if not result.get("ok"):
            err = str(result.get("error", "ingest_failed"))
            if err == "unauthorized":
                raise HTTPException(status_code=401, detail=err)
            if err == "browser_ingest_disabled":
                raise HTTPException(status_code=503, detail=err)
            if err in ("invalid_jpeg", "payload_too_large"):
                raise HTTPException(
                    status_code=400 if err == "invalid_jpeg" else 413,
                    detail=err,
                )
            raise HTTPException(status_code=400, detail=err)
        return result

    return app
