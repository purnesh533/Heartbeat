"""FastAPI: GET /status, GET /health, POST /ingest/frame (browser JPEG)."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .evidence_store import EvidenceStore
from .state_manager import StateManager


def create_api_app(
    state_manager: StateManager,
    camera_ok_ref: List[bool],
    *,
    title: str = "Heartbeat AI",
    cors_enabled: bool = True,
    api_only: bool = False,
    last_anomaly_getter: Optional[Callable[[], Dict[str, Any]]] = None,
    browser_ingest_handler: Optional[
        Callable[[bytes, Optional[str]], Dict[str, Any]]
    ] = None,
    browser_ingest_max_bytes: int = 6_000_000,
    evidence_store: Optional[EvidenceStore] = None,
    evidence_read_api_key: str = "",
) -> FastAPI:
    app = FastAPI(title=title, version="1.0.0")

    def _evidence_key_ok(x_evidence_key: Optional[str]) -> None:
        expected = (evidence_read_api_key or "").strip()
        if not expected:
            return
        if (x_evidence_key or "").strip() != expected:
            raise HTTPException(status_code=401, detail="evidence_key_required")

    if cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/")
    def root() -> Dict[str, Any]:
        """Service index; static UI is deployed separately under ``frontend/``."""

        return {
            "service": "Heartbeat AI API",
            "endpoints": {
                "health": "/health",
                "status": "/status",
                "last_anomaly": "/last_anomaly",
                "ingest_frame": "POST /ingest/frame",
                "evidence_recent": "/evidence/recent",
                "evidence_image": "/evidence/{id}/image",
                "openapi": "/docs",
            },
            "evidence_database": bool(evidence_store and evidence_store.ready),
        }

    @app.get("/status")
    def get_status() -> Dict[str, Any]:
        return state_manager.to_status_dict()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "camera_ok": bool(camera_ok_ref[0]),
            "api_only": api_only,
        }

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

    @app.get("/evidence/recent")
    def evidence_recent(
        limit: int = 30,
        x_evidence_key: Optional[str] = Header(None, alias="X-Evidence-Key"),
    ) -> Dict[str, Any]:
        """List recent rows from PostgreSQL evidence store (metadata + full detection JSON)."""

        _evidence_key_ok(x_evidence_key)
        if evidence_store is None or not evidence_store.ready:
            raise HTTPException(status_code=503, detail="evidence_database_disabled")
        rows = evidence_store.list_recent(limit)
        return {"ok": True, "count": len(rows), "items": rows}

    @app.get("/evidence/{row_id}/image")
    def evidence_image(
        row_id: int,
        x_evidence_key: Optional[str] = Header(None, alias="X-Evidence-Key"),
    ) -> Response:
        """Return stored annotated JPEG for a given evidence id."""

        _evidence_key_ok(x_evidence_key)
        if evidence_store is None or not evidence_store.ready:
            raise HTTPException(status_code=503, detail="evidence_database_disabled")
        data = evidence_store.fetch_jpeg(row_id)
        if not data:
            raise HTTPException(status_code=404, detail="not_found")
        return Response(content=data, media_type="image/jpeg")

    return app
