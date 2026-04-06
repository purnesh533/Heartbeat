"""
Persist anomaly evidence (annotated JPEG + detection JSON) to PostgreSQL.

Set ``HEARTBEAT_EVIDENCE_DATABASE_URL`` (e.g. Render / Railway / Supabase Postgres).
Disk export under ``exports/evidence/`` is optional and separate.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("heartbeat")

_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS heartbeat_evidence (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    evidence_source TEXT NOT NULL DEFAULT 'camera',
    payload_json JSONB NOT NULL,
    image_jpeg BYTEA NOT NULL
)
"""
_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_heartbeat_evidence_created_at
    ON heartbeat_evidence (created_at DESC)
"""


def _clean_payload_for_db(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Drop large / redundant keys before storing as JSONB."""

    skip = frozenset({"image_jpeg_base64", "image_mime", "evidence"})
    return {k: v for k, v in payload.items() if k not in skip}


class EvidenceStore:
    """Thread-safe PostgreSQL persistence for annotated anomaly frames."""

    def __init__(self, dsn: str) -> None:
        self._dsn = (dsn or "").strip()
        self._lock = threading.Lock()
        self._ready = False
        self._psycopg2: Any = None
        if self._dsn:
            try:
                import psycopg2  # noqa: WPS433

                self._psycopg2 = psycopg2
            except ImportError:
                logger.error(
                    "HEARTBEAT_EVIDENCE_DATABASE_URL is set but psycopg2 is not installed; "
                    "pip install psycopg2-binary"
                )

    @property
    def ready(self) -> bool:
        return self._ready

    def init(self) -> bool:
        if not self._dsn or self._psycopg2 is None:
            return False
        try:
            with self._lock:
                conn = self._psycopg2.connect(self._dsn)
                try:
                    conn.autocommit = True
                    with conn.cursor() as cur:
                        cur.execute(_TABLE_SQL)
                        cur.execute(_INDEX_SQL)
                finally:
                    conn.close()
            self._ready = True
            logger.info("Evidence PostgreSQL: schema ready")
            return True
        except Exception:
            logger.exception("Evidence PostgreSQL: init failed")
            self._ready = False
            return False

    def insert(
        self,
        *,
        jpeg_bytes: bytes,
        payload: Dict[str, Any],
        evidence_source: str,
    ) -> Optional[int]:
        if not self._ready or not jpeg_bytes:
            return None
        from psycopg2.extras import Json  # noqa: WPS433

        clean = _clean_payload_for_db(payload)
        try:
            with self._lock:
                conn = self._psycopg2.connect(self._dsn)
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO heartbeat_evidence
                                (evidence_source, payload_json, image_jpeg)
                            VALUES (%s, %s, %s)
                            RETURNING id
                            """,
                            (evidence_source[:64], Json(clean), self._psycopg2.Binary(jpeg_bytes)),
                        )
                        row = cur.fetchone()
                    conn.commit()
                    if row:
                        return int(row[0])
                finally:
                    conn.close()
        except Exception:
            logger.exception("Evidence PostgreSQL: insert failed")
        return None

    def list_recent(self, limit: int) -> List[Dict[str, Any]]:
        if not self._ready:
            return []
        lim = max(1, min(200, int(limit)))
        try:
            with self._lock:
                conn = self._psycopg2.connect(self._dsn)
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT id, created_at, evidence_source, payload_json
                            FROM heartbeat_evidence
                            ORDER BY id DESC
                            LIMIT %s
                            """,
                            (lim,),
                        )
                        rows = cur.fetchall()
                finally:
                    conn.close()
        except Exception:
            logger.exception("Evidence PostgreSQL: list_recent failed")
            return []
        out: List[Dict[str, Any]] = []
        for r in rows:
            pid, created_at, src, pj = r
            if isinstance(pj, str):
                try:
                    pj = json.loads(pj)
                except json.JSONDecodeError:
                    pj = {}
            det = pj if isinstance(pj, dict) else {}
            out.append(
                {
                    "id": int(pid),
                    "created_at": created_at.isoformat() if created_at else None,
                    "evidence_source": src,
                    "frame_index": det.get("frame_index"),
                    "status": det.get("status"),
                    "face_count": det.get("face_count"),
                    "phone_detected": det.get("phone_detected"),
                    "high_risk": det.get("high_risk"),
                    "anomaly_reasons": det.get("anomaly_reasons"),
                    "timestamp_unix": det.get("timestamp_unix"),
                    "detection": det,
                }
            )
        return out

    def fetch_jpeg(self, row_id: int) -> Optional[bytes]:
        if not self._ready:
            return None
        try:
            with self._lock:
                conn = self._psycopg2.connect(self._dsn)
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT image_jpeg FROM heartbeat_evidence WHERE id = %s",
                            (int(row_id),),
                        )
                        row = cur.fetchone()
                finally:
                    conn.close()
            if row and row[0]:
                return bytes(row[0])
        except Exception:
            logger.exception("Evidence PostgreSQL: fetch_jpeg failed")
        return None
