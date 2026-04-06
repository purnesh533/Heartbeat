"""Optional SQLite persistence for monitoring snapshots."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
class EventDatabase:
    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._lock = threading.Lock()
        self._last_insert_mono: float = 0.0

    def init(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    face_count INTEGER NOT NULL,
                    phone_detected INTEGER NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def insert_if_due(
        self,
        *,
        face_count: int,
        phone_detected: bool,
        status: str,
        min_interval_sec: float,
    ) -> None:
        """Insert at most once per min_interval_sec wall time."""

        now = time.monotonic()
        with self._lock:
            if now - self._last_insert_mono < min_interval_sec:
                return
            self._last_insert_mono = now
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        try:
            with sqlite3.connect(self._path) as conn:
                conn.execute(
                    "INSERT INTO events (timestamp, face_count, phone_detected, status) VALUES (?, ?, ?, ?)",
                    (ts, face_count, 1 if phone_detected else 0, status),
                )
                conn.commit()
        except Exception:
            # Logged by caller if needed
            pass

    def insert_event(
        self,
        *,
        face_count: int,
        phone_detected: bool,
        status: str,
    ) -> None:
        """Unconditional insert (e.g. on significant transition)."""

        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        try:
            with sqlite3.connect(self._path) as conn:
                conn.execute(
                    "INSERT INTO events (timestamp, face_count, phone_detected, status) VALUES (?, ?, ?, ?)",
                    (ts, face_count, 1 if phone_detected else 0, status),
                )
                conn.commit()
        except Exception:
            pass
