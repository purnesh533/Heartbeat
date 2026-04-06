"""Thread-safe global monitoring state for API and processing loop."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MonitorState:
    face_count: int = 0
    is_present: bool = False
    phone_detected: bool = False
    status: str = "No User"
    last_updated: float = field(default_factory=time.time)
    high_risk: bool = False
    # Optional: last idle duration in seconds (cleared after read if desired)
    last_idle_duration_sec: Optional[float] = None
    # Newest first; each entry: filename, relative_path, absolute_path, saved_at_unix
    evidence_history: List[Dict[str, Any]] = field(default_factory=list)
    # Last browser ``POST /ingest/frame`` summary (optional)
    browser_ingest: Optional[Dict[str, Any]] = None


class StateManager:
    """
    Central state with RLock. Status string rules:
    - No User / Single User / Multiple Users (HIGH RISK if configured)
    - Append ' | PHONE DETECTED' when phone_detected
    """

    def __init__(
        self,
        high_risk_on_multi_user: bool = True,
        *,
        evidence_history_max: int = 30,
    ) -> None:
        self._lock = threading.RLock()
        self._high_risk_on_multi = high_risk_on_multi_user
        self._evidence_history_max = max(1, int(evidence_history_max))
        self._state = MonitorState()

    def update(
        self,
        *,
        face_count: int,
        is_present: bool,
        phone_detected: bool,
        last_idle_duration_sec: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._state.face_count = face_count
            self._state.is_present = is_present
            self._state.phone_detected = phone_detected
            self._state.last_updated = time.time()
            if last_idle_duration_sec is not None:
                self._state.last_idle_duration_sec = last_idle_duration_sec

            if face_count == 0:
                base = "No User"
                self._state.high_risk = False
            elif face_count == 1:
                base = "Single User"
                self._state.high_risk = False
            else:
                base = "Multiple Users (HIGH RISK)" if self._high_risk_on_multi else "Multiple Users"
                self._state.high_risk = self._high_risk_on_multi

            status = base
            if phone_detected:
                status = f"{base} | PHONE DETECTED"
            self._state.status = status

    def record_evidence(self, ev: Dict[str, Any]) -> None:
        """Append a saved-on-disk evidence record (newest first, capped)."""

        entry = {**ev, "saved_at_unix": time.time()}
        with self._lock:
            self._state.evidence_history.insert(0, entry)
            if len(self._state.evidence_history) > self._evidence_history_max:
                del self._state.evidence_history[self._evidence_history_max :]

    def record_browser_ingest(self, summary: Dict[str, Any]) -> None:
        """Latest remote frame metadata for GET /status."""

        with self._lock:
            self._state.browser_ingest = {**summary, "last_at_unix": time.time()}

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            hist = list(self._state.evidence_history)
            return {
                "face_count": self._state.face_count,
                "is_present": self._state.is_present,
                "phone_detected": self._state.phone_detected,
                "status": self._state.status,
                "last_updated": self._state.last_updated,
                "high_risk": self._state.high_risk,
                "evidence": hist[0] if hist else None,
                "evidence_history": hist,
                "browser_ingest": self._state.browser_ingest,
            }

    def to_status_dict(self) -> Dict[str, Any]:
        """JSON-serializable payload for GET /status."""

        with self._lock:
            hist = list(self._state.evidence_history)
            return {
                "face_count": self._state.face_count,
                "is_present": self._state.is_present,
                "phone_detected": self._state.phone_detected,
                "status": self._state.status,
                "last_updated": self._state.last_updated,
                "evidence": hist[0] if hist else None,
                "evidence_history": hist,
                "browser_ingest": self._state.browser_ingest,
            }
