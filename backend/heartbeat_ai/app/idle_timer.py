"""Absence buffer and idle duration: present immediately; absent only after N seconds without a face."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class IdleTickResult:
    """Result of one tick: debounced presence and optional idle duration when user returns."""

    is_present: bool
    idle_ended_sec: Optional[float]  # set once when transitioning absent -> face seen again


class IdleTimer:
    """
    - last_face_mono: last time.monotonic() when face_count >= 1
    - While face_count == 0: if now - last_face > buffer → absent (is_present False)
    - While face_count == 0 and within buffer → still is_present True (debounce)
    - On return (face_count >= 1) after absent: idle_ended_sec = now - (last_face_mono + buffer)
    """

    def __init__(self, absence_buffer_sec: float = 5.0) -> None:
        self._buffer = absence_buffer_sec
        self._last_face_mono: float = time.monotonic()
        self._was_absent: bool = False
        # Avoid logging a bogus multi-minute idle when the first face appears at startup
        self._had_seen_face: bool = False

    def reset_clock(self) -> None:
        """Call on startup so first frame doesn't think user was away for hours."""

        self._last_face_mono = time.monotonic()
        self._was_absent = False
        self._had_seen_face = False

    def tick(self, face_count: int, now_mono: Optional[float] = None) -> IdleTickResult:
        now = now_mono if now_mono is not None else time.monotonic()
        idle_ended: Optional[float] = None

        if face_count >= 1:
            if self._was_absent and self._had_seen_face:
                away_start = self._last_face_mono + self._buffer
                idle_ended = max(0.0, now - away_start)
            self._last_face_mono = now
            self._had_seen_face = True
            self._was_absent = False
            return IdleTickResult(is_present=True, idle_ended_sec=idle_ended)

        # no face
        gap = now - self._last_face_mono
        if gap > self._buffer:
            self._was_absent = True
            return IdleTickResult(is_present=False, idle_ended_sec=None)

        # still inside debounce window → treat as present
        return IdleTickResult(is_present=True, idle_ended_sec=None)
