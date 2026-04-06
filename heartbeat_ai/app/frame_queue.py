"""Bounded frame queue: producer drops oldest on overflow to keep real-time."""

from __future__ import annotations

import queue
from typing import Optional

import numpy as np


class FrameQueue:
    """Thin wrapper around queue.Queue for BGR frames (numpy ndarray)."""

    def __init__(self, maxsize: int = 5) -> None:
        self._q: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=maxsize)

    def put_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """
        Non-blocking put: if full, remove oldest then put (drop-oldest policy).
        timestamp: monotonic time from time.monotonic() for ordering/debug.
        """

        item = (frame, timestamp)
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(item)
            except queue.Full:
                pass

    def get_frame(self, timeout: Optional[float] = 0.05) -> Optional[tuple[np.ndarray, float]]:
        """Block briefly for a frame; None if timeout (lets caller check shutdown)."""

        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        return self._q.qsize()
