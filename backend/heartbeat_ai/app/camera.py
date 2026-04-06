"""OpenCV camera capture in a dedicated thread with reconnect + backoff."""

from __future__ import annotations

import logging
import threading
import time
import sys
from typing import Callable, Optional

import cv2

from .frame_queue import FrameQueue

logger = logging.getLogger("heartbeat")


class CameraThread(threading.Thread):
    """
    Captures 640x480 BGR frames into FrameQueue (drop-oldest on overflow).
    Sets camera_ok flag False on sustained failure.
    """

    def __init__(
        self,
        frame_queue: FrameQueue,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        retry_initial: float = 0.5,
        retry_max: float = 8.0,
        read_fail_threshold: int = 15,
        camera_ok_callback: Optional[Callable[[bool], None]] = None,
        shutdown_event: Optional[threading.Event] = None,
        daemon: bool = True,
    ) -> None:
        super().__init__(daemon=daemon, name="CameraThread")
        self._queue = frame_queue
        self._index = camera_index
        self._w = width
        self._h = height
        self._retry_initial = retry_initial
        self._retry_max = retry_max
        self._read_fail_threshold = read_fail_threshold
        self._camera_ok_cb = camera_ok_callback
        self._shutdown = shutdown_event or threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None

    def _set_camera_ok(self, ok: bool) -> None:
        if self._camera_ok_cb:
            self._camera_ok_cb(ok)

    def _try_backend(self, backend: Optional[int]) -> Optional[cv2.VideoCapture]:
        """
        Open the camera with a specific backend and verify it can actually stream.
        Returns a ready VideoCapture or None.

        MSMF (Windows Media Foundation) supports shared camera access — multiple apps
        (Zoom, Teams, etc.) can read the same camera simultaneously. DirectShow uses
        exclusive access and conflicts when another app holds the camera.
        We attempt MSMF first; if it opens but fails to deliver frames (a common MSMF
        quirk with some drivers), we fall back to DirectShow.
        """
        try:
            cap = cv2.VideoCapture(self._index, backend) if backend is not None else cv2.VideoCapture(self._index)
        except Exception:
            return None
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
        # Verify the backend can actually deliver frames before committing to it.
        # MSMF can report isOpened()=True but immediately error on grabFrame.
        ok, _ = cap.read()
        if not ok:
            try:
                cap.release()
            except Exception:
                pass
            return None
        return cap

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """
        Try backends in order: MSMF → DSHOW (Windows) or default (other OS).
        MSMF is tried first because it uses the Windows Camera Frame Server which
        allows shared access — Zoom/Teams can run at the same time without conflict.
        Falls back to DSHOW if MSMF cannot stream (driver compatibility issue).
        """
        backoff = self._retry_initial
        while not self._shutdown.is_set():
            backends: list[Optional[int]]
            if sys.platform == "win32":
                backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW]
                backend_names = ["MSMF", "DSHOW"]
            else:
                backends = [None]
                backend_names = ["default"]

            for backend, name in zip(backends, backend_names):
                cap = self._try_backend(backend)
                if cap is not None:
                    self._set_camera_ok(True)
                    logger.info(
                        "Camera opened | index=%s resolution=%sx%s backend=%s",
                        self._index, self._w, self._h, name,
                    )
                    return cap
                logger.debug("Camera backend %s unavailable | index=%s", name, self._index)

            logger.warning("Camera open failed (all backends) | index=%s retry in %.1fs", self._index, backoff)
            self._set_camera_ok(False)
            self._shutdown.wait(timeout=backoff)
            backoff = min(self._retry_max, backoff * 1.5)
        return None

    def run(self) -> None:
        read_fails = 0
        while not self._shutdown.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open_camera()
                if self._cap is None:
                    continue
                read_fails = 0

            ok, frame = self._cap.read()
            if not ok or frame is None:
                read_fails += 1
                if read_fails >= self._read_fail_threshold:
                    logger.error("Camera read failed repeatedly; reconnecting")
                    self._set_camera_ok(False)
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    self._cap = None
                    read_fails = 0
                time.sleep(0.02)
                continue

            read_fails = 0
            self._set_camera_ok(True)
            # Ensure size (some drivers ignore CAP_PROP)
            if frame.shape[0] != self._h or frame.shape[1] != self._w:
                frame = cv2.resize(frame, (self._w, self._h), interpolation=cv2.INTER_AREA)

            self._queue.put_frame(frame, time.monotonic())

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                logger.exception("Error releasing camera")
