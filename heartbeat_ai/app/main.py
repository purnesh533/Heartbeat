"""Orchestration: processing thread, models, state, optional DB; entry helpers for run.py."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .anomaly_export import AnomalyExporter, build_detection_payload
from .anti_spoof import AntiSpoofFilter
from .config import Settings
from .database import EventDatabase
from .frame_queue import FrameQueue
from .idle_timer import IdleTickResult, IdleTimer
from .logger import log_event
from .phone_detector import PhoneBox, PhoneDetector
from .presence import (
    FaceDetectionResult,
    close_detector,
    configure_presence,
    detect_faces,
    face_backend,
)
from .state_manager import StateManager

logger = logging.getLogger("heartbeat")


def _put_text_outline(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    outline_px: int = 2,
) -> None:
    x, y = int(org[0]), int(org[1])
    for ox in range(-outline_px, outline_px + 1):
        for oy in range(-outline_px, outline_px + 1):
            if ox == 0 and oy == 0:
                continue
            cv2.putText(
                img,
                text,
                (x + ox, y + oy),
                font,
                scale,
                (0, 0, 0),
                thickness + 1,
                cv2.LINE_AA,
            )
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _draw_labeled_box(
    vis: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    label: str,
) -> None:
    """Draw rectangle and a small filled label bar above the top edge (outlined text)."""

    if len(label) > 40:
        label = label[:37] + "..."
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.52
    th = 1
    (tw, thh), bl = cv2.getTextSize(label, font, fs, th)
    pad = 4
    ty2 = y1
    ty1 = max(0, y1 - thh - bl - 2 * pad)
    tx2 = min(vis.shape[1] - 1, x1 + tw + 2 * pad)
    cv2.rectangle(vis, (x1, ty1), (tx2, ty2), (0, 0, 0), -1)
    cv2.rectangle(vis, (x1, ty1), (tx2, ty2), color, 1)
    _put_text_outline(vis, label, (x1 + pad, ty2 - pad), font, fs, color, th, 2)


def _draw_confidence_badge(
    vis: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    confidence: float,
    color: tuple[int, int, int],
) -> None:
    """Large numeric confidence under the box, readable on any background."""

    txt = f"{confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thick = 2
    (tw, th), bl = cv2.getTextSize(txt, font, scale, thick)
    cx = (x1 + x2) // 2
    yb = min(vis.shape[0] - 2, y2 + th + bl + 12)
    xa = max(2, cx - tw // 2 - 8)
    xb = min(vis.shape[1] - 2, xa + tw + 16)
    ya = max(0, yb - th - bl - 10)
    cv2.rectangle(vis, (xa, ya), (xb, yb), (0, 0, 0), -1)
    cv2.rectangle(vis, (xa, ya), (xb, yb), color, 1)
    _put_text_outline(vis, txt, (xa + 8, yb - 5), font, scale, color, thick, 2)


class MonitorService:
    """
    Owns queue, state, idle timer, phone detector, optional DB.
    processing_loop runs in ProcessingThread until shutdown_event.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        debug: bool = False,
        visual_only: bool = False,
        camera_ok_holder: Optional[list] = None,
    ) -> None:
        self.settings = settings
        configure_presence(settings)
        self.debug = debug
        self.visual_only = visual_only
        # OpenCV preview when --debug or --visual (visual_only also skips HTTP API in run.py)
        self.show_preview = debug or visual_only
        self._camera_ok = camera_ok_holder if camera_ok_holder is not None else [False]

        self.frame_queue = FrameQueue(maxsize=settings.queue_max_size)
        self.state = StateManager(
            high_risk_on_multi_user=settings.high_risk_on_multi_user,
            evidence_history_max=settings.evidence_status_history_max,
        )
        self.idle_timer = IdleTimer(absence_buffer_sec=settings.absence_buffer_sec)
        self.idle_timer.reset_clock()

        self._face_zone_expand = settings.yolo_phone_face_zone_expand

        self.anti_spoof: Optional[AntiSpoofFilter] = None
        if settings.anti_spoof_enabled:
            self.anti_spoof = AntiSpoofFilter(
                models_dir=settings.models_dir(),
                threshold=settings.anti_spoof_threshold,
            )

        self.phone_detector = PhoneDetector(
            model_path=str(settings.yolo_path()),
            target_substrings=settings.yolo_target_class_substrings,
            infer_max_side=settings.yolo_infer_max_side,
            conf_threshold=settings.yolo_conf_threshold,
            bottom_crop_fallback=settings.yolo_bottom_crop_fallback,
            bottom_crop_start_ratio=settings.yolo_bottom_crop_start_ratio,
            side_crop_fallback=settings.yolo_side_crop_fallback,
            side_crop_width_ratio=settings.yolo_side_crop_width_ratio,
            filter_landscape_low_conf=settings.yolo_phone_filter_landscape,
            landscape_aspect=settings.yolo_phone_landscape_aspect,
            landscape_min_conf=settings.yolo_phone_landscape_min_conf,
            min_box_height_frac=settings.yolo_phone_min_height_frac,
        )

        self.db: Optional[EventDatabase] = None
        try:
            self.db = EventDatabase(settings.db_path())
            self.db.init()
        except Exception:
            logger.exception("SQLite init failed; continuing without DB")

        self.shutdown_event = threading.Event()

        self._prev_present: Optional[bool] = None
        self._prev_face_count: int = -1
        self._prev_phone: bool = False
        self._frame_index: int = 0
        self._last_phone_boxes: List[PhoneBox] = []
        self._last_spoof_filtered: int = 0
        self._last_liveness_scores: List[float] = []

        self._anomaly_api_lock = threading.Lock()
        self._last_anomaly_api: Optional[Dict[str, Any]] = None
        self._last_anomaly_api_mono: float = 0.0
        # At most one evidence history row per processed frame (API + exporter may both save).
        self._evidence_logged_for_frame: int = -1

        self._exporter: Optional[AnomalyExporter] = None
        if settings.export_enabled:
            self._exporter = AnomalyExporter(
                settings,
                on_anomaly_payload=self._record_evidence_if_new,
            )

    def _record_evidence_if_new(self, payload: Dict[str, Any]) -> None:
        ev = payload.get("evidence")
        if not ev:
            return
        if self._evidence_logged_for_frame == self._frame_index:
            return
        self._evidence_logged_for_frame = self._frame_index
        self.state.record_evidence(ev)

    def get_last_anomaly_for_api(self) -> Dict[str, Any]:
        """Thread-safe payload for GET /last_anomaly (demo + downstream)."""
        if not self.settings.api_last_anomaly_enabled:
            return {"available": False, "reason": "disabled"}
        with self._anomaly_api_lock:
            if self._last_anomaly_api is None:
                return {"available": False}
            return {"available": True, **self._last_anomaly_api}

    def camera_ok_setter(self, ok: bool) -> None:
        self._camera_ok[0] = ok

    def processing_loop(self) -> None:
        """Consumer: face + periodic YOLO + idle + state + logs."""

        yolo_ok = self.phone_detector.ensure_model_loaded()
        if not yolo_ok:
            log_event(logger, "YOLO degraded", self.phone_detector.degraded_reason or "load failed")

        if self.anti_spoof is not None:
            spoof_ok = self.anti_spoof.ensure_loaded()
            if not spoof_ok:
                log_event(logger, "AntiSpoof degraded", self.anti_spoof._degraded_reason)

        while not self.shutdown_event.is_set():
            item = self.frame_queue.get_frame(timeout=0.05)
            if item is None:
                continue

            frame, _ts = item
            self._frame_index += 1

            try:
                face_res = detect_faces(frame)
            except Exception:
                logger.exception("Face detection error")
                face_res = FaceDetectionResult(0, [], [])

            # Anti-spoofing: remove photos / screens / printed posters
            if self.anti_spoof is not None and self.anti_spoof.available and face_res.face_count > 0:
                try:
                    live_scores = self.anti_spoof.liveness_scores(frame, face_res.boxes_xywh)
                    fb, fc, n_spoof = self.anti_spoof.filter(
                        frame, face_res.boxes_xywh, face_res.confidences
                    )
                    self._last_spoof_filtered = n_spoof
                    self._last_liveness_scores = live_scores
                    face_res = FaceDetectionResult(len(fb), fb, fc)
                except Exception:
                    logger.exception("Anti-spoof filter error")
                    self._last_spoof_filtered = 0
                    self._last_liveness_scores = []
            else:
                self._last_spoof_filtered = 0
                self._last_liveness_scores = []

            # Background face size filter: drop any face whose box area is
            # smaller than face_size_ratio_min × the dominant (largest) face area.
            # Catches posters / screens / far-away people that anti-spoof misses.
            if (
                self.settings.face_size_ratio_min > 0
                and face_res.face_count > 1
            ):
                areas = [w * h for (_, _, w, h) in face_res.boxes_xywh]
                max_area = max(areas)
                threshold_area = self.settings.face_size_ratio_min * max_area
                kept = [
                    i for i, a in enumerate(areas) if a >= threshold_area
                ]
                if len(kept) < face_res.face_count:
                    dropped = face_res.face_count - len(kept)
                    logger.debug(
                        "Size filter: dropped %d background face(s) "
                        "(ratio < %.2f of dominant)",
                        dropped,
                        self.settings.face_size_ratio_min,
                    )
                    new_boxes = [face_res.boxes_xywh[i] for i in kept]
                    new_confs = [face_res.confidences[i] for i in kept]
                    new_scores = [
                        self._last_liveness_scores[i]
                        for i in kept
                        if i < len(self._last_liveness_scores)
                    ]
                    self._last_liveness_scores = new_scores
                    face_res = FaceDetectionResult(len(new_boxes), new_boxes, new_confs)

            face_count = face_res.face_count
            now_mono = time.monotonic()

            try:
                idle_res = self.idle_timer.tick(face_count, now_mono=now_mono)
            except Exception:
                logger.exception("Idle timer error")
                idle_res = IdleTickResult(is_present=(face_count >= 1), idle_ended_sec=None)

            is_present = idle_res.is_present

            phone_detected = self._prev_phone
            if self.phone_detector.available and (self._frame_index % self.settings.yolo_every_n_frames == 0):
                try:
                    raw_boxes = self.phone_detector.detect_phones(frame)
                    if self._face_zone_expand > 0 and face_res.boxes_xywh:
                        fh, fw = frame.shape[:2]
                        raw_boxes = PhoneDetector.filter_by_face_zone(
                            raw_boxes,
                            face_res.boxes_xywh,
                            frame_w=fw,
                            frame_h=fh,
                            expand_factor=self._face_zone_expand,
                        )
                    self._last_phone_boxes = raw_boxes
                    phone_detected = len(self._last_phone_boxes) > 0
                except Exception:
                    logger.exception("Phone detection error")
                    self._last_phone_boxes = []
                    phone_detected = False

            # Logs: transitions and rising edges
            if self._prev_present is True and is_present is False:
                log_event(logger, "User absent", "no face beyond buffer")
            if self._prev_present is False and is_present is True:
                log_event(logger, "User returned", "face detected")
            if idle_res.idle_ended_sec is not None and idle_res.idle_ended_sec > 0:
                log_event(
                    logger,
                    "Idle duration",
                    f"seconds={idle_res.idle_ended_sec:.1f}",
                )

            if face_count > 1 and self._prev_face_count <= 1:
                log_event(logger, "Multiple users detected", f"face_count={face_count}")

            if phone_detected and not self._prev_phone:
                log_event(logger, "Phone detected", "cell phone or tablet class")

            self._prev_present = is_present
            self._prev_face_count = face_count
            self._prev_phone = phone_detected

            self.state.update(
                face_count=face_count,
                is_present=is_present,
                phone_detected=phone_detected,
                last_idle_duration_sec=idle_res.idle_ended_sec,
            )

            snap = self.state.snapshot()
            spoof_name = (
                self.anti_spoof.model_name
                if self.anti_spoof is not None
                else "disabled"
            )

            # Annotated frame for GET /last_anomaly (demo frontend) while anomaly active
            if self.settings.api_last_anomaly_enabled:
                risk = bool(snap.get("phone_detected") or snap.get("high_risk"))
                if risk:
                    if (
                        now_mono - self._last_anomaly_api_mono
                        >= self.settings.api_last_anomaly_min_interval_sec
                    ):
                        self._last_anomaly_api_mono = now_mono
                        try:
                            ev_dir = (
                                self.settings.evidence_dir_path()
                                if self.settings.evidence_save_enabled
                                else None
                            )
                            ev_root = self.settings.models_dir().parent if ev_dir else None
                            pl = build_detection_payload(
                                frame_index=self._frame_index,
                                face_res=face_res,
                                phone_boxes=self._last_phone_boxes,
                                liveness_scores=self._last_liveness_scores,
                                spoof_filtered=self._last_spoof_filtered,
                                snap=snap,
                                face_engine=face_backend(),
                                anti_spoof_name=spoof_name,
                                include_image_b64=True,
                                frame_bgr=frame,
                                jpeg_quality=self.settings.api_last_anomaly_jpeg_quality,
                                save_evidence_to=ev_dir,
                                evidence_package_root=ev_root,
                            )
                            with self._anomaly_api_lock:
                                self._last_anomaly_api = pl
                            self._record_evidence_if_new(pl)
                        except Exception:
                            logger.exception("Last-anomaly API snapshot failed")
                else:
                    with self._anomaly_api_lock:
                        self._last_anomaly_api = None

            if self._exporter is not None:
                try:
                    self._exporter.tick(
                        frame_bgr=frame,
                        face_res=face_res,
                        phone_boxes=self._last_phone_boxes,
                        liveness_scores=self._last_liveness_scores,
                        spoof_filtered=self._last_spoof_filtered,
                        snap=snap,
                        frame_index=self._frame_index,
                        face_engine=face_backend(),
                        anti_spoof_name=spoof_name,
                        now_mono=now_mono,
                    )
                except Exception:
                    logger.exception("Anomaly export tick failed")

            if self.db:
                snap = self.state.snapshot()
                self.db.insert_if_due(
                    face_count=snap["face_count"],
                    phone_detected=snap["phone_detected"],
                    status=str(snap["status"]),
                    min_interval_sec=self.settings.db_insert_min_interval_sec,
                )

            if self.show_preview:
                vis = frame.copy()
                snap = self.state.snapshot()
                status = str(snap["status"])
                present = snap["is_present"]
                high = snap["high_risk"]

                confs = face_res.confidences
                for i, (x, y, bw, bh) in enumerate(face_res.boxes_xywh):
                    color = (0, 0, 255) if high else (0, 255, 0)
                    x2, y2 = x + bw, y + bh
                    cf = confs[i] if i < len(confs) else 0.0
                    live = (self._last_liveness_scores[i]
                            if i < len(self._last_liveness_scores) else None)
                    label = f"Face  live:{live:.2f}" if live is not None else "Face"
                    _draw_labeled_box(vis, x, y, x2, y2, color, label)
                    _draw_confidence_badge(vis, x, y, x2, y2, max(0.0, min(1.0, cf)), color)

                red = (0, 0, 255)
                for pb in self._last_phone_boxes:
                    short = pb.label if len(pb.label) <= 22 else pb.label[:19] + "..."
                    _draw_labeled_box(vis, pb.x1, pb.y1, pb.x2, pb.y2, red, short)
                    _draw_confidence_badge(
                        vis,
                        pb.x1,
                        pb.y1,
                        pb.x2,
                        pb.y2,
                        max(0.0, min(1.0, pb.confidence)),
                        red,
                    )

                y0 = 28
                lh = 26
                fmax = max(face_res.confidences) if face_res.confidences else 0.0
                pmax = max((pb.confidence for pb in self._last_phone_boxes), default=0.0)
                spoof_tag = (f"  |  spoof blocked: {self._last_spoof_filtered}"
                             if self._last_spoof_filtered > 0 else "")
                spoof_engine = (
                    f"  |  anti-spoof: {self.anti_spoof.model_name}"
                    if self.anti_spoof is not None else ""
                )
                lines = [
                    f"Status: {status}",
                    f"Present: {present}  |  faces: {face_count}{spoof_tag}  |  phone: {phone_detected}",
                    f"Face engine: {face_backend()}  |  max face conf: {fmax:.2f}  |  max phone conf: {pmax:.2f}{spoof_engine}",
                ]
                if self.visual_only:
                    lines.append("Press Q in this window to quit")

                for i, line in enumerate(lines):
                    y = y0 + i * lh
                    cv2.putText(
                        vis,
                        line,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255) if not high else (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )

                win = "heartbeat_live" if self.visual_only else "heartbeat_debug"
                cv2.imshow(win, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    self.shutdown_event.set()
                try:
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        self.shutdown_event.set()
                except Exception:
                    pass

        if self.show_preview:
            for win in ("heartbeat_live", "heartbeat_debug"):
                try:
                    cv2.destroyWindow(win)
                except Exception:
                    pass
        close_detector()


def create_service(
    settings: Settings,
    *,
    debug: bool = False,
    visual_only: bool = False,
    camera_ok_holder: Optional[list] = None,
) -> MonitorService:
    return MonitorService(
        settings=settings,
        debug=debug,
        visual_only=visual_only,
        camera_ok_holder=camera_ok_holder,
    )
