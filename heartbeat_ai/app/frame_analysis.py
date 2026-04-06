"""
Single-frame face + phone pipeline for uploaded browser JPEGs.

No idle debounce — ``is_present`` is ``face_count >= 1``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .anti_spoof import AntiSpoofFilter
from .config import Settings
from .phone_detector import PhoneBox, PhoneDetector
from .presence import FaceDetectionResult, detect_faces

logger = logging.getLogger("heartbeat")


def build_snap_from_detection(
    *,
    face_count: int,
    phone_detected: bool,
    high_risk_on_multi_user: bool,
) -> Dict[str, object]:
    """Status string and flags aligned with ``StateManager.update`` rules."""

    if face_count == 0:
        base = "No User"
        high_risk = False
    elif face_count == 1:
        base = "Single User"
        high_risk = False
    else:
        high_risk = bool(high_risk_on_multi_user)
        base = "Multiple Users (HIGH RISK)" if high_risk else "Multiple Users"
    status = f"{base} | PHONE DETECTED" if phone_detected else base
    return {
        "status": status,
        "is_present": face_count >= 1,
        "phone_detected": phone_detected,
        "high_risk": high_risk,
        "face_count": face_count,
    }


def analyze_frame_bgr(
    frame_bgr: np.ndarray,
    settings: Settings,
    phone_detector: PhoneDetector,
    anti_spoof: Optional[AntiSpoofFilter],
    *,
    face_zone_expand: float,
    run_phone_detection: bool = True,
) -> Tuple[FaceDetectionResult, List[PhoneBox], List[float], int, bool]:
    """Face + optional anti-spoof + size filter + optional YOLO phones."""

    try:
        face_res = detect_faces(frame_bgr)
    except Exception:
        logger.exception("Face detection error (frame_analysis)")
        face_res = FaceDetectionResult(0, [], [])

    liveness_scores: List[float] = []
    spoof_filtered = 0

    if anti_spoof is not None and anti_spoof.available and face_res.face_count > 0:
        try:
            liveness_scores = anti_spoof.liveness_scores(frame_bgr, face_res.boxes_xywh)
            fb, fc, n_spoof = anti_spoof.filter(
                frame_bgr, face_res.boxes_xywh, face_res.confidences
            )
            spoof_filtered = n_spoof
            face_res = FaceDetectionResult(len(fb), fb, fc)
        except Exception:
            logger.exception("Anti-spoof filter error (frame_analysis)")
            liveness_scores = []
            spoof_filtered = 0

    if settings.face_size_ratio_min > 0 and face_res.face_count > 1:
        areas = [w * h for (_, _, w, h) in face_res.boxes_xywh]
        max_area = max(areas)
        threshold_area = settings.face_size_ratio_min * max_area
        kept = [i for i, a in enumerate(areas) if a >= threshold_area]
        if len(kept) < face_res.face_count:
            new_boxes = [face_res.boxes_xywh[i] for i in kept]
            new_confs = [face_res.confidences[i] for i in kept]
            new_scores = [
                liveness_scores[i]
                for i in kept
                if i < len(liveness_scores)
            ]
            liveness_scores = new_scores
            face_res = FaceDetectionResult(len(new_boxes), new_boxes, new_confs)

    phone_boxes: List[PhoneBox] = []
    phone_detected = False
    if run_phone_detection and phone_detector.available:
        try:
            raw_boxes = phone_detector.detect_phones(frame_bgr)
            if face_zone_expand > 0 and face_res.boxes_xywh:
                fh, fw = frame_bgr.shape[:2]
                raw_boxes = PhoneDetector.filter_by_face_zone(
                    raw_boxes,
                    face_res.boxes_xywh,
                    frame_w=fw,
                    frame_h=fh,
                    expand_factor=face_zone_expand,
                )
            phone_boxes = raw_boxes
            phone_detected = len(phone_boxes) > 0
        except Exception:
            logger.exception("Phone detection error (frame_analysis)")
            phone_boxes = []
            phone_detected = False

    return face_res, phone_boxes, liveness_scores, spoof_filtered, phone_detected
