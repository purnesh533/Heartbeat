"""Face detection: MediaPipe BlazeFace → OpenCV YuNet (DNN) → Haar cascade."""

from __future__ import annotations

import importlib
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

# Merge duplicate Haar hits on the same face (overlapping boxes)
_OPENCV_HAAR_NMS_IOU = 0.35

# Official OpenCV Zoo YuNet (2023-Mar) — auto-downloaded into heartbeat_ai/models/
YUNET_ONNX_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)

logger = logging.getLogger("heartbeat")

# Lazy detector; False = MediaPipe init failed permanently for this process
_mp_detector: Optional[object] = None
_cv_cascade: Optional[cv2.CascadeClassifier] = None
_cv_cascade_failed: bool = False

# YuNet: None = not tried, False = init/download failed
_yunet_detector: Optional[Any] = None
_yunet_failed: bool = False

# Which backend produced last result
_face_backend: str = "none"
_fallback_logged: bool = False
_yunet_logged: bool = False

# Populated by configure_presence() from MonitorService startup
_settings: Optional[Any] = None


def configure_presence(settings: Any) -> None:
    """Call once with app Settings so YuNet paths/thresholds match config."""

    global _settings
    _settings = settings


def face_backend() -> str:
    """Active face pipeline name (for UI / debugging)."""

    return _face_backend


def _yunet_onnx_path() -> Path:
    override = os.environ.get("HEARTBEAT_YUNET_PATH", "").strip()
    if override:
        p = Path(override)
        if p.is_file():
            return p
    if _settings is not None:
        return _settings.yunet_onnx_path()
    return Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"


def _yunet_thresholds() -> tuple[float, float]:
    if _settings is not None:
        return float(_settings.yunet_score_threshold), float(_settings.yunet_nms_threshold)
    return 0.45, 0.35


def _ensure_yunet_onnx() -> Optional[Path]:
    path = _yunet_onnx_path()
    if path.is_file() and path.stat().st_size > 10_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading YuNet face model (~1.3 MB) to %s", path)
    try:
        req = urllib.request.Request(YUNET_ONNX_URL, headers={"User-Agent": "heartbeat-ai/1.0"})
        with urllib.request.urlopen(req, timeout=180) as resp, open(path, "wb") as out:
            out.write(resp.read())
        return path if path.is_file() and path.stat().st_size > 10_000 else None
    except Exception as e:
        logger.error("YuNet ONNX download failed (%s); using Haar fallback.", e)
        return None


def _get_yunet_detector(frame_w: int, frame_h: int) -> Optional[Any]:
    """Lazy-create OpenCV FaceDetectorYN; None if unavailable."""

    global _yunet_detector, _yunet_failed
    if _yunet_failed:
        return None
    if not hasattr(cv2, "FaceDetectorYN"):
        _yunet_failed = True
        return None
    if _yunet_detector is not None:
        return _yunet_detector

    onnx = _ensure_yunet_onnx()
    if onnx is None:
        _yunet_failed = True
        return None

    score_th, nms_th = _yunet_thresholds()
    try:
        _yunet_detector = cv2.FaceDetectorYN.create(
            str(onnx),
            "",
            (max(1, frame_w), max(1, frame_h)),
            score_th,
            nms_th,
            5000,
        )
        return _yunet_detector
    except Exception:
        logger.exception("YuNet FaceDetectorYN init failed; using Haar fallback.")
        _yunet_detector = None
        _yunet_failed = True
        return None


def _detect_yunet(frame: np.ndarray) -> Optional[FaceDetectionResult]:
    """
    YuNet on BGR frame. Returns None only if YuNet cannot run (use Haar).
    """

    global _face_backend, _yunet_logged
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return FaceDetectionResult(0, [], [])

    det = _get_yunet_detector(w, h)
    if det is None:
        return None

    if not _yunet_logged:
        _yunet_logged = True
        logger.info(
            "Face detection using OpenCV YuNet (MediaPipe unavailable). "
            "Better multi-face / angle than Haar."
        )

    det.setInputSize((w, h))
    try:
        _, faces = det.detect(frame)
    except Exception:
        logger.exception("YuNet detect() failed")
        _face_backend = "opencv_yunet"
        return FaceDetectionResult(0, [], [])

    if faces is None or len(faces) == 0:
        _face_backend = "opencv_yunet"
        return FaceDetectionResult(0, [], [])

    boxes: List[Tuple[int, int, int, int]] = []
    confs: List[float] = []
    for row in np.asarray(faces):
        r = np.asarray(row, dtype=np.float64).ravel()
        if r.size < 5:
            continue
        x, y, bw, bh = float(r[0]), float(r[1]), float(r[2]), float(r[3])
        sc = float(r[-1]) if r.size >= 5 else 0.0
        xi = max(0, min(w - 1, int(round(x))))
        yi = max(0, min(h - 1, int(round(y))))
        wi = max(1, min(w - xi, int(round(bw))))
        hi = max(1, min(h - yi, int(round(bh))))
        boxes.append((xi, yi, wi, hi))
        confs.append(max(0.0, min(1.0, sc)))

    min_cf = 0.52
    if _settings is not None:
        min_cf = float(getattr(_settings, "yunet_min_face_confidence", 0.52))

    f_boxes: List[Tuple[int, int, int, int]] = []
    f_confs: List[float] = []
    for b, c in zip(boxes, confs):
        if c >= min_cf:
            f_boxes.append(b)
            f_confs.append(c)

    _face_backend = "opencv_yunet"
    return FaceDetectionResult(len(f_boxes), f_boxes, confidences=f_confs)


def _get_detector():
    """
    Build MediaPipe FaceDetection once. Returns None if MediaPipe is missing or broken.
    """

    global _mp_detector
    if _mp_detector is False:
        return None
    if _mp_detector is not None:
        return _mp_detector

    FaceDetection = None
    mp_file = "unknown"
    try:
        import mediapipe as mp

        mp_file = str(getattr(mp, "__file__", "unknown"))
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            FaceDetection = mp.solutions.face_detection.FaceDetection
    except Exception:
        pass

    if FaceDetection is None:
        try:
            fd_mod = importlib.import_module("mediapipe.solutions.face_detection")
            FaceDetection = fd_mod.FaceDetection
        except ImportError:
            try:
                import mediapipe as mp2

                mp_file = str(getattr(mp2, "__file__", mp_file))
            except Exception:
                pass
            logger.error(
                "MediaPipe face module missing (install/repair). Module: %s\n"
                "  Fix: pip uninstall mediapipe -y && pip install --no-cache-dir \"mediapipe>=0.10.14,<0.11\"\n"
                "  Using OpenCV YuNet, then Haar if needed.",
                mp_file,
            )
            _mp_detector = False
            return None

    try:
        _mp_detector = FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )
        return _mp_detector
    except Exception:
        logger.exception(
            "MediaPipe FaceDetection init failed; using YuNet / Haar. "
            "Try: pip install --no-cache-dir \"mediapipe>=0.10.14,<0.11\""
        )
        _mp_detector = False
        return None


def _get_haar_cascade() -> Optional[cv2.CascadeClassifier]:
    """Bundled with opencv-python; last-resort fallback."""

    global _cv_cascade, _cv_cascade_failed
    if _cv_cascade_failed:
        return None
    if _cv_cascade is not None:
        return _cv_cascade
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            logger.error("OpenCV Haar cascade file missing or invalid: %s", path)
            _cv_cascade_failed = True
            return None
        _cv_cascade = cascade
        return _cv_cascade
    except Exception:
        logger.exception("OpenCV Haar cascade load failed")
        _cv_cascade_failed = True
        return None


def _nms_merge_xywh(
    boxes: List[Tuple[int, int, int, int]],
    scores: Optional[List[float]] = None,
    nms_threshold: float = _OPENCV_HAAR_NMS_IOU,
) -> tuple[List[Tuple[int, int, int, int]], List[float]]:
    if not boxes:
        return [], []
    if len(boxes) == 1:
        if scores and len(scores) == 1:
            disp = _normalize_display_scores([float(scores[0])])[0]
        else:
            disp = 0.92
        return boxes, [disp]

    b = [[int(x), int(y), int(w), int(h)] for x, y, w, h in boxes]
    if scores is None or len(scores) != len(boxes):
        nms_scores = [1.0] * len(b)
    else:
        arr = np.asarray(scores, dtype=np.float64)
        nms_scores = (arr - arr.min() + 1e-3).tolist()

    idx = cv2.dnn.NMSBoxes(b, nms_scores, score_threshold=0.05, nms_threshold=nms_threshold)
    if idx is None or len(idx) == 0:
        flat = list(range(len(boxes)))
    else:
        flat = np.asarray(idx).reshape(-1).tolist()

    kept_boxes = [boxes[int(i)] for i in flat]
    kept_raw = [float(scores[int(i)]) if scores and len(scores) == len(boxes) else 1.0 for i in flat]
    kept_disp = _normalize_display_scores(kept_raw)
    return kept_boxes, kept_disp


def _normalize_display_scores(raw: List[float]) -> List[float]:
    if not raw:
        return []
    mn, mx = min(raw), max(raw)
    if mx <= mn:
        return [0.92] * len(raw)
    return [float(0.05 + 0.94 * (r - mn) / (mx - mn)) for r in raw]


def _detect_opencv_haar(frame: np.ndarray) -> FaceDetectionResult:
    global _fallback_logged, _face_backend
    cascade = _get_haar_cascade()
    if cascade is None:
        _face_backend = "none"
        return FaceDetectionResult(0, [], [])

    if not _fallback_logged:
        _fallback_logged = True
        logger.warning(
            "Face detection using OpenCV Haar (YuNet/MediaPipe unavailable). "
            "Install MediaPipe or ensure YuNet ONNX downloads for better results."
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    boxes: List[Tuple[int, int, int, int]] = []
    weights: Optional[List[float]] = None

    try:
        out = cascade.detectMultiScale3(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE,
            outputRejectLevels=True,
        )
        if out is not None and len(out) == 3:
            rects, _reject, level_weights = out
            rects = np.asarray(rects, dtype=np.int32)
            if rects.size != 0:
                if rects.ndim == 1:
                    rects = rects.reshape(1, 4)
                lw = np.asarray(level_weights, dtype=np.float64).ravel()
                for row in rects:
                    x, y, bw, bh = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    boxes.append((x, y, bw, bh))
                if len(boxes) == len(lw):
                    weights = [float(lw[i]) for i in range(len(boxes))]
    except (cv2.error, TypeError, ValueError):
        boxes = []
        weights = None

    if not boxes:
        rects = cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for x, y, bw, bh in rects:
            boxes.append((int(x), int(y), int(bw), int(bh)))
        weights = None

    boxes, confidences = _nms_merge_xywh(boxes, scores=weights)

    _face_backend = "opencv_haar"
    return FaceDetectionResult(len(boxes), boxes, confidences=confidences)


@dataclass
class FaceDetectionResult:
    face_count: int
    boxes_xywh: List[Tuple[int, int, int, int]]
    confidences: List[float] = field(default_factory=list)


def detect_faces(frame: np.ndarray) -> FaceDetectionResult:
    """
    MediaPipe when available; else OpenCV YuNet (DNN); else Haar.
    """

    global _face_backend
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return FaceDetectionResult(0, [], [])

    det = _get_detector()
    if det is None:
        yunet_res = _detect_yunet(frame)
        if yunet_res is not None:
            return yunet_res
        return _detect_opencv_haar(frame)

    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = det.process(rgb)
        if not res.detections:
            _face_backend = "mediapipe"
            return FaceDetectionResult(0, [], [])

        boxes: List[Tuple[int, int, int, int]] = []
        confs: List[float] = []
        for detection in res.detections:
            loc = detection.location_data.relative_bounding_box
            xmin = max(0, int(loc.xmin * w))
            ymin = max(0, int(loc.ymin * h))
            bw = max(0, int(loc.width * w))
            bh = max(0, int(loc.height * h))
            boxes.append((xmin, ymin, bw, bh))
            sc = 0.0
            try:
                scores = getattr(detection, "score", None)
                if scores is not None and len(scores) > 0:
                    sc = float(scores[0])
            except (TypeError, ValueError, IndexError):
                pass
            confs.append(sc)

        _face_backend = "mediapipe"
        return FaceDetectionResult(len(boxes), boxes, confidences=confs)
    except Exception:
        logger.exception("MediaPipe runtime error; switching to YuNet / Haar")
        global _mp_detector
        try:
            if _mp_detector is not None and _mp_detector is not False:
                _mp_detector.close()
        except Exception:
            pass
        _mp_detector = False
        yunet_res = _detect_yunet(frame)
        if yunet_res is not None:
            return yunet_res
        return _detect_opencv_haar(frame)


def close_detector() -> None:
    """Release detectors on shutdown."""

    global _mp_detector, _cv_cascade, _yunet_detector, _face_backend
    if _mp_detector is not None and _mp_detector is not False:
        try:
            _mp_detector.close()
        except Exception:
            logger.exception("Error closing MediaPipe face detector")
    _mp_detector = None
    _cv_cascade = None
    _yunet_detector = None
    _face_backend = "none"
