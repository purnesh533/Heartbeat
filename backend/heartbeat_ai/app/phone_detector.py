"""YOLOv8n phone / tablet (name-based) detection; CPU-friendly resize before infer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Set

import cv2
import numpy as np

logger = logging.getLogger("heartbeat")


@dataclass
class PhoneBox:
    """One detection in full-frame pixel coordinates (axis-aligned)."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str


class PhoneDetector:
    """
    Wraps ultralytics YOLOv8n. COCO includes 'cell phone' (not 'tablet').
    Model load is **lazy** so importing MonitorService does not pull torch/ultralytics
    until the first frame (avoids startup crash when NumPy/torch mismatch).

    Uses a **bottom-crop fallback** pass: phones are often small in full-frame views
    (especially with high/upward camera angles); a second pass on the lower portion
    of the frame runs at higher relative resolution.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        target_substrings: tuple[str, ...] = ("cell phone", "phone", "tablet"),
        infer_max_side: int = 640,
        conf_threshold: float = 0.18,
        bottom_crop_fallback: bool = True,
        bottom_crop_start_ratio: float = 0.52,
        side_crop_fallback: bool = True,
        side_crop_width_ratio: float = 0.38,
        filter_landscape_low_conf: bool = False,
        landscape_aspect: float = 1.80,
        landscape_min_conf: float = 0.80,
        min_box_height_frac: float = 0.10,
    ) -> None:
        self._model_path = model_path
        self._infer_max_side = infer_max_side
        self._conf_threshold = conf_threshold
        self._bottom_crop_fallback = bottom_crop_fallback
        self._bottom_crop_start_ratio = bottom_crop_start_ratio
        self._side_crop_fallback = side_crop_fallback
        self._side_crop_width_ratio = side_crop_width_ratio
        self._filter_landscape = filter_landscape_low_conf
        self._landscape_aspect = landscape_aspect
        self._landscape_min_conf = landscape_min_conf
        self._min_box_height_frac = min_box_height_frac
        self._target_substrings = tuple(s.lower() for s in target_substrings)
        self._model = None
        self._target_class_ids: Optional[Set[int]] = None
        self._load_error: Optional[str] = None
        self._load_attempted: bool = False
        self._infer_broken: bool = False

    def ensure_model_loaded(self) -> bool:
        """
        Load YOLO once (imports ultralytics/torch here). Returns True if inference is possible.
        """

        if self._load_attempted:
            return self._model is not None and bool(self._target_class_ids)
        self._load_attempted = True

        try:
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)
            names = getattr(self._model, "names", None) or {}
            self._target_class_ids = set()
            for cid, name in names.items():
                n = str(name).lower()
                if any(sub in n for sub in self._target_substrings):
                    self._target_class_ids.add(int(cid))
            if not self._target_class_ids:
                logger.warning(
                    "No YOLO classes matched %s; phone detection may be empty. "
                    "Standard COCO uses 'cell phone' only.",
                    self._target_substrings,
                )
            return self._model is not None
        except Exception as e:
            self._load_error = str(e)
            logger.exception(
                "YOLO model load failed (%s). If you see NumPy errors, run: pip install \"numpy>=1.26.2,<2\"",
                e,
            )
            self._model = None
            self._target_class_ids = set()
            return False

    @property
    def available(self) -> bool:
        return self._model is not None and bool(self._target_class_ids)

    @property
    def degraded_reason(self) -> Optional[str]:
        return self._load_error

    def _resize_for_infer(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Resize so longest side == infer_max_side; return scale applied to map boxes back."""

        h, w = frame.shape[:2]
        m = max(h, w)
        if m <= 0:
            return frame, 1.0
        scale = self._infer_max_side / float(m)
        if scale >= 1.0:
            return frame, 1.0
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return small, scale

    def _forward_to_boxes(
        self, frame: np.ndarray, y_offset: int = 0, x_offset: int = 0
    ) -> List[PhoneBox]:
        """
        Single YOLO forward pass on ``frame``.
        Returned box coordinates are shifted by ``(x_offset, y_offset)`` so they sit correctly
        inside the full original frame (used by crop-based fallback passes).
        """

        if self._model is None or not self._target_class_ids:
            return []

        orig_h, orig_w = frame.shape[:2]
        if orig_w <= 0 or orig_h <= 0:
            return []

        small, _scale = self._resize_for_infer(frame)
        sh, sw = small.shape[:2]
        sx = orig_w / float(sw)
        sy = orig_h / float(sh)

        cls_list = sorted(self._target_class_ids)

        results = self._model.predict(
            small,
            verbose=False,
            conf=self._conf_threshold,
            imgsz=self._infer_max_side,
            classes=cls_list,
        )
        if not results:
            return []
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        names = getattr(self._model, "names", None) or {}
        xyxy = r0.boxes.xyxy.cpu().numpy()
        cls_ids = r0.boxes.cls.cpu().numpy().astype(int)
        confs = r0.boxes.conf.cpu().numpy()

        out: List[PhoneBox] = []
        for i, cid in enumerate(cls_ids):
            if int(cid) not in self._target_class_ids:
                continue
            x1, y1, x2, y2 = xyxy[i]
            cf = float(confs[i])
            x1o = max(0, min(orig_w - 1, int(x1 * sx)))
            y1o = max(0, min(orig_h - 1, int(y1 * sy)))
            x2o = max(0, min(orig_w - 1, int(x2 * sx)))
            y2o = max(0, min(orig_h - 1, int(y2 * sy)))
            try:
                label = str(names[int(cid)])
            except (KeyError, TypeError):
                label = "phone"
            out.append(
                PhoneBox(
                    x1o + x_offset,
                    y1o + y_offset,
                    x2o + x_offset,
                    y2o + y_offset,
                    cf,
                    label,
                )
            )
        return out

    def _filter_false_positives(self, boxes: List[PhoneBox], frame_h: int = 0) -> List[PhoneBox]:
        """
        Per-box filters that run without face context:

        1. **Minimum height** — boxes shorter than ``min_box_height_frac × frame_h`` are
           discarded; desk clutter (keyboard, mug, remote) appears small from camera height.
        2. **Landscape / flat-object** — if width > height × landscape_aspect AND confidence is
           below landscape_min_conf, drop it. Mice seen from above are wider than tall and
           usually score lower than a real phone held in portrait.
        """

        if not boxes:
            return boxes
        min_h_px = frame_h * self._min_box_height_frac if frame_h > 0 else 0
        kept: List[PhoneBox] = []
        for p in boxes:
            bw = float(p.x2 - p.x1)
            bh = float(p.y2 - p.y1)
            if bw <= 0 or bh <= 0:
                continue
            if min_h_px > 0 and bh < min_h_px:
                continue
            if (
                self._filter_landscape
                and bw > bh * self._landscape_aspect
                and p.confidence < self._landscape_min_conf
            ):
                continue
            kept.append(p)
        return kept

    @staticmethod
    def filter_by_face_zone(
        boxes: List[PhoneBox],
        face_boxes_xywh: list,
        frame_w: int,
        frame_h: int,
        expand_factor: float = 2.5,
    ) -> List[PhoneBox]:
        """
        Spatial context filter: a phone being actively used is near the person's body.

        Each detected face box is expanded ``expand_factor``× in every direction to create
        a "person zone". A phone box is kept only if its centre falls inside at least one
        person zone. Desk objects (mouse, remote, mug) that lie far from any face are
        rejected even at decent confidence.

        Only applied when ``face_boxes_xywh`` is non-empty; skipped otherwise so phones
        can still be detected when the face is temporarily out of frame.
        """

        if not face_boxes_xywh or not boxes:
            return boxes

        zones = []
        for x, y, w, h in face_boxes_xywh:
            cx = x + w / 2.0
            cy = y + h / 2.0
            hw = w * expand_factor / 2.0
            hh = h * expand_factor / 2.0
            zones.append((
                max(0.0, cx - hw),
                max(0.0, cy - hh),
                min(float(frame_w), cx + hw),
                min(float(frame_h), cy + hh),
            ))

        kept: List[PhoneBox] = []
        for p in boxes:
            pcx = (p.x1 + p.x2) / 2.0
            pcy = (p.y1 + p.y2) / 2.0
            if any(rx1 <= pcx <= rx2 and ry1 <= pcy <= ry2 for rx1, ry1, rx2, ry2 in zones):
                kept.append(p)
        return kept

    def _nms_phone_boxes(self, boxes: List[PhoneBox], iou_thresh: float = 0.45) -> List[PhoneBox]:
        if len(boxes) <= 1:
            return boxes
        b = [[p.x1, p.y1, p.x2 - p.x1, p.y2 - p.y1] for p in boxes]
        scores = [p.confidence for p in boxes]
        idx = cv2.dnn.NMSBoxes(b, scores, score_threshold=0.01, nms_threshold=iou_thresh)
        if idx is None or len(idx) == 0:
            return boxes
        flat = np.asarray(idx).reshape(-1)
        return [boxes[int(i)] for i in flat]

    def detect_phones(self, frame: np.ndarray) -> List[PhoneBox]:
        """
        Run YOLO; return phone/tablet-class boxes in **full** frame coordinates with confidence.

        Passes (each skipped if the previous found something):
          1. Full frame — catches centred / large phones.
          2. Bottom crop — phones near the desk/lap level that are small in the full frame.
          3. Right-edge crop — phone held to the right, partially outside frame.
          4. Left-edge crop  — phone held to the left, partially outside frame.
        """

        if self._infer_broken:
            return []
        if not self.ensure_model_loaded():
            return []
        if self._model is None or not self._target_class_ids:
            return []

        fh, fw = frame.shape[:2]

        try:
            # Pass 1: full frame
            primary = self._filter_false_positives(
                self._forward_to_boxes(frame, y_offset=0), frame_h=fh
            )
            if primary:
                return self._nms_phone_boxes(primary)

            # Pass 2: bottom crop
            if self._bottom_crop_fallback:
                y0 = int(fh * self._bottom_crop_start_ratio)
                if fh - y0 >= 120:
                    crop = frame[y0:fh, :]
                    bottom = self._filter_false_positives(
                        self._forward_to_boxes(crop, y_offset=y0), frame_h=fh
                    )
                    if bottom:
                        return self._nms_phone_boxes(bottom)

            # Pass 3 & 4: side crops (phone partially outside left/right edge)
            if self._side_crop_fallback:
                x_w = int(fw * self._side_crop_width_ratio)
                if x_w >= 80:
                    # right edge
                    x0r = fw - x_w
                    right = self._filter_false_positives(
                        self._forward_to_boxes(frame[:, x0r:fw], x_offset=x0r), frame_h=fh
                    )
                    if right:
                        return self._nms_phone_boxes(right)
                    # left edge
                    left = self._filter_false_positives(
                        self._forward_to_boxes(frame[:, 0:x_w], x_offset=0), frame_h=fh
                    )
                    if left:
                        return self._nms_phone_boxes(left)

            return []

        except RuntimeError as e:
            msg = str(e).lower()
            if "numpy" in msg:
                logger.error(
                    "YOLO inference disabled: PyTorch reports NumPy is unavailable (%s). "
                    "Run: pip install \"numpy>=1.26.2,<2\" then restart (or upgrade PyTorch).",
                    e,
                )
            else:
                logger.exception("YOLO inference failed")
            self._infer_broken = True
            return []
        except Exception:
            logger.exception("YOLO inference failed")
            self._infer_broken = True
            return []

    def detect(self, frame: np.ndarray) -> bool:
        """Return True if any target class detected."""

        return len(self.detect_phones(frame)) > 0
