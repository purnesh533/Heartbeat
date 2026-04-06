"""
Periodic JSON export + optional webhook POST.

- Normal operation: write ``latest.json`` (throttled) and optionally POST JSON only
  (no image) on an interval.
- Anomaly: ``phone_detected`` or ``high_risk`` — POST JSON with optional
  ``image_jpeg_base64`` (annotated frame) plus save ``last_anomaly.json``.

Downstream can decode base64 → JPEG bytes. Standard pattern for JSON APIs.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

import cv2
import numpy as np

from .config import Settings
from .phone_detector import PhoneBox
from .presence import FaceDetectionResult

if TYPE_CHECKING:
    from .evidence_store import EvidenceStore

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
                img, text, (x + ox, y + oy), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA,
            )
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def render_annotated_frame(
    frame_bgr: np.ndarray,
    face_res: FaceDetectionResult,
    phone_boxes: List[PhoneBox],
    liveness_scores: List[float],
    spoof_filtered: int,
    *,
    status: str,
    present: bool,
    high_risk: bool,
    face_engine: str,
    anti_spoof_name: str,
) -> np.ndarray:
    """Copy of HUD drawing logic without OpenCV window (for export / base64)."""

    vis = frame_bgr.copy()
    confs = face_res.confidences
    for i, (x, y, bw, bh) in enumerate(face_res.boxes_xywh):
        color = (0, 0, 255) if high_risk else (0, 255, 0)
        x2, y2 = x + bw, y + bh
        cf = confs[i] if i < len(confs) else 0.0
        live = liveness_scores[i] if i < len(liveness_scores) else None
        label = f"Face  live:{live:.2f}" if live is not None else "Face"
        cv2.rectangle(vis, (x, y), (x2, y2), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.52, 1
        (tw, thh), bl = cv2.getTextSize(label, font, fs, th)
        pad = 4
        ty2, ty1 = y, max(0, y - thh - bl - 2 * pad)
        tx2 = min(vis.shape[1] - 1, x + tw + 2 * pad)
        cv2.rectangle(vis, (x, ty1), (tx2, ty2), (0, 0, 0), -1)
        cv2.rectangle(vis, (x, ty1), (tx2, ty2), color, 1)
        _put_text_outline(vis, label, (x + pad, ty2 - pad), font, fs, color, th, 2)
        txt = f"{max(0.0, min(1.0, cf)):.2f}"
        scale, thick = 0.62, 2
        (tw2, th2), bl2 = cv2.getTextSize(txt, font, scale, thick)
        cx = (x + x2) // 2
        yb = min(vis.shape[0] - 2, y2 + th2 + bl2 + 12)
        xa = max(2, cx - tw2 // 2 - 8)
        xb = min(vis.shape[1] - 2, xa + tw2 + 16)
        ya = max(0, yb - th2 - bl2 - 10)
        cv2.rectangle(vis, (xa, ya), (xb, yb), (0, 0, 0), -1)
        cv2.rectangle(vis, (xa, ya), (xb, yb), color, 1)
        _put_text_outline(vis, txt, (xa + 8, yb - 5), font, scale, color, thick, 2)

    red = (0, 0, 255)
    for pb in phone_boxes:
        short = pb.label if len(pb.label) <= 22 else pb.label[:19] + "..."
        cv2.rectangle(vis, (pb.x1, pb.y1), (pb.x2, pb.y2), red, 2)
        (tw, thh), bl = cv2.getTextSize(short, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        pad = 4
        ty2, ty1 = pb.y1, max(0, pb.y1 - thh - bl - 2 * pad)
        tx2 = min(vis.shape[1] - 1, pb.x1 + tw + 2 * pad)
        cv2.rectangle(vis, (pb.x1, ty1), (tx2, ty2), (0, 0, 0), -1)
        cv2.rectangle(vis, (pb.x1, ty1), (tx2, ty2), red, 1)
        _put_text_outline(vis, short, (pb.x1 + pad, ty2 - pad), cv2.FONT_HERSHEY_SIMPLEX, 0.52, red, 1, 2)
        cf = max(0.0, min(1.0, pb.confidence))
        txt = f"{cf:.2f}"
        scale, thick = 0.62, 2
        (tw2, th2), bl2 = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cx = (pb.x1 + pb.x2) // 2
        yb = min(vis.shape[0] - 2, pb.y2 + th2 + bl2 + 12)
        xa = max(2, cx - tw2 // 2 - 8)
        xb2 = min(vis.shape[1] - 2, xa + tw2 + 16)
        ya = max(0, yb - th2 - bl2 - 10)
        cv2.rectangle(vis, (xa, ya), (xb2, yb), (0, 0, 0), -1)
        cv2.rectangle(vis, (xa, ya), (xb2, yb), red, 1)
        _put_text_outline(vis, txt, (xa + 8, yb - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, red, thick, 2)

    fmax = max(face_res.confidences) if face_res.confidences else 0.0
    pmax = max((pb.confidence for pb in phone_boxes), default=0.0)
    spoof_tag = f"  |  spoof blocked: {spoof_filtered}" if spoof_filtered > 0 else ""
    lines = [
        f"Status: {status}",
        f"Present: {present}  |  faces: {face_res.face_count}{spoof_tag}  |  phone: {len(phone_boxes) > 0}",
        f"Face engine: {face_engine}  |  max face conf: {fmax:.2f}  |  max phone conf: {pmax:.2f}  |  anti-spoof: {anti_spoof_name}",
    ]
    y0, lh = 28, 26
    col = (0, 165, 255) if high_risk else (0, 255, 255)
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (10, y0 + i * lh), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    return vis


def _jpeg_base64(bgr: np.ndarray, quality: int) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok or buf is None:
        return ""
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def save_evidence_to_disk(
    annotated_bgr: np.ndarray,
    directory: Path,
    frame_index: int,
    jpeg_quality: int,
    package_root: Path,
) -> Optional[Dict[str, Any]]:
    """Write annotated frame as JPEG; return metadata for JSON ``evidence`` field."""
    directory.mkdir(parents=True, exist_ok=True)
    ms = int(time.time() * 1000)
    name = f"evidence_{ms}_{frame_index}.jpg"
    path = directory / name
    if not cv2.imwrite(
        str(path),
        annotated_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    ):
        logger.warning("evidence: cv2.imwrite failed %s", path)
        return None
    try:
        rel = str(path.resolve().relative_to(package_root.resolve())).replace("\\", "/")
    except ValueError:
        rel = str(path)
    return {
        "filename": name,
        "relative_path": rel,
        "absolute_path": str(path.resolve()),
        "frame_index": int(frame_index),
    }


def build_detection_payload(
    *,
    frame_index: int,
    face_res: FaceDetectionResult,
    phone_boxes: List[PhoneBox],
    liveness_scores: List[float],
    spoof_filtered: int,
    snap: Dict[str, Any],
    face_engine: str,
    anti_spoof_name: str,
    include_image_b64: bool,
    frame_bgr: Optional[np.ndarray],
    jpeg_quality: int,
    save_evidence_to: Optional[Path] = None,
    evidence_package_root: Optional[Path] = None,
    evidence_store: Optional["EvidenceStore"] = None,
    evidence_source: str = "camera",
) -> Dict[str, Any]:
    reasons: List[str] = []
    if snap.get("phone_detected"):
        reasons.append("phone_detected")
    if snap.get("high_risk"):
        reasons.append("high_risk")

    faces_out: List[Dict[str, Any]] = []
    for i, (x, y, w, h) in enumerate(face_res.boxes_xywh):
        faces_out.append(
            {
                "xywh": [int(x), int(y), int(w), int(h)],
                "confidence": float(face_res.confidences[i]) if i < len(face_res.confidences) else 0.0,
                "liveness": float(liveness_scores[i]) if i < len(liveness_scores) else None,
            }
        )

    phones_out: List[Dict[str, Any]] = []
    for pb in phone_boxes:
        phones_out.append(
            {
                "xyxy": [int(pb.x1), int(pb.y1), int(pb.x2), int(pb.y2)],
                "confidence": float(pb.confidence),
                "label": pb.label,
            }
        )

    anomaly = len(reasons) > 0
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "timestamp_unix": time.time(),
        "frame_index": int(frame_index),
        "anomaly": anomaly,
        "anomaly_reasons": reasons,
        "status": str(snap.get("status", "")),
        "face_count": int(face_res.face_count),
        "is_present": bool(snap.get("is_present", False)),
        "phone_detected": bool(snap.get("phone_detected", False)),
        "high_risk": bool(snap.get("high_risk", False)),
        "faces": faces_out,
        "phones": phones_out,
        "spoof_filtered_count": int(spoof_filtered),
        "face_engine": face_engine,
        "anti_spoof_engine": anti_spoof_name,
    }

    can_save_evidence = bool(
        save_evidence_to is not None
        and anomaly
        and evidence_package_root is not None
    )
    save_to_db = bool(
        anomaly
        and frame_bgr is not None
        and evidence_store is not None
        and getattr(evidence_store, "ready", False)
    )
    if frame_bgr is not None and (include_image_b64 or can_save_evidence or save_to_db):
        ann = render_annotated_frame(
            frame_bgr,
            face_res,
            phone_boxes,
            liveness_scores,
            spoof_filtered,
            status=str(snap.get("status", "")),
            present=bool(snap.get("is_present", False)),
            high_risk=bool(snap.get("high_risk", False)),
            face_engine=face_engine,
            anti_spoof_name=anti_spoof_name,
        )
        if can_save_evidence and save_evidence_to is not None and evidence_package_root is not None:
            ev = save_evidence_to_disk(
                ann,
                save_evidence_to,
                frame_index,
                jpeg_quality,
                evidence_package_root,
            )
            if ev:
                payload["evidence"] = ev
        if save_to_db and evidence_store is not None:
            ok_enc, buf = cv2.imencode(
                ".jpg",
                ann,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            if ok_enc and buf is not None:
                row_id = evidence_store.insert(
                    jpeg_bytes=buf.tobytes(),
                    payload=payload,
                    evidence_source=evidence_source,
                )
                if row_id is not None:
                    ev_out: Dict[str, Any] = dict(payload.get("evidence") or {})
                    ev_out["database_id"] = row_id
                    ev_out["storage"] = "postgres"
                    payload["evidence"] = ev_out
        if include_image_b64:
            b64 = _jpeg_base64(ann, jpeg_quality)
            if b64:
                payload["image_mime"] = "image/jpeg"
                payload["image_jpeg_base64"] = b64

    return payload


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _post_json(url: str, payload: Dict[str, Any], api_key: str, timeout_sec: float) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    if api_key:
        req.add_header("X-API-Key", api_key)
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        resp.read()


class AnomalyExporter:
    """
    Throttled disk export + optional HTTP webhook.

    See ``Settings`` fields prefixed with ``export_``.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        on_anomaly_payload: Optional[
            Union[
                Callable[[Dict[str, Any]], None],
                Callable[[Dict[str, Any], str], None],
            ]
        ] = None,
        evidence_store: Optional["EvidenceStore"] = None,
    ) -> None:
        self._s = settings
        self._on_anomaly_payload = on_anomaly_payload
        self._evidence_store = evidence_store
        self._last_disk_write: float = 0.0
        self._last_normal_http: float = 0.0
        self._last_anomaly_http: float = 0.0
        self._prev_anomaly: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self._s.export_enabled)

    def _export_dir(self) -> Path:
        p = Path(self._s.export_dir)
        if not p.is_absolute():
            return Path(__file__).resolve().parent.parent / p
        return p

    def _emit_anomaly(self, payload: Dict[str, Any], evidence_source: str) -> None:
        cb = self._on_anomaly_payload
        if cb is None:
            return
        try:
            cb(payload, evidence_source)  # type: ignore[misc]
        except TypeError:
            cb(payload)  # type: ignore[misc]

    def tick(
        self,
        *,
        frame_bgr: np.ndarray,
        face_res: FaceDetectionResult,
        phone_boxes: List[PhoneBox],
        liveness_scores: List[float],
        spoof_filtered: int,
        snap: Dict[str, Any],
        frame_index: int,
        face_engine: str,
        anti_spoof_name: str,
        now_mono: float,
        evidence_source: str = "camera",
    ) -> None:
        if not self._s.export_enabled:
            return

        anomaly = bool(snap.get("phone_detected") or snap.get("high_risk"))
        base_kwargs = dict(
            frame_index=frame_index,
            face_res=face_res,
            phone_boxes=phone_boxes,
            liveness_scores=liveness_scores,
            spoof_filtered=spoof_filtered,
            snap=snap,
            face_engine=face_engine,
            anti_spoof_name=anti_spoof_name,
        )
        ev_dir = self._s.evidence_dir_path() if self._s.evidence_save_enabled else None
        ev_root = self._s.models_dir().parent if ev_dir else None
        ev_db = self._evidence_store if self._evidence_store and self._evidence_store.ready else None

        # JSON-only payload (never embed image) for periodic disk + normal HTTP
        payload_json_only = build_detection_payload(
            **base_kwargs,
            include_image_b64=False,
            frame_bgr=None,
            jpeg_quality=self._s.export_jpeg_quality,
        )

        # Throttled: latest.json on disk
        if now_mono - self._last_disk_write >= self._s.export_disk_min_interval_sec:
            self._last_disk_write = now_mono
            try:
                _write_json(self._export_dir() / "latest.json", payload_json_only)
            except Exception:
                logger.exception("export: failed to write latest.json")

        url = (self._s.export_webhook_url or "").strip()
        if url:
            try:
                if anomaly:
                    send_now = (not self._prev_anomaly) or (
                        now_mono - self._last_anomaly_http >= self._s.export_anomaly_http_cooldown_sec
                    )
                    if send_now:
                        if self._s.export_image_on_anomaly:
                            payload_full = build_detection_payload(
                                **base_kwargs,
                                include_image_b64=True,
                                frame_bgr=frame_bgr,
                                jpeg_quality=self._s.export_jpeg_quality,
                                save_evidence_to=ev_dir,
                                evidence_package_root=ev_root,
                                evidence_store=ev_db,
                                evidence_source=evidence_source,
                            )
                            _post_json(
                                url,
                                payload_full,
                                self._s.export_webhook_api_key.strip(),
                                self._s.export_http_timeout_sec,
                            )
                            try:
                                _write_json(self._export_dir() / "last_anomaly.json", payload_full)
                            except Exception:
                                logger.exception("export: failed to write last_anomaly.json")
                            self._emit_anomaly(payload_full, evidence_source)
                        else:
                            payload_anomaly = build_detection_payload(
                                **base_kwargs,
                                include_image_b64=False,
                                frame_bgr=frame_bgr,
                                jpeg_quality=self._s.export_jpeg_quality,
                                save_evidence_to=ev_dir,
                                evidence_package_root=ev_root,
                                evidence_store=ev_db,
                                evidence_source=evidence_source,
                            )
                            _post_json(
                                url,
                                payload_anomaly,
                                self._s.export_webhook_api_key.strip(),
                                self._s.export_http_timeout_sec,
                            )
                            try:
                                _write_json(self._export_dir() / "last_anomaly.json", payload_anomaly)
                            except Exception:
                                logger.exception("export: failed to write last_anomaly.json")
                            self._emit_anomaly(payload_anomaly, evidence_source)
                        self._last_anomaly_http = now_mono
                elif now_mono - self._last_normal_http >= self._s.export_normal_http_interval_sec:
                    _post_json(
                        url,
                        payload_json_only,
                        self._s.export_webhook_api_key.strip(),
                        self._s.export_http_timeout_sec,
                    )
                    self._last_normal_http = now_mono
            except urllib.error.HTTPError as exc_http:
                logger.warning("export: webhook HTTP %s %s", exc_http.code, exc_http.reason)
            except urllib.error.URLError as exc_url:
                logger.warning("export: webhook URL error %s", exc_url.reason)
            except Exception:
                logger.exception("export: webhook POST failed")

        self._prev_anomaly = anomaly
