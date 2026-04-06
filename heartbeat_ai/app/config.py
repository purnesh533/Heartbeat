"""Central configuration for Heartbeat AI (override via attributes or env in run.py)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    """Application settings; all tunable parameters live here."""

    # Camera
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    queue_max_size: int = 5
    camera_open_retry_initial_sec: float = 0.5
    camera_open_retry_max_sec: float = 8.0
    camera_read_fail_threshold: int = 15  # consecutive failures before reconnect

    # Presence / idle
    absence_buffer_sec: float = 5.0
    # OpenCV YuNet (DNN) when MediaPipe missing — better multi-face / angle than Haar
    yunet_onnx_name: str = "face_detection_yunet_2023mar.onnx"
    yunet_score_threshold: float = 0.45
    yunet_nms_threshold: float = 0.35
    # Drop weak YuNet hits (e.g. face-like texture on a mouse) — does not apply to MediaPipe/Haar
    yunet_min_face_confidence: float = 0.52

    # YOLO (phone / tablet)
    yolo_model_path: str = "yolov8n.pt"
    yolo_every_n_frames: int = 5  # more frequent checks for small/partial phones
    yolo_infer_max_side: int = 640  # higher = better small objects (more CPU)
    # Lower threshold so partial/edge/angled phones are still detected.
    # Face-zone filter is the primary false-positive defence.
    yolo_conf_threshold: float = 0.18
    yolo_bottom_crop_fallback: bool = True   # pass 2: bottom strip (waist/desk level phones)
    yolo_bottom_crop_start_ratio: float = 0.52
    yolo_side_crop_fallback: bool = True     # pass 3/4: left & right edge strips
    yolo_side_crop_width_ratio: float = 0.38  # fraction of frame width used for each side crop
    # Landscape filter off — side-held or angled phones can appear wider-than-tall.
    # Face-zone (below) is the primary FP defence; landscape would cause false negatives.
    yolo_phone_filter_landscape: bool = False
    yolo_phone_landscape_aspect: float = 1.80
    yolo_phone_landscape_min_conf: float = 0.80
    # Minimum box height relative to frame — skips tiny distant/clutter detections.
    yolo_phone_min_height_frac: float = 0.10
    # Face-zone: phone centre must fall inside expand_factor× the face bounding box.
    # Set to 0 to disable (needed when phone is at frame edge and face is centre).
    yolo_phone_face_zone_expand: float = 0.0
    # COCO: cell phone = 67; tablet is not in COCO — we also match name substrings
    yolo_target_class_substrings: tuple[str, ...] = ("cell phone", "phone", "tablet")

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Allow browser demos (demo_frontend) to call GET /status from another origin.
    # Set False in locked-down production.
    api_cors_enabled: bool = True
    # GET /last_anomaly — annotated JPEG (base64) while phone or multi-user risk is active.
    api_last_anomaly_enabled: bool = True
    api_last_anomaly_min_interval_sec: float = 1.0
    api_last_anomaly_jpeg_quality: int = 82
    # Save annotated anomaly frames to disk; JSON includes ``evidence`` (paths + filename).
    evidence_save_enabled: bool = True
    evidence_dir: str = "exports/evidence"
    # Max entries returned on GET /status (``evidence_history`` + ``evidence``).
    evidence_status_history_max: int = 30

    # Browser → POST JPEG to ``POST /ingest/frame`` (e.g. Render; no server webcam).
    browser_ingest_enabled: bool = True
    browser_ingest_api_key: str = ""
    browser_ingest_max_image_bytes: int = 6_000_000
    # No local camera thread (``HEARTBEAT_API_ONLY=1`` or ``--api-only``).
    api_only: bool = False

    # Paths (relative to package root heartbeat_ai/)
    log_file: str = "heartbeat.log"
    sqlite_db: str = "events.db"

    # Silent-Face Anti-Spoofing (rejects photos, screens, printed posters)
    anti_spoof_enabled: bool = True
    # Liveness score below this threshold → face classified as spoof and removed.
    # Range 0–1.  Lower = less strict (fewer real-face rejections).
    anti_spoof_threshold: float = 0.55

    # Background face size filter — secondary faces whose bounding-box area is
    # less than this fraction of the largest face are treated as background
    # (posters, screens, people far away) and excluded from the face count.
    # Set to 0.0 to disable.  0.50 means a secondary face must be at least
    # half the area of the primary face to count as a real additional person.
    face_size_ratio_min: float = 0.50

    # Risk
    high_risk_on_multi_user: bool = True

    # SQLite throttle (max rows per second for periodic snapshots)
    db_insert_min_interval_sec: float = 1.0

    # Export / downstream webhook (JSON; base64 JPEG only on anomaly when enabled)
    export_enabled: bool = False
    # Relative to heartbeat_ai/ unless absolute
    export_dir: str = "exports"
    export_disk_min_interval_sec: float = 2.0
    # POST target; empty = disk-only (latest.json still written if export_enabled)
    export_webhook_url: str = ""
    export_webhook_api_key: str = ""
    export_http_timeout_sec: float = 15.0
    # When not in anomaly: POST JSON-only at most this often
    export_normal_http_interval_sec: float = 30.0
    # Anomaly = phone_detected OR high_risk; POST (with image if enabled) at most this often
    export_anomaly_http_cooldown_sec: float = 8.0
    export_image_on_anomaly: bool = True
    export_jpeg_quality: int = 85

    def log_path(self) -> Path:
        return _root() / self.log_file

    def db_path(self) -> Path:
        return _root() / self.sqlite_db

    def yolo_path(self) -> Path:
        """Resolve YOLO weights relative to package root (heartbeat_ai/)."""

        p = Path(self.yolo_model_path)
        if p.is_absolute():
            return p
        return _root() / p

    def models_dir(self) -> Path:
        return _root() / "models"

    def evidence_dir_path(self) -> Path:
        """Directory for saved anomaly JPEG evidence files."""
        p = Path(self.evidence_dir)
        if p.is_absolute():
            return p
        return _root() / p

    def yunet_onnx_path(self) -> Path:
        return self.models_dir() / self.yunet_onnx_name


def settings_from_env() -> Settings:
    """Build Settings with optional environment overrides."""

    s = Settings()
    if v := os.environ.get("HEARTBEAT_CAMERA_INDEX"):
        s.camera_index = int(v)
    if v := os.environ.get("HEARTBEAT_API_HOST"):
        s.api_host = v
    if v := os.environ.get("HEARTBEAT_API_PORT"):
        s.api_port = int(v)
    if v := os.environ.get("HEARTBEAT_ABSENCE_BUFFER_SEC"):
        s.absence_buffer_sec = float(v)
    if os.environ.get("HEARTBEAT_EXPORT_ENABLED", "").strip().lower() in ("1", "true", "yes"):
        s.export_enabled = True
    if v := os.environ.get("HEARTBEAT_EXPORT_WEBHOOK_URL", "").strip():
        s.export_webhook_url = v
    if v := os.environ.get("HEARTBEAT_EXPORT_WEBHOOK_KEY", "").strip():
        s.export_webhook_api_key = v
    if v := os.environ.get("HEARTBEAT_EXPORT_DIR", "").strip():
        s.export_dir = v

    def _truthy(name: str) -> Optional[bool]:
        raw = os.environ.get(name, "").strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        return None

    bi = _truthy("HEARTBEAT_BROWSER_INGEST")
    if bi is not None:
        s.browser_ingest_enabled = bi
    if v := os.environ.get("HEARTBEAT_BROWSER_INGEST_KEY", "").strip():
        s.browser_ingest_api_key = v
    if v := os.environ.get("HEARTBEAT_BROWSER_INGEST_MAX_BYTES", "").strip():
        try:
            s.browser_ingest_max_image_bytes = max(64_000, int(v))
        except ValueError:
            pass
    ao = _truthy("HEARTBEAT_API_ONLY")
    if ao is not None:
        s.api_only = ao
    return s
