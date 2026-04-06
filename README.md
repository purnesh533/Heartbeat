# Heartbeat AI — Office Presence & Device Monitoring

Production-oriented Python service that reads the laptop webcam, estimates presence and headcount via MediaPipe BlazeFace, detects phones/tablets with YOLOv8n (periodic inference for low CPU use), tracks idle/absence with a debounce buffer, and exposes JSON over FastAPI for a .NET (or other) heartbeat client.

For a full technical overview (architecture, face pipeline YuNet/Haar, modules, API, troubleshooting), see **[heartbeat_ai/PROJECT.md](heartbeat_ai/PROJECT.md)**.

## Requirements

- Python 3.10+
- Windows, macOS, or Linux (camera backend may differ; Windows uses DirectShow where applicable)
- First run downloads `yolov8n.pt` via Ultralytics (internet required once)

## Setup

```bash
cd heartbeat_ai
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

Run from the **parent of** the `heartbeat_ai` folder (e.g. project root `T2`):

```bash
python -m heartbeat_ai.run
```

Other useful modes:

```bash
python -m heartbeat_ai.run --debug   # HTTP API + OpenCV preview window
python -m heartbeat_ai.run --visual  # Preview only; no HTTP API
python -m heartbeat_ai.run --export  # Disk export + optional webhook (see below)
python -m heartbeat_ai.run --api-only  # No local webcam; browser sends JPEGs to POST /ingest/frame
```

Or from inside `heartbeat_ai`:

```bash
python run.py
```

## Demo frontend (browser)

The static dashboard lives in **`demo_frontend/`** at the repo root (sibling of `heartbeat_ai/`). It polls `GET /status`, `GET /health`, and `GET /last_anomaly` and shows evidence paths from `/status`.

1. **Start the API** (required — do not use `--visual`, which disables HTTP):

   ```bash
   # From repo root (parent of heartbeat_ai/)
   python -m heartbeat_ai.run
   ```

2. **Serve `demo_frontend` over HTTP** (avoids `file://` fetch restrictions):

   ```bash
   cd demo_frontend
   python -m http.server 8080
   ```

3. **Open the dashboard** in a browser: [http://127.0.0.1:8080/index.html](http://127.0.0.1:8080/index.html)

   If the API is not on `127.0.0.1:8000`, change the **API base URL** in the page (saved in `localStorage`).

4. **Browser webcam → API** (e.g. backend on Render): start the service with `--api-only` (or `HEARTBEAT_API_ONLY=1`). Open [http://127.0.0.1:8080/browser_camera.html](http://127.0.0.1:8080/browser_camera.html), set the API URL to your deployed host, allow the camera, and frames POST to `/ingest/frame`. Optional shared secret: set `HEARTBEAT_BROWSER_INGEST_KEY` on the server and the same value in the page as **X-Ingest-Key**.

Webhook / export testing (`mock_webhook_server.py`, env vars) is documented in **[demo_frontend/README.md](demo_frontend/README.md)**.

## Configuration

- Defaults live in [`heartbeat_ai/app/config.py`](heartbeat_ai/app/config.py).
- Environment overrides include: `HEARTBEAT_CAMERA_INDEX`, `HEARTBEAT_API_HOST`, `HEARTBEAT_API_PORT`, `HEARTBEAT_ABSENCE_BUFFER_SEC`, and (for export) `HEARTBEAT_EXPORT_ENABLED`, `HEARTBEAT_EXPORT_WEBHOOK_URL`, `HEARTBEAT_EXPORT_WEBHOOK_KEY`, `HEARTBEAT_EXPORT_DIR`.
- **Browser ingest:** `HEARTBEAT_API_ONLY`, `HEARTBEAT_BROWSER_INGEST` (`0`/`1`), `HEARTBEAT_BROWSER_INGEST_KEY`, `HEARTBEAT_BROWSER_INGEST_MAX_BYTES`.

Logs are written to `heartbeat.log` next to the package (under `heartbeat_ai/`). Optional SQLite events DB: `events.db`.

### Evidence JPEGs (anomaly snapshots)

When `evidence_save_enabled` is true (default), each time an annotated anomaly frame is produced (for `GET /last_anomaly` and/or export webhooks), a JPEG is written under `heartbeat_ai/` at `evidence_dir` (default `exports/evidence/`). Anomaly export and webhook JSON include an `evidence` object: `filename`, `relative_path`, `absolute_path`, and `frame_index`. `GET /status` mirrors the latest saves for dashboards (see API below). History length on `/status` is capped by `evidence_status_history_max`.

### JSON export and webhooks

With `export_enabled` (or `--export` / `HEARTBEAT_EXPORT_ENABLED`), the service writes throttled `exports/latest.json`, and on anomalies can POST JSON (and optionally embed `image_jpeg_base64`) plus write `exports/last_anomaly.json`. See `export_*` settings in config.

## API

- `GET http://127.0.0.1:8000/status` — JSON includes:
  - `face_count`, `is_present`, `phone_detected`, `status`, `last_updated` (Unix seconds).
  - `evidence` — most recent saved anomaly JPEG metadata, or `null`: `filename`, `relative_path`, `absolute_path`, `frame_index`, `saved_at_unix`.
  - `evidence_history` — same-shaped objects, newest first (up to `evidence_status_history_max`).
- `GET http://127.0.0.1:8000/health` — `{ "ok": true, "camera_ok": bool }`.
- `GET http://127.0.0.1:8000/last_anomaly` — while phone or multi-user risk is active: `available`, `image_jpeg_base64` (annotated JPEG when enabled), detection fields aligned with export JSON, and `evidence` when a file was saved. Otherwise `{ "available": false }`. Tuning: `api_last_anomaly_*` in config; set `api_last_anomaly_enabled` to disable.
- `POST http://127.0.0.1:8000/ingest/frame` — `multipart/form-data` with field **`file`** (JPEG). Optional header **`X-Ingest-Key`** if `HEARTBEAT_BROWSER_INGEST_KEY` is set. Response JSON matches export-style detection fields (`ok`, `anomaly`, `face_count`, `phones`, …). With `api_only`, `GET /status` is driven from the latest ingest. `GET /status` also includes `browser_ingest` (last request summary) when ingest has run.

CORS for browser demos is controlled by `api_cors_enabled` (default on for local demos).

## CLI

| Flag | Description |
|------|-------------|
| `--debug` | HTTP API plus OpenCV window with face / phone overlay. |
| `--visual` | OpenCV window only; no FastAPI server. |
| `--export` | Enable `exports/` JSON and optional webhook (`HEARTBEAT_EXPORT_WEBHOOK_URL`). |
| `--api-only` | No local camera thread; use browser `POST /ingest/frame` (set `HEARTBEAT_API_ONLY=1` on hosts like Render). |
| `--host`, `--port` | Override API bind address. |

## PyInstaller (single executable, no console)

From the `heartbeat_ai` directory (or pass full paths), with `PYTHONPATH` set to the **parent** of the `heartbeat_ai` package if needed:

```bash
set PYTHONPATH=..
pyinstaller --onefile --noconsole run.py
```

If the build misses assets, add collectors (after a failed run):

```bash
pyinstaller --onefile --noconsole --collect-all mediapipe --collect-all ultralytics run.py
```

## Tablet detection note

Standard COCO YOLOv8n includes **cell phone** but not **tablet**. The detector matches class names containing `phone` or `tablet` (and `cell phone`). For reliable tablet detection, consider a fine-tuned model and point `yolo_model_path` in config to your weights.

## Security

Bind the API to `127.0.0.1` by default; expose only on trusted networks if you change the host.
