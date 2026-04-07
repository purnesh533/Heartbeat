# Heartbeat AI — monorepo

Office presence and device monitoring: **backend** (Python, FastAPI, MediaPipe, YOLO) and **frontend** (static dashboard + browser camera client).

| Directory | Role |
|-----------|------|
| [`backend/`](backend/) | API and CV pipeline (`heartbeat_ai` package). |
| [`frontend/`](frontend/) | Static HTML (`index.html`, `browser_camera.html`). |

Deep dive: [`backend/heartbeat_ai/PROJECT.md`](backend/heartbeat_ai/PROJECT.md).

## Quick start (local)

**Backend** (from `backend/`):

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
pip install .
python -m heartbeat_ai.run --host 127.0.0.1 --port 8000
```

- **`--api-only`** — No local webcam; use `frontend/browser_camera.html` to send frames to `POST /ingest/frame`.
- **`--debug`** — HTTP API + OpenCV preview.
- **`--visual`** — Preview only; no HTTP API.

**Frontend** (from `frontend/`):

```bash
cd frontend
npm start
# or: python -m http.server 8080
```

Open [http://127.0.0.1:8080/](http://127.0.0.1:8080/). Set the **API base URL** to `http://127.0.0.1:8000` (saved in `localStorage`).

### Install from repository root

```bash
pip install -r requirements.txt
cd backend && pip install .
```

This includes [`backend/requirements.txt`](backend/requirements.txt) via `-r`. From `backend/`, run **`pip install .`** once so `python -m heartbeat_ai.run` resolves. Then:

```bash
cd backend && python -m heartbeat_ai.run --api-only --host 127.0.0.1 --port 8000
```

## Docker

From repository root:

```bash
docker compose up --build
```

- API: [http://127.0.0.1:8000](http://127.0.0.1:8000) (`/health`, `/status`, `/docs`)
- Static UI: [http://127.0.0.1:8080](http://127.0.0.1:8080)

## Deploy

### Render (Blueprint)

This repo includes [`render.yaml`](render.yaml): **root directory `backend`**, start command with `--api-only` and `$PORT`.

Manual settings:

- **Root directory:** `backend`
- **Build:** `pip install -r requirements.txt && pip install .`
- **Start:** `HEARTBEAT_API_ONLY=1 python -m heartbeat_ai.run --api-only --host 0.0.0.0 --port $PORT`
- **Health check:** `/health`

### Railway / Fly.io / other

Same pattern: build and start from **`backend/`**, bind **`0.0.0.0`**, use the platform port variable. See [`backend/README.md`](backend/README.md).

### Frontend

Deploy **`frontend/`** as a static site (Netlify, Vercel, Cloudflare Pages, S3, or the `frontend` service in `docker-compose.yml`). Point the UI’s API base URL at your public API URL.

## Requirements

- Python **3.10+** (`backend/runtime.txt` pins 3.10.x for hosts that read it)
- First run downloads **`yolov8n.pt`** (Ultralytics; network once)

## Configuration

- Defaults: [`backend/heartbeat_ai/app/config.py`](backend/heartbeat_ai/app/config.py)
- **Camera / API:** `HEARTBEAT_CAMERA_INDEX`, `HEARTBEAT_API_HOST`, `HEARTBEAT_API_PORT`, `HEARTBEAT_ABSENCE_BUFFER_SEC`
- **Browser ingest:** `HEARTBEAT_API_ONLY`, `HEARTBEAT_BROWSER_INGEST`, `HEARTBEAT_BROWSER_INGEST_KEY`, `HEARTBEAT_BROWSER_INGEST_MAX_BYTES`
- **Evidence database (PostgreSQL):** `HEARTBEAT_EVIDENCE_DATABASE_URL` — stores each anomaly as annotated JPEG + JSON (no PC disk needed). Optional `HEARTBEAT_EVIDENCE_READ_KEY` — then use header `X-Evidence-Key` on `GET /evidence/recent` and `GET /evidence/{id}/image`.
- **Export / webhook:** `HEARTBEAT_EXPORT_ENABLED`, `HEARTBEAT_EXPORT_WEBHOOK_URL`, `HEARTBEAT_EXPORT_WEBHOOK_KEY`, `HEARTBEAT_EXPORT_DIR`

Logs and SQLite default to files under `backend/heartbeat_ai/`.

## API (summary)

- **`GET /`** — Service index JSON
- **`GET /status`** — `face_count`, `phone_detected`, `evidence` (may include `database_id` / `storage: postgres`), `browser_ingest`, …
- **`GET /health`** — `ok`, `camera_ok`, `api_only`
- **`GET /last_anomaly`** — Annotated frame (base64) while risk active
- **`POST /ingest/frame`** — Multipart JPEG (`file`); optional `X-Ingest-Key`
- **`GET /evidence/recent?limit=30`** — Recent PostgreSQL evidence rows (when `HEARTBEAT_EVIDENCE_DATABASE_URL` is set). Optional `X-Evidence-Key` if `HEARTBEAT_EVIDENCE_READ_KEY` is set.
- **`GET /evidence/{id}/image`** — Stored annotated JPEG for that row (same auth rules as above).

## CLI flags

| Flag | Description |
|------|-------------|
| `--debug` | API + OpenCV overlay window |
| `--visual` | OpenCV only; no API |
| `--export` | Enable disk export + optional webhook |
| `--api-only` | No local camera; browser ingest |
| `--host`, `--port` | API bind address |

## PyInstaller

From **`backend/`**, with `PYTHONPATH` including the current directory:

```bash
cd backend
set PYTHONPATH=.
pyinstaller --onefile --noconsole heartbeat_ai/run.py
```

Add `--collect-all mediapipe --collect-all ultralytics` if assets are missing.

## Security

Default API bind is loopback. For public deploys, use TLS, consider `HEARTBEAT_BROWSER_INGEST_KEY`, and tighten CORS (`api_cors_enabled`) when not using the open demo.

## Tablet detection note

COCO YOLOv8n includes **cell phone** but not **tablet** by default. The detector matches name substrings; for tablets, use custom weights and `yolo_model_path` in config.
