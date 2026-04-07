# Heartbeat AI — backend

FastAPI service, computer-vision pipeline (MediaPipe / YuNet, YOLO phone detection), and `POST /ingest/frame` for browser-uploaded JPEGs.

## Run locally

From this directory (`backend/`):

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
pip install .
python -m heartbeat_ai.run --api-only --host 127.0.0.1 --port 8000
```

Full stack with local webcam (no `--api-only`):

```bash
python -m heartbeat_ai.run --host 127.0.0.1 --port 8000
```

## Docker

From repository root:

```bash
docker compose build backend
docker compose up backend
```

Or build only this image:

```bash
docker build -t heartbeat-api ./backend
docker run -p 8000:8000 -e PORT=8000 heartbeat-api
```

## Deploy (Render / Railway / Fly.io)

- **Root directory:** `backend` (if the platform supports it), or run commands from `backend/`.
- **Build:** `pip install -r requirements.txt && pip install .`  
  (`pip install .` registers the `heartbeat_ai` package so `python -m heartbeat_ai.run` works on Render, etc.)
- **Start:** `HEARTBEAT_API_ONLY=1 python -m heartbeat_ai.run --api-only --host 0.0.0.0 --port $PORT`  
  (Use your platform’s port variable, e.g. `$PORT` on Render/Railway.)
- **Health check path:** `/health`
- **Python:** `runtime.txt` pins 3.10.x; override with your host’s Python setting if needed.

Host the static UI from [`../frontend`](../frontend) separately (or any static host).

### PostgreSQL evidence (no local JPEG folder)

Provision a **Postgres** instance (Render/Railway/Supabase), then set:

- `HEARTBEAT_EVIDENCE_DATABASE_URL=postgresql://USER:PASS@HOST:PORT/DB?sslmode=require`

On each anomaly (phone / multi-user risk), the service stores the **annotated JPEG** and **detection JSON** in table `heartbeat_evidence`. Disable local files with `evidence_save_enabled` in config if you only want the database.

Optional: `HEARTBEAT_EVIDENCE_READ_KEY` — require header `X-Evidence-Key` on `GET /evidence/recent` and `GET /evidence/{id}/image`.

## Configuration

Environment variables are documented in the repository root [README.md](../README.md). Code defaults live in `heartbeat_ai/app/config.py`.
