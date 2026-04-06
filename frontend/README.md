# Heartbeat AI — frontend

Static **status dashboard** (`index.html`), **browser camera → API** (`browser_camera.html`), and a small **mock webhook** server for testing exports.

Deploy this folder to any static host (Netlify, Vercel, Cloudflare Pages, S3, or `npm start` / `python -m http.server`). Point the **API base URL** in each page to your deployed backend (e.g. `https://your-api.onrender.com`).

## 1. Status dashboard (`index.html`)

Polls `GET /status`, `GET /health`, and `GET /last_anomaly`. When a phone or multi-user risk is active, shows the annotated frame from `/last_anomaly`.

### Local dev

1. Start the API from [`../backend`](../backend):

   ```bash
   cd backend
   python -m venv .venv && .venv\Scripts\activate
   pip install -r requirements.txt
   python -m heartbeat_ai.run --host 127.0.0.1 --port 8000
   ```

   For cloud-style ingest only: add `--api-only` or `HEARTBEAT_API_ONLY=1`.

2. Serve this directory:

   ```bash
   cd frontend
   npm start
   # or: python -m http.server 8080
   ```

3. Open **http://127.0.0.1:8080/** (or the port `serve` prints).

CORS is enabled on the API by default (`api_cors_enabled` in `backend/heartbeat_ai/app/config.py`).

### Browser camera (`browser_camera.html`)

For backends **without** a server webcam. Open **http://127.0.0.1:8080/browser_camera.html**, set the API URL to your deployed API, start the camera. Optional: `HEARTBEAT_BROWSER_INGEST_KEY` on the server + **X-Ingest-Key** in the page.

---

## 2. Webhook / export testing

1. Start the mock receiver:

   ```bash
   python mock_webhook_server.py
   ```

2. From `backend/`, run the API with export env vars:

   **PowerShell**

   ```powershell
   cd ..\backend
   $env:HEARTBEAT_EXPORT_ENABLED = "1"
   $env:HEARTBEAT_EXPORT_WEBHOOK_URL = "http://127.0.0.1:9000/ingest"
   python -m heartbeat_ai.run
   ```

   **bash**

   ```bash
   cd ../backend
   export HEARTBEAT_EXPORT_ENABLED=1
   export HEARTBEAT_EXPORT_WEBHOOK_URL=http://127.0.0.1:9000/ingest
   python -m heartbeat_ai.run
   ```

3. Open **http://127.0.0.1:9000/** for the webhook viewer.

4. Last payload is also written to `frontend/last_webhook.json` (gitignored).

---

## Files

| File | Role |
|------|------|
| `index.html` | Polls `/status`, `/health`, `/last_anomaly` |
| `browser_camera.html` | Webcam → `POST /ingest/frame` |
| `webhook_viewer.html` | Used by mock server |
| `mock_webhook_server.py` | `POST /ingest`, `GET /api/last` |
| `package.json` | `npm start` → static server on port 8080 |
