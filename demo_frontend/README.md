# Heartbeat AI — demo frontend

Two ways to test integration without .NET.

## 1. Status dashboard (`index.html`)

Shows live `face_count`, `status`, `phone_detected`, raw JSON from `GET /status`, and **while a phone or multi-user risk is active**, the annotated camera frame from `GET /last_anomaly` (JPEG as base64).

### Steps

1. Start the Python service **with** the HTTP API (no `--visual`):

   ```bash
   python -m heartbeat_ai.run
   ```

   Optional: `python -m heartbeat_ai.run --debug` to also see the OpenCV window.

2. Serve this folder over HTTP (browsers restrict `file://` fetches on some setups):

   ```bash
   cd demo_frontend
   python -m http.server 8080
   ```

3. Open **http://127.0.0.1:8080/**

CORS is enabled on the FastAPI app when `api_cors_enabled` is `True` (default in `heartbeat_ai/app/config.py`), so the dashboard can call the API from another port.

You can change the **API base URL** in the page (stored in `localStorage`).

---

## 2. Webhook / export testing

1. Start the mock receiver:

   ```bash
   python demo_frontend/mock_webhook_server.py
   ```

2. In another terminal, enable export and point the webhook at the mock server:

   **PowerShell**

   ```powershell
   $env:HEARTBEAT_EXPORT_ENABLED = "1"
   $env:HEARTBEAT_EXPORT_WEBHOOK_URL = "http://127.0.0.1:9000/ingest"
   python -m heartbeat_ai.run
   ```

   **bash**

   ```bash
   export HEARTBEAT_EXPORT_ENABLED=1
   export HEARTBEAT_EXPORT_WEBHOOK_URL=http://127.0.0.1:9000/ingest
   python -m heartbeat_ai.run
   ```

   Or: `python -m heartbeat_ai.run --export` and set only `HEARTBEAT_EXPORT_WEBHOOK_URL`.

3. Open **http://127.0.0.1:9000/** — viewer polls `/api/last` and shows the last JSON payload; if an anomaly POST included `image_jpeg_base64`, the annotated frame is shown.

4. The full last body is also written to `demo_frontend/last_webhook.json`.

---

## Files

| File | Role |
|------|------|
| `index.html` | Polls Heartbeat `/status` + `/health` |
| `webhook_viewer.html` | Served by mock server; shows last webhook |
| `mock_webhook_server.py` | `POST /ingest`, `GET /api/last`, `GET /` |
| `last_webhook.json` | Created after first POST (gitignored if you add it) |
