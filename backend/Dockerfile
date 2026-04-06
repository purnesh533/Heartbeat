# Heartbeat AI API (browser ingest + optional local camera when run without --api-only)
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HEARTBEAT_API_ONLY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY heartbeat_ai ./heartbeat_ai

EXPOSE 8000

# Render/Railway set PORT; local default 8000
CMD ["sh", "-c", "exec python -m heartbeat_ai.run --api-only --host 0.0.0.0 --port ${PORT:-8000}"]
