# App-only image: Postgres lives externally (a managed provider, e.g. Neon),
# reached via PG_AGENT_DSN. Host-agnostic by design -- the same image runs on
# Render, Fly, HF Spaces Docker, or anywhere else that can set env vars and
# expose a port, since nothing here is tied to a specific platform's Postgres
# story. See README's Deploying section for the one-time provisioning steps
# (db/init.sql, then load_data.py) run against the external database.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
# Shell form (not exec form) so ${PORT} expands: most platforms (Render, Fly)
# inject PORT and expect the app to bind it; falls back to 7860 (the app's own
# pinned default) when nothing sets it.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}
