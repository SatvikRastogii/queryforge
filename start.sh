#!/bin/bash
# Single-container entrypoint: start Postgres (the postgres:16 image's own
# entrypoint logic, backgrounded), wait for the app role/db from db/init.sql
# to actually be ready, load TPC-H data fresh, then serve.
#
# Waiting on `queryforge_agent`/`queryforge` specifically (not just
# pg_isready) matters: the official postgres entrypoint starts Postgres
# once briefly to run docker-entrypoint-initdb.d/*, stops it, then starts it
# again for real -- pg_isready alone would false-positive during that first,
# temporary start.
set -euo pipefail

docker-entrypoint.sh postgres &

echo "Waiting for Postgres (queryforge_agent/queryforge) to be ready..."
until PGPASSWORD=agentpw psql -h localhost -U queryforge_agent -d queryforge -c 'SELECT 1' >/dev/null 2>&1; do
  sleep 1
done

echo "Loading TPC-H data..."
python3 load_data.py

echo "Starting QueryForge..."
# ${PORT:-7860}: Render (and similar platforms) inject PORT and expect the
# app to bind it; falls back to the project's own pinned 7860 otherwise.
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-7860}"
