# Single-container image: Postgres 16 (pinned, per CLAUDE.md) + the FastAPI
# app in one image. No persistent disk on Render's/HF's free tiers, so
# start.sh loads TPC-H fresh on every boot -- the project already treats DB
# state as disposable/reproducible (oracle.reset_indexes(), load_data.py is
# idempotent).
FROM postgres:16

# Debian bookworm's python3 is 3.11, matching the pinned interpreter version.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# db/init.sql already runs as-is via docker-compose locally (see
# docker-entrypoint-initdb.d convention); reused unchanged here.
COPY db/init.sql /docker-entrypoint-initdb.d/init.sql

ENV POSTGRES_PASSWORD=postgres
ENV PG_AGENT_DSN=postgresql://queryforge_agent:agentpw@localhost:5432/queryforge

WORKDIR /app
COPY requirements.txt .
# ponytail: --break-system-packages sidesteps Debian's PEP 668 guard rather
# than adding a venv layer -- fine for a single-purpose container image that
# runs nothing else; switch to a venv if this image ever grows other uses.
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .
RUN chmod +x start.sh

EXPOSE 7860
CMD ["./start.sh"]
