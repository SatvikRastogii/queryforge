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

# Unbuffered stdout/stderr: Python fully-buffers when stdout isn't a TTY
# (always true in a container), so load_data.py's progress prints (data
# load, the 22-query smoke test) can sit invisible in a buffer for minutes
# while the process is genuinely still running -- confirmed live: a real
# deploy's log went silent right after the ANALYZE step and never showed
# "Starting QueryForge" even though nothing had crashed, just because
# nothing had flushed yet. This makes real progress visible as it happens.
ENV PYTHONUNBUFFERED=1

ENV POSTGRES_PASSWORD=postgres
# Unix socket, not TCP -- see start.sh's listen_addreses='' comment. The
# empty authority (postgresql://user:pass@/db) plus ?host=<socket dir> is
# libpq's documented URI form for a Unix-socket connection; local
# connections use trust auth by default (the image's own initdb warning
# confirms this every boot), so the password here is inert, just kept for
# clarity/consistency with the other PG_AGENT_DSN defaults in the codebase.
ENV PG_AGENT_DSN=postgresql://queryforge_agent:agentpw@/queryforge?host=/var/run/postgresql

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
