-- Runs once at first container boot (docker-entrypoint-initdb.d), as superuser.
--
-- Security Control 2 (least privilege): queryforge_agent is the ONLY role the
-- application ever connects DDL through. It cannot create roles or databases
-- and is not a superuser. It owns the dedicated `queryforge` database and the
-- TPC-H tables inside it — ownership is REQUIRED on PostgreSQL 16 for both
-- CREATE INDEX and ANALYZE (the MAINTAIN privilege only arrives in PG17).
-- Blast radius if the regex allowlist (Control 1) ever fails: the sandbox
-- database and nothing else on the server.

CREATE ROLE queryforge_agent
    LOGIN PASSWORD 'agentpw'
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;

CREATE DATABASE queryforge OWNER queryforge_agent;

-- The agent has no business in the admin database.
REVOKE CONNECT ON DATABASE postgres FROM PUBLIC;
