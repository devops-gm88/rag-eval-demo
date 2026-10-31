#!/usr/bin/env bash
# Start, stop and create the workspace-local Postgres 16 + pgvector cluster.
# On a machine with Homebrew:  PG_BIN=/opt/homebrew/opt/postgresql@16/bin make db
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_BIN="${PG_BIN:-$(dirname "$(command -v pg_ctl || echo /usr/local/bin/pg_ctl)")}"
PGDATA="${PGDATA:-$ROOT/.pgdata}"
PGPORT="${PGPORT:-55432}"
PGLOG="$ROOT/.pgdata.log"
PGDATABASE="${PGDATABASE:-rag_eval}"

export PATH="$PG_BIN:$PATH"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing $1 in PG_BIN=$PG_BIN" >&2
    echo "install postgresql@16 + pgvector, then set PG_BIN" >&2
    exit 1
  }
}

case "${1:-up}" in
  init)
    need initdb
    if [ ! -f "$PGDATA/PG_VERSION" ]; then
      initdb -D "$PGDATA" -U "$(whoami)" --auth=trust --no-locale -E UTF8 >/dev/null
      echo "initialised $PGDATA"
    else
      echo "cluster already initialised"
    fi
    ;;
  start)
    need pg_ctl
    if pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
      echo "already running on port $PGPORT"
    else
      pg_ctl -D "$PGDATA" -l "$PGLOG" -o "-p $PGPORT -k /tmp" -w start >/dev/null
      echo "started on port $PGPORT"
    fi
    ;;
  create)
    need psql
    if ! psql -h 127.0.0.1 -p "$PGPORT" -d postgres -tAc \
        "SELECT 1 FROM pg_database WHERE datname='$PGDATABASE'" | grep -q 1; then
      createdb -h 127.0.0.1 -p "$PGPORT" "$PGDATABASE"
      echo "created database $PGDATABASE"
    else
      echo "database $PGDATABASE exists"
    fi
    ;;
  up)
    "$0" init
    "$0" start
    "$0" create
    psql -h 127.0.0.1 -p "$PGPORT" -d "$PGDATABASE" -tAc "SELECT version();"
    ;;
  stop)
    need pg_ctl
    pg_ctl -D "$PGDATA" -m fast -w stop >/dev/null && echo "stopped"
    ;;
  status)
    need pg_ctl
    pg_ctl -D "$PGDATA" status || true
    ;;
  *)
    echo "usage: $0 {up|init|start|create|stop|status}" >&2
    exit 2
    ;;
esac
