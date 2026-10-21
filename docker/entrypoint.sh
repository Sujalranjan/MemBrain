#!/bin/sh
set -e

# Apply pending migrations before serving. Disable on extra replicas/workers so only one process migrates.
if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
    alembic upgrade head
fi

# Optionally load the sample notes into an empty workspace (used by the hosted demo).
if [ "${SEED_DEMO:-false}" = "true" ]; then
    python -m app.cli seed-demo
fi

case "$1" in
    api)
        exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips='*'
        ;;
    worker)
        exec python -m app.worker
        ;;
    *)
        exec "$@"
        ;;
esac
