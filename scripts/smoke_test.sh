#!/bin/sh
# End-to-end check of a running MemBrain: store, search, upload a document and wait for it to be indexed.
# Usage: scripts/smoke_test.sh [base_url]
set -eu
BASE="${1:-http://localhost:8000}"

json() { python3 -c "import json, sys; print(json.load(sys.stdin)$1)"; }

echo "health: $(curl -fsS "$BASE/health")"

curl -fsS -X POST "$BASE/api/memories" -H 'Content-Type: application/json' \
    -d '{"title":"Smoke test outage","body":"The checkout outage was caused by an expired TLS certificate.","tags":["smoke"]}' >/dev/null
top=$(curl -fsS -X POST "$BASE/api/search" -H 'Content-Type: application/json' \
    -d '{"query":"why did checkout go down"}' | json '["results"][0]["title"]')
[ "$top" = "Smoke test outage" ] || { echo "search returned '$top'"; exit 1; }
echo "search: ok"

printf 'Smoke test SOP\n\nRotate the staging database credentials every quarter.\n' > /tmp/smoke.md
job=$(curl -fsS -F "file=@/tmp/smoke.md" "$BASE/api/documents" | json '["id"]')
for _ in $(seq 1 60); do
    status=$(curl -fsS "$BASE/api/documents/jobs/$job" | json '["status"]')
    [ "$status" = "succeeded" ] && break
    [ "$status" = "failed" ] && { curl -fsS "$BASE/api/documents/jobs/$job"; exit 1; }
    sleep 1
done
[ "$status" = "succeeded" ] || { echo "ingestion job stuck in '$status'"; exit 1; }
echo "ingestion: ok"
echo "smoke test passed"
