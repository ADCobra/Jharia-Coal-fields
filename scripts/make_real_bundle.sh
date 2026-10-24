#!/usr/bin/env bash
# Process the downloaded real Landsat scenes from scratch and copy the result to bundle/real, which
# `firewatch serve` (and the Docker image) serves read-only under /real.
set -euo pipefail
cd "$(dirname "$0")/.."

FW=".venv/bin/firewatch"
[ -x "$FW" ] || FW="firewatch"
ls data/real/scenes/*.tif >/dev/null 2>&1 || { echo "No scenes in data/real/scenes; run: bash get_real_data.sh" >&2; exit 2; }

"$FW" -c config/real.yaml real --fresh
python3 - <<'PY'
import sqlite3
con = sqlite3.connect("var/real/state.db")
con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
con.close()
PY
rm -rf bundle/real
mkdir -p bundle/real
cp -R var/real/blobs bundle/real/blobs
cp var/real/state.db bundle/real/state.db
du -sh bundle/real
echo "Bundled. firewatch serve now shows it under /real."
