#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for name in python3.13 python3.12 python3.14; do
  for dir in "" /opt/homebrew/bin/ /usr/local/bin/ "$HOME/.homebrew/bin/" /Library/Frameworks/Python.framework/Versions/Current/bin/; do
    if command -v "$dir$name" >/dev/null 2>&1; then PY="$dir$name"; break 2; fi
  done
done
if [ -z "$PY" ]; then
  echo "Jharia Fire Watch needs Python 3.12 or newer."
  echo "Install it from https://www.python.org/downloads/ (or: brew install python@3.13) and run this again."
  exit 1
fi
echo "Using $("$PY" --version) ($PY)"

if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install --quiet --upgrade pip
if ! .venv/bin/python -m pip install --quiet -r requirements.txt; then
  echo "Retrying with a rasterio build that suits this macOS version..."
  grep -v '^rasterio' requirements.txt > .requirements-mac.txt
  .venv/bin/python -m pip install --quiet -r .requirements-mac.txt
  .venv/bin/python -m pip install --quiet --only-binary=:all: "rasterio>=1.4"
fi
.venv/bin/python -m pip install --quiet --no-deps -e .
.venv/bin/firewatch download-models

if ls data/real/scenes/*.tif >/dev/null 2>&1; then
  .venv/bin/firewatch -c config/real.yaml real "$@"
  CONFIG="-c config/real.yaml"
else
  echo "No real scenes downloaded yet (run: bash get_real_data.sh). Starting the synthetic demo instead."
  .venv/bin/firewatch demo --fresh
  CONFIG=""
fi

echo
echo "Opening http://localhost:8000 (user demo, password demo). Press Ctrl+C here to stop."
(sleep 4; open "http://localhost:8000") &
.venv/bin/firewatch $CONFIG serve
