#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for name in python3.13 python3.12 python3.14 python3.11 python3.10 python3.9 python3; do
  for dir in "" /opt/homebrew/bin/ /usr/local/bin/ "$HOME/.homebrew/bin/" /Library/Frameworks/Python.framework/Versions/Current/bin/; do
    if command -v "$dir$name" >/dev/null 2>&1; then PY="$dir$name"; break 2; fi
  done
done
if [ -z "$PY" ]; then
  echo "Python 3 was not found. Install it from https://www.python.org/downloads/ and run this again."
  exit 1
fi
echo "Using $("$PY" --version) ($PY)"

if [ ! -x .venv-fetch/bin/python ]; then
  "$PY" -m venv .venv-fetch
fi
.venv-fetch/bin/python -m pip install --quiet --upgrade pip
if ! .venv-fetch/bin/python -m pip install --quiet --only-binary=:all: numpy rasterio certifi; then
  echo "Could not install rasterio for this Python. Install Python 3.12 or newer from https://www.python.org/downloads/, delete the .venv-fetch folder, and run this again."
  exit 1
fi

.venv-fetch/bin/python scripts/fetch_real_data.py "$@"
echo
echo "Next step: bash run_real.sh"
