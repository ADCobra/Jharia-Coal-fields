#!/usr/bin/env bash
# Runs ON a Graviton instance launched from the COOL (Cloud-Optimized OpenCV) Marketplace AMI.
# Benchmarks COOL against the opencv-contrib wheel from PyPI on the same machine:
#   1. the core OpenCV kernels of a pass (scripts/cool_bench.py), in alternating rounds;
#   2. the whole pipeline (`firewatch eval bench`) once under each build.
# Results go to $WORK/results.tgz; with PRINT_RESULTS=1 they are also printed base64-encoded between markers,
# which is how infra/cool_bench.sh collects them over SSM. To run it by hand over SSH:
#   sudo REPO_URL=https://github.com/ADCobra/Jharia-Coal-fields.git bash cool_remote.sh
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ADCobra/Jharia-Coal-fields.git}"
ROUNDS="${ROUNDS:-3}"
REPEATS="${REPEATS:-15}"
WORK="${WORK:-/opt/fwbench}"
COOL_ROOT="${COOL_ROOT:-/opt/cool}"
export HOME="${HOME:-/root}" PIP_DISABLE_PIP_VERSION_CHECK=1 DEBIAN_FRONTEND=noninteractive MPLCONFIGDIR=/tmp/mpl

log() { echo "[cool-bench] $*" >&2; }

COOL_PY="$COOL_ROOT/venvs/python_3.12/bin/python"
[ -x "$COOL_PY" ] || { log "COOL's Python 3.12 environment was not found at $COOL_PY. Is this the COOL AMI?"; exit 2; }

log "installing git and python3.12-venv"
apt-get update -qq >/dev/null
apt-get install -y -qq git python3.12-venv >/dev/null

rm -rf "$WORK" && mkdir -p "$WORK/out"
git clone -q --depth 1 "$REPO_URL" "$WORK/repo"
cd "$WORK/repo"
WHEEL="$(grep -E '^opencv-contrib-python-headless==' requirements.txt)"
COOL_NUMPY="$("$COOL_PY" -c 'import numpy; print(numpy.__version__)')"
log "COOL: $("$COOL_PY" -c 'import cv2; print(cv2.__version__, cv2.__file__)'), numpy $COOL_NUMPY"

log "making a venv with $WHEEL"
python3.12 -m venv "$WORK/pip"
"$WORK/pip/bin/pip" install -q "$WHEEL" "numpy==$COOL_NUMPY" || "$WORK/pip/bin/pip" install -q "$WHEEL"
log "wheel: $("$WORK/pip/bin/python" -c 'import cv2; print(cv2.__version__, cv2.__file__)')"

log "kernel benchmark, $ROUNDS alternating rounds"
for r in $(seq 1 "$ROUNDS"); do
  "$COOL_PY" scripts/cool_bench.py --label cool --repeats "$REPEATS" --out "$WORK/out/cool-$r.json"
  "$WORK/pip/bin/python" scripts/cool_bench.py --label pip --repeats "$REPEATS" --out "$WORK/out/pip-$r.json"
done

# The whole pipeline: Fire Watch installed twice, once importing COOL's cv2 and once the wheel's.
pipeline() {
  local name="$1" venv="$WORK/fw-$1"
  python3.12 -m venv "$venv"
  if [ "$name" = cool ]; then
    grep -vE '^(opencv-contrib-python-headless|numpy)==' requirements.txt > "$WORK/req-cool.txt"
    "$venv/bin/pip" install -q "numpy==$COOL_NUMPY" -r "$WORK/req-cool.txt"
  else
    "$venv/bin/pip" install -q -r requirements.txt
  fi
  "$venv/bin/pip" install -q --no-deps .
  local env=()
  if [ "$name" = cool ]; then
    # Expose only COOL's cv2 package, so the rest of COOL's site-packages cannot shadow Fire Watch's dependencies.
    mkdir -p "$WORK/coolpath"
    ln -sfn "$("$COOL_PY" -c 'import cv2, os; print(os.path.dirname(cv2.__file__))')" "$WORK/coolpath/cv2"
    env=(env "LD_LIBRARY_PATH=$COOL_ROOT/cpp_sdk/lib:${LD_LIBRARY_PATH:-}" "PYTHONPATH=$WORK/coolpath")
  fi
  "${env[@]}" "$venv/bin/python" -c "import cv2; print('$name pipeline uses', cv2.__version__, cv2.__file__)" >&2
  "${env[@]}" "$venv/bin/firewatch" download-models --dest models >/dev/null
  "${env[@]}" "$venv/bin/firewatch" eval bench --out "$WORK/out/pipeline-$name"
}
for name in cool pip; do
  log "whole pipeline under $name"
  pipeline "$name" || log "the whole-pipeline benchmark under $name failed; the kernel results still stand"
done

tar -C "$WORK/out" -czf "$WORK/results.tgz" .
log "done: $WORK/results.tgz"
if [ "${PRINT_RESULTS:-0}" = 1 ]; then
  echo "===RESULTS-BEGIN==="
  base64 -w0 "$WORK/results.tgz"
  echo
  echo "===RESULTS-END==="
fi
