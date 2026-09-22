#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Let veRL explicitly select the rank's device. P2P fallback can be overridden
# to 0 on a machine whose peer-to-peer collective test passes.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
PY_INCLUDE="$("$ROOT/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
if [[ ! -f "$PY_INCLUDE/Python.h" ]]; then
  echo "Python.h missing at $PY_INCLUDE; create .venv using a Python with development headers." >&2
  exit 1
fi
export C_INCLUDE_PATH="$PY_INCLUDE${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
exec "$ROOT/.venv/bin/python" scripts/train_grpo.py "$@"
