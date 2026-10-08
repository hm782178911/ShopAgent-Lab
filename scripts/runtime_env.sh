#!/usr/bin/env bash
# Sourced by GPU entry points after they determine ROOT.
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH CUDA_PATH CPATH CPLUS_INCLUDE_PATH LIBRARY_PATH
export PYTHONNOUSERSITE=1
export CUDA_HOME="$ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
if [[ ! -f "$CUDA_HOME/bin/nvcc" ]]; then
  echo "Project CUDA compiler missing; run bash scripts/setup.sh first." >&2
  exit 1
fi
export PATH="$ROOT/.venv/bin:$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin"
export C_INCLUDE_PATH="$("$ROOT/.venv/bin/python" -I -c 'import sysconfig; print(sysconfig.get_path("include"))')"
if [[ ! -f "$C_INCLUDE_PATH/Python.h" ]]; then
  echo "Project Python.h missing; run bash scripts/setup.sh first." >&2
  exit 1
fi
