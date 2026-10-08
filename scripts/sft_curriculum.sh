#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$repo_root"
source "$ROOT/scripts/runtime_env.sh"
sft_python="$ROOT/.venv/bin/python"
if [[ ! -x "${sft_python}" ]]; then
  echo "找不到 Python 环境：${sft_python}；请先运行 scripts/setup.sh。" >&2
  exit 2
fi
exec "${sft_python}" "${repo_root}/scripts/run_sft_curriculum.py" "$@"
