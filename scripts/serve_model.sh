#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/runtime_env.sh"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" scripts/serve_sglang.py "$@"
