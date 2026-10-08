#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT/environments/ShopSimulator/.venv-shopsim"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "ShopSimulator is not installed. Run: bash scripts/setup.sh" >&2
  exit 1
fi

unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
export PATH="$ENV_DIR/bin:/usr/local/bin:/usr/bin:/bin"
exec "$ROOT/environments/ShopSimulator/shop_env/start.sh"
