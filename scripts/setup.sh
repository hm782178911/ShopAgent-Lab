#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHOPSIM_ROOT="${ROOT}/environments/ShopSimulator"
SHOP_ENV_ROOT="${SHOPSIM_ROOT}/shop_env"
ENV_DIR="${SHOPSIM_ROOT}/.venv-shopsim"
COMPRESSED_PRODUCTS="${SHOP_ENV_ROOT}/data/fine_items_eval_train_all.json.gz"
PRODUCTS="${SHOP_ENV_ROOT}/data/items_eval_train.json"
EXPECTED_PRODUCT_SHA256="57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f"
MAIN_PYTHON="3.12.13"
SHOPSIM_PYTHON="3.10.20"

if [[ ! -f "${COMPRESSED_PRODUCTS}" ]]; then
  echo "Missing embedded product archive: ${COMPRESSED_PRODUCTS}" >&2
  exit 1
fi
if [[ "$(uname -s)/$(uname -m)" != "Linux/x86_64" ]]; then
  echo "This runtime lock supports Linux x86_64 only." >&2
  exit 1
fi
for tool in curl tar gcc g++ patch gzip sha256sum; do
  command -v "$tool" >/dev/null || { echo "Missing system tool: $tool" >&2; exit 1; }
done
if [[ -L "$ROOT/.runtime" ]]; then
  echo "Refusing a linked .runtime directory." >&2
  exit 1
fi
UV_VERSION=0.11.30
if ! command -v uv >/dev/null || [[ "$(uv --version | awk '{print $2}')" != "$UV_VERSION" ]]; then
  mkdir -p "$ROOT/.runtime/uv"
  if [[ ! -x "$ROOT/.runtime/uv/uv" ]] || [[ "$("$ROOT/.runtime/uv/uv" --version | awk '{print $2}')" != "$UV_VERSION" ]]; then
    if [[ "${UV_OFFLINE:-false}" == "true" || "${UV_OFFLINE:-0}" == "1" ]]; then
      echo "uv $UV_VERSION is unavailable locally and UV_OFFLINE forbids downloading it." >&2
      exit 1
    fi
    archive="$(mktemp "$ROOT/.runtime/uv-download.XXXXXX")"
    trap 'rm -f "$archive"' EXIT
    curl --fail --location --retry 3 \
      "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" -o "$archive"
    echo "04bc7d180d6138bf6dc08387acf507a823f397a98fea55da36b0ccc7fbce3b68  $archive" | sha256sum --check
    tar -xzf "$archive" --strip-components=1 -C "$ROOT/.runtime/uv"
    rm -f "$archive"
    trap - EXIT
  fi
  export PATH="$ROOT/.runtime/uv:$PATH"
fi

cd "${ROOT}"
# Never inherit a calling conda environment's Python import paths.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 UV_LINK_MODE=copy UV_PYTHON_PREFERENCE=only-managed
export UV_PYTHON_INSTALL_DIR="$ROOT/.runtime/python"
uv python install --no-bin "$MAIN_PYTHON" "$SHOPSIM_PYTHON"
if [[ -L "$ROOT/.venv" ]]; then
  echo "Refusing a linked .venv; use a fresh checkout." >&2
  exit 1
fi
if [[ -e "$ROOT/.venv" ]]; then
  "$ROOT/.venv/bin/python" -I - "$ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'src'))
from shopping_grpo.runtime import validate_isolation
validate_isolation()
if not Path(sys.base_prefix).resolve().is_relative_to(root / '.runtime/python'):
    raise SystemExit('Existing .venv uses an external Python; use a fresh checkout.')
if sys.version_info[:3] != (3, 12, 13):
    raise SystemExit('Existing .venv has a different Python version; use a fresh checkout.')
PY
else
  uv venv --python "$MAIN_PYTHON" "$ROOT/.venv"
fi
uv pip install --python "$ROOT/.venv/bin/python" --require-hashes -r "$ROOT/configs/build.lock"
uv pip sync --no-build-isolation --python "$ROOT/.venv/bin/python" --require-hashes "$ROOT/configs/runtime.lock"
uv pip install --python "$ROOT/.venv/bin/python" --no-deps --no-build-isolation -e "$ROOT"
uv pip check --python "$ROOT/.venv/bin/python"
"$ROOT/.venv/bin/python" -m shopping_grpo.runtime --prepare-cuda
"$ROOT/.venv/bin/python" scripts/apply_verl_dynamic_sampling_patch.py
"$ROOT/.venv/bin/python" scripts/apply_verl_sglang_patches.py

if [[ -L "$ENV_DIR" ]]; then
  echo "Refusing a linked ShopSimulator environment: $ENV_DIR" >&2
  exit 1
fi
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  uv venv --python "${SHOPSIM_PYTHON}" "${ENV_DIR}"
fi
"$ENV_DIR/bin/python" -I - "$ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'src'))
from shopping_grpo.runtime import validate_isolation
validate_isolation()
if not Path(sys.base_prefix).resolve().is_relative_to(root / '.runtime/python') or sys.version_info[:3] != (3, 10, 20):
    raise SystemExit('Existing simulator environment uses an external/different Python; use a fresh checkout.')
PY
uv pip install --python "$ENV_DIR/bin/python" --require-hashes -r "$ROOT/configs/build.lock"
uv pip sync --no-build-isolation --python "${ENV_DIR}/bin/python" --require-hashes "$ROOT/configs/shopsim.lock"
uv pip check --python "${ENV_DIR}/bin/python"

if [[ ! -f "${PRODUCTS}" ]]; then
  temporary_products="${PRODUCTS}.preparing"
  trap 'rm -f "${temporary_products}"' EXIT
  gzip -cd "${COMPRESSED_PRODUCTS}" > "${temporary_products}"
  actual_sha256="$(sha256sum "${temporary_products}" | awk '{print $1}')"
  if [[ "${actual_sha256}" != "${EXPECTED_PRODUCT_SHA256}" ]]; then
    echo "Product data SHA-256 mismatch: ${actual_sha256}" >&2
    exit 1
  fi
  mv "${temporary_products}" "${PRODUCTS}"
  trap - EXIT
fi

actual_sha256="$(sha256sum "${PRODUCTS}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_PRODUCT_SHA256}" ]]; then
  echo "Existing product data SHA-256 mismatch: ${actual_sha256}" >&2
  exit 1
fi

cd "${SHOP_ENV_ROOT}"
PYTHONPATH=. "${ENV_DIR}/bin/python" scripts/build_index.py

cd "${ROOT}"
echo "Shopping Agent training environment is ready."
echo "Product SHA-256: ${actual_sha256}"
echo "Index: ${SHOP_ENV_ROOT}/search_engine/products.sqlite3"
