#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-$("$ROOT/.venv/bin/python" "$ROOT/scripts/download_model.py" --config "${MODEL_CONFIG:-$ROOT/configs/model.yaml}" --local-only)}"
ADAPTER_DIR="${SFT_ADAPTER_DIR:-$ROOT/outputs/models/sft-lora}"
MERGED_DIR="${SFT_MERGED_DIR:-$ROOT/outputs/models/sft-merged}"

cd "$ROOT"
for directory in "$ADAPTER_DIR" "$MERGED_DIR"; do
  if [[ -L "$directory" ]] || { [[ -e "$directory" ]] && { [[ ! -d "$directory" ]] || [[ -n "$(find "$directory" -mindepth 1 -maxdepth 1 -print -quit)" ]]; }; }; then
    echo "SFT output must be new or empty: $directory" >&2
    exit 1
  fi
done
"$ROOT/.venv/bin/python" scripts/train_lora_sft.py \
  --model "$BASE_MODEL" \
  --train data/sft/train.jsonl \
  --validation data/sft/validation.jsonl \
  --output "$ADAPTER_DIR" \
  --dtype auto \
  --gradient-checkpointing \
  --attention-implementation sdpa \
  "$@"

exec "$ROOT/.venv/bin/python" scripts/merge_lora_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER_DIR" \
  --output "$MERGED_DIR" \
  --bf16
