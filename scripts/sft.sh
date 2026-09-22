#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-$("$ROOT/.venv/bin/python" "$ROOT/scripts/download_model.py" --config "${MODEL_CONFIG:-$ROOT/configs/model.yaml}" --local-only)}"
ADAPTER_DIR="${SFT_ADAPTER_DIR:-$ROOT/outputs/models/sft-lora}"
MERGED_DIR="${SFT_MERGED_DIR:-$ROOT/outputs/models/sft-merged}"

cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/train_lora_sft.py \
  --model "$BASE_MODEL" \
  --train data/sft/train.jsonl \
  --validation data/sft/validation.jsonl \
  --output "$ADAPTER_DIR" \
  --dtype auto \
  --gradient-checkpointing \
  --attention-implementation sdpa

exec "$ROOT/.venv/bin/python" scripts/merge_lora_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER_DIR" \
  --output "$MERGED_DIR" \
  --bf16
