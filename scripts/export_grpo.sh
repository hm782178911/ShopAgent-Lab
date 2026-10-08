#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTOR_CHECKPOINT="$(realpath -m -- "${1:?usage: bash scripts/export_grpo.sh <global_step_*/actor> [output_dir]}")"
OUTPUT_DIR="${2:-$ROOT/outputs/models/grpo-merged}"
if [[ ! -f "$ACTOR_CHECKPOINT/huggingface/config.json" ]]; then
  echo "checkpoint is missing actor/huggingface/config.json: $ACTOR_CHECKPOINT" >&2
  exit 1
fi
if [[ -L "$OUTPUT_DIR" ]] || { [[ -e "$OUTPUT_DIR" ]] && { [[ ! -d "$OUTPUT_DIR" ]] || [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; }; }; then
  echo "output must be new or empty: $OUTPUT_DIR" >&2
  exit 1
fi
OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
mkdir -p "$(dirname "$OUTPUT_DIR")"
WORK="$(mktemp -d "$(dirname "$OUTPUT_DIR")/.grpo-export.XXXXXX")"
trap 'echo "Export failed; intermediate files retained at: $WORK" >&2' ERR

"$ROOT/.venv/bin/python" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_CHECKPOINT" \
  --target_dir "$WORK/raw" \
  --trust-remote-code

if [[ -f "$WORK/raw/lora_adapter/adapter_config.json" ]]; then
  "$ROOT/.venv/bin/python" "$ROOT/scripts/merge_lora_adapter.py" \
    --base-model "$WORK/raw" --adapter "$WORK/raw/lora_adapter" \
    --output "$WORK/final" --bf16
  MERGED=true
else
  mv "$WORK/raw" "$WORK/final"
  MERGED=false
fi
"$ROOT/.venv/bin/python" - "$WORK/final" "$ACTOR_CHECKPOINT" "$MERGED" <<'PY'
import json, sys
from pathlib import Path
target = Path(sys.argv[1])
record = {'checkpoint': sys.argv[2], 'lora_merged': sys.argv[3] == 'true'}
(target / 'export_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
PY
# Do not replace a directory that became nonempty while the export was running.
if [[ -d "$OUTPUT_DIR" ]]; then rmdir "$OUTPUT_DIR"; fi
mv -T "$WORK/final" "$OUTPUT_DIR"
trap - ERR
rm -rf -- "$WORK"
echo "Final inference model: $OUTPUT_DIR"
