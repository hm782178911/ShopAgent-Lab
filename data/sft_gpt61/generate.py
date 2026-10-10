#!/usr/bin/env python3
"""Build train/validation SFT files from audited teacher rollouts."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from shopping_grpo.collection.sft import build_collection_artifacts, task_ids_from_jsonl

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--held-out', type=Path, default=Path('data/evaluation/tasks.jsonl'))
    parser.add_argument('--validation-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-length', type=int, default=24576)
    args = parser.parse_args()
    if not args.raw.is_file(): parser.error(f'raw trajectory file does not exist: {args.raw}')
    if args.output.exists() and any(args.output.iterdir()): parser.error(f'output directory must be new or empty: {args.output}')
    if not 0 <= args.validation_ratio < 1: parser.error('validation ratio must be in [0, 1)')
    args.output.mkdir(parents=True, exist_ok=True)
    result = build_collection_artifacts(raw_path=args.raw, output_dir=args.output,
        held_out_task_ids=task_ids_from_jsonl(args.held_out), validation_ratio=args.validation_ratio,
        seed=args.seed, collection_config={'teacher':'gpt-6.1-sol','reward':'shopsimulator-reward-v3',
        'environment':'shopsimulator-environment-v2.1','max_length':args.max_length,'seed':args.seed})
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
