# GPT-6.1 Sol teacher-data post-training

## Teacher-data generation

The teacher data was generated through the ChatGPT-authenticated Codex CLI, rather than by calling a separate API key. Each shopping task used a fresh ephemeral `codex exec` context with GPT-6.1 Sol. A local stdio MCP bridge exposed only the ShopSimulator shopping actions and returned the real page observations; the teacher had no access to reward labels, gold answers, training files or evaluation outputs.

For every task, the collector recorded the CLI event stream, executed tool calls, simulator observations and the final native trajectory. The bridge validated action names and arguments before execution. Only a completed trajectory satisfying the original ShopSimulator Reward v3 strict `gold_purchase` contract was retained as a candidate. Failed purchases, partial alternatives, invalid or interrupted runs stayed in the audit area and were excluded from training.

Before promotion, the collector checked that task IDs and instruction hashes did not overlap the original SFT, GRPO or Final-200 evaluation partitions. It then rendered the candidate with the repository's Qwen3.5 processor and training template at the 24576-token limit. Rows that could not load cleanly were rejected. The final frozen split contains exactly 800 training rows and 200 validation rows, with zero loader drops and zero partition overlap; its checksums are recorded in `data/sft_gpt61/metadata.json`.

The raw Codex events and machine-specific simulator logs are intentionally kept outside the repository. The repository contains only the final trainable JSONL files, their metadata, and the aggregate result summary.

## Reproducing post-training

The checked-in data can be used with the existing training entry point:

```bash
SFT_DATA_DIR=$PWD/data/sft_gpt61 bash scripts/sft.sh
```

This runs the same 3-epoch LoRA SFT and merge path. Then use the existing `scripts/grpo.sh` with the merged SFT model and the read-only `data/grpo` parquet split for the 500-step SGLang GRPO recipe. Evaluate the exported model on the fixed Final-200 benchmark with the existing SGLang and vLLM serving paths.


The subsequent GRPO recipe is the repository `scripts/grpo.sh` with the merged SFT model, the existing read-only `data/grpo` parquet split, 500 training steps, and the normal SGLang rollout. Evaluate the exported model with `scripts/evaluate.sh` on the fixed Final-200 benchmark using both SGLang and vLLM.

The completed run used 1000 teacher rows (800/200), retained all rows at 24576 tokens, and achieved 74.5% SFT strict success and 77.5% after GRPO500 on both serving backends. These are single fixed-200-task observations; the report retains the known 143-token backend input-count caveat.
