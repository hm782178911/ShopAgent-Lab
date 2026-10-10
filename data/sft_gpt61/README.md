# GPT-6.1 Sol SFT data

This task-disjoint 800/200 split was collected with GPT-6.1 Sol through Codex CLI and filtered by ShopSimulator Reward v3 strict `gold_purchase`. Qwen3.5 preprocessing at max length 24576 retained all rows. Raw trajectories and machine-specific audit logs remain outside the repository.

The complete teacher-data workflow is documented in [GENERATION.md](GENERATION.md). The reproducible raw-to-derived build entry point is [generate.py](generate.py):

```bash
python data/sft_gpt61/generate.py \
  --raw /path/to/raw.jsonl \
  --output /tmp/gpt61-sft \
  --held-out data/evaluation/tasks.jsonl \
  --validation-ratio 0.2 \
  --seed 42
```

The script applies the repository's Reward v3 acceptance and split checks. The Codex CLI collection step remains tied to the authenticated machine and its local simulator; its raw event logs are kept for audit and are not committed.
