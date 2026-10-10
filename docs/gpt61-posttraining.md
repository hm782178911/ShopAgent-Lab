# GPT-6.1 Sol teacher-data post-training

The reproducible SFT data is in `data/sft_gpt61/`. Set `SFT_DATA_DIR=$PWD/data/sft_gpt61` when invoking the existing `scripts/sft.sh`; it runs the same 3-epoch LoRA recipe and merge path used by the experiment.

The subsequent GRPO recipe is the repository `scripts/grpo.sh` with the merged SFT model, the existing read-only `data/grpo` parquet split, 500 training steps, and the normal SGLang rollout. Evaluate the exported model with `scripts/evaluate.sh` on the fixed Final-200 benchmark using both SGLang and vLLM.

The completed run used 1000 teacher rows (800/200), retained all rows at 24576 tokens, and achieved 74.5% SFT strict success and 77.5% after GRPO500 on both serving backends. These are single fixed-200-task observations; the report retains the known 143-token backend input-count caveat.
