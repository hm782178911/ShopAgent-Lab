# GPT-6.1 Sol teacher-data post-training

## Teacher-data generation

The teacher data was generated through the ChatGPT-authenticated Codex CLI, rather than by calling a separate API key. Each shopping task used a fresh ephemeral `codex exec` context with GPT-6.1 Sol. A local stdio MCP bridge exposed only the ShopSimulator shopping actions and returned the real page observations; the teacher had no access to reward labels, gold answers, training files or evaluation outputs.

For every task, the collector recorded the CLI event stream, executed tool calls, simulator observations and the final native trajectory. The bridge validated action names and arguments before execution. Only a completed trajectory satisfying the original ShopSimulator Reward v3 strict `gold_purchase` contract was retained as a candidate. Failed purchases, partial alternatives, invalid or interrupted runs stayed in the audit area and were excluded from training.

Before promotion, the collector checked that task IDs and instruction hashes did not overlap the original SFT, GRPO or Final-200 evaluation partitions. It then rendered the candidate with the repository's Qwen3.5 processor and training template at the 24576-token limit. Rows that could not load cleanly were rejected. The final frozen split contains exactly 800 training rows and 200 validation rows, with zero loader drops and zero partition overlap; its checksums are recorded in `data/sft_gpt61/metadata.json`.

The raw Codex events and machine-specific simulator logs are intentionally kept outside the repository. The repository contains only the final trainable JSONL files, their metadata, and the aggregate result summary.


## Codex context and simulator interaction

A normal Codex CLI conversation is a sequence of context items rather than one flat prompt. The effective context can contain runtime instructions, project `AGENTS.md` instructions, the user request, assistant messages, tool calls, and tool outputs. Codex CLI injects discovered `AGENTS.md` files before the user request and uses the tool result from one turn as input to the next turn.

For teacher collection, the same conversation pattern is restricted to the shopping environment. One task runs in one ephemeral `codex exec` session. The effective input at each decision is conceptually:

```text
collection instructions
+ task instruction
+ previous assistant tool calls
+ previous ShopSimulator observations
+ available shop MCP tool schemas
```

The model first calls `get_task`. The bridge resets the simulator and returns the task instruction plus a structured page observation. The model then emits one shop tool call, such as `search_products` or `open_product`. The bridge validates the tool name and arguments, executes one environment step, records the native trajectory, and returns the new page observation as the next tool result. The model uses that updated conversation to choose the next action. A complete trajectory therefore contains multiple model decisions, not one prediction of the whole action sequence.

The bridge exposes only the ShopSimulator MCP tools for this collection. It does not expose shell, file, web, reward-label, training-data, or evaluation tools. At termination it returns a generic completion message to the teacher; detailed Reward v3 fields are retained in the audit trajectory for filtering and are not shown to the teacher. The resulting SFT row keeps the task context, observations, and action calls while removing private reasoning and audit-only reward fields.

A simplified message slice looks like this:

```text
user:      task instruction
assistant: search_products({"query": "..."})
tool:      [SHOPPING_OBSERVATION_V2] search results and clickable actions
assistant: open_product({"asin": "..."})
tool:      [SHOPPING_OBSERVATION_V2] product details and available options
assistant: select_option({"value": "..."})
tool:      [SHOPPING_OBSERVATION_V2] updated selected options
assistant: buy_now({})
tool:      task completed
```


### Collection commands

The durable campaign command runs the authenticated Codex CLI jobs in 50-task batches. Run it on the collection machine after configuring the local proxy and project environment:

```bash
cd /path/to/dataset/gpt_6_1
source ~/env_proxy.sh

nohup /path/to/project/.venv/bin/python campaign.py \
  > campaign.log 2>&1 &
```

Use the read-only gate check before starting a new supervisor:

```bash
python campaign.py --check
```

The supervisor starts each batch through `run.sh`. To run one batch explicitly for a new run name:

```bash
bash run.sh \
  --run full-1000-batch-0001 \
  --count 50 \
  --task-timeout 1800
```

The batch command calls `collect.py`, which starts one ephemeral `codex exec` per task with GPT-6.1 Sol and the local ShopSimulator MCP bridge. Do not start a second supervisor while `campaign.py` or a batch is alive. The raw event stream and simulator logs stay in the campaign run directory for audit.

After collecting raw trajectories, build the derived SFT files with:

```bash
python data/sft_gpt61/generate.py \
  --raw /path/to/raw.jsonl \
  --output /tmp/gpt61-sft \
  --held-out data/evaluation/tasks.jsonl \
  --validation-ratio 0.2 \
  --seed 42
```

## Reproducing post-training

The checked-in data can be used with the existing training entry point:

```bash
SFT_DATA_DIR=$PWD/data/sft_gpt61 bash scripts/sft.sh
```

This runs the same 3-epoch LoRA SFT and merge path. Then use the existing `scripts/grpo.sh` with the merged SFT model and the read-only `data/grpo` parquet split for the 500-step SGLang GRPO recipe. Evaluate the exported model on the fixed Final-200 benchmark with the existing SGLang and vLLM serving paths.


The subsequent GRPO recipe is the repository `scripts/grpo.sh` with the merged SFT model, the existing read-only `data/grpo` parquet split, 500 training steps, and the normal SGLang rollout. Evaluate the exported model with `scripts/evaluate.sh` on the fixed Final-200 benchmark using both SGLang and vLLM.

The completed run used 1000 teacher rows (800/200), retained all rows at 24576 tokens, and achieved 74.5% SFT strict success and 77.5% after GRPO500 on both serving backends. These are single fixed-200-task observations; the report retains the known 143-token backend input-count caveat.
