# GPT-6.1 Sol 数据生成

## 两步流程

### 1. 采集原始教师轨迹

使用 ChatGPT 登录的 Codex CLI，每个任务启动独立临时上下文；通过本地 MCP bridge 只开放 ShopSimulator 购物动作。bridge 记录真实观测、工具调用和终态 reward，但不向教师暴露 reward、标准答案、训练文件或 Final-200 评估结果。

原始记录包含任务 ID、消息、工具调用、观测和终态结果。采集完成后保留 raw JSONL 和审计日志，不把它们直接当作训练数据。


## 模型每一步看到什么

Codex CLI 的一次任务不是把完整操作序列一次性生成出来，而是连续多轮决策。每一轮的上下文由任务提示、历史工具调用、历史仿真观察和当前可用的 shop MCP 工具 schema 组成。模型先调用 `get_task`，bridge 重置环境并返回结构化页面观察；之后每调用一个购物工具，仿真环境推进一步，新的页面观察再作为下一轮 tool result 返回给模型。

简化后的数据流如下：

```text
user:      购物任务
assistant: search_products({"query": "..."})
tool:      [SHOPPING_OBSERVATION_V2] 搜索结果和可点击动作
assistant: open_product({"asin": "..."})
tool:      [SHOPPING_OBSERVATION_V2] 商品详情和可用选项
assistant: select_option({"value": "..."})
tool:      [SHOPPING_OBSERVATION_V2] 更新后的选择状态
assistant: buy_now({})
tool:      任务结束
```

因此一条教师轨迹由多次模型推理和环境动作组成。教师只看到页面观察和工具结果，不看到 Reward v3 的详细字段、标准答案、训练文件或评估结果；这些字段只保存在审计轨迹中，用于后续严格筛选。

## 实际采集命令

在已经登录 ChatGPT 的采集机器上运行持久化调度器：

```bash
cd /path/to/dataset/gpt_6_1
source ~/env_proxy.sh

nohup /path/to/project/.venv/bin/python campaign.py \
  > campaign.log 2>&1 &
```

开始新的一批任务时，调度器会自动调用 `run.sh`。需要单独重跑一批时使用新的 run 名称：

```bash
bash run.sh \
  --run full-1000-batch-0001 \
  --count 50 \
  --task-timeout 1800
```

`run.sh` 会调用 `collect.py`；它为每个任务启动独立的临时 `codex exec`，启用 GPT-6.1 Sol 和本地 ShopSimulator MCP bridge。`campaign.py --check` 只检查当前采集门槛，不会启动任务。已有 supervisor 或 batch 进程存活时不要重复启动。

### 2. 从 raw JSONL 生成最终数据

```bash
python data/sft_gpt61/generate.py \
  --raw /path/to/raw.jsonl \
  --output /tmp/gpt61-sft \
  --held-out data/evaluation/tasks.jsonl \
  --validation-ratio 0.2 \
  --seed 42
```

脚本调用仓库现有的 Reward v3 接受规则，只保留完整 `gold_purchase` 轨迹；失败、替代购买、无效和中断轨迹会进入拒绝统计。它排除 `--held-out` 任务 ID，去除重复任务，并按 seed 42 生成 train/validation。输出目录必须为空，避免覆盖已有结果。

生成后还要使用实际 Qwen3.5 processor 和训练模板做 24576-token 加载校验；只有 `dropped=0` 的数据才能发布到 `data/sft_gpt61/`。最终目录应包含 `train.jsonl`、`validation.jsonl`、`metadata.json` 和本说明文件。

原始轨迹、Codex 事件、模拟器日志和凭据不提交仓库；仓库只保留最终可训练数据和可复查元数据。
