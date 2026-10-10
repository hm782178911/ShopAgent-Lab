# ShopAgent-Lab

ShopAgent-Lab 是一个面向购物 Agent 的后训练实践项目，覆盖 **SFT、GRPO、模型导出和固定任务评估**。项目基于 [shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon) 进行 SGLang 适配，沿用 ShopSimulator 与 Reward v3。来源和授权信息见 [SOURCES.md](SOURCES.md)。

## 快速开始

```bash
git clone git@github.com:hm782178911/ShopAgent-Lab.git
cd ShopAgent-Lab
bash scripts/setup.sh
```

完整流程是：**准备环境 → Base 推理 → SFT → GRPO → 导出 → Final-200 评估**。模型权重、缓存、运行日志和中间轨迹不会提交到仓库；仓库只保留可复现的代码、最终数据、元数据和结果摘要。

## 本次新增工作：GPT-6.1 Sol 后训练扩展

本项目在原有 ShopAgent-Lab 流程上新增了一套 **GPT-6.1 Sol 教师数据采集与后训练实验**。这部分工作对应独立的数据、训练输出和结果摘要，不会修改或覆盖仓库原有的 `data/sft/`、`data/grpo/`、模型检查点或原实验输出。

### 做了什么

- 使用 ChatGPT 登录的 Codex CLI，以 `gpt-6.1-sol` 作为教师模型；
- 为每个购物任务启动独立临时上下文，通过本地 MCP bridge 与 ShopSimulator 联动；
- 让模型根据结构化 `SHOPPING_OBSERVATION_V2` 观察逐步调用购物工具，生成完整多轮轨迹；
- 用 ShopSimulator Reward v3 严格筛选 `gold_purchase`，失败、无效和中断轨迹留在审计目录，不进入训练集；
- 使用 Qwen3.5 的真实处理器和训练模板做 24576 token 加载检查，最终得到 800 条训练数据和 200 条验证数据，加载丢弃数为 0；
- 在独立目录中复现 3 epoch SFT、500 步 GRPO，以及 SGLang/vLLM 双后端固定 200 任务评估。

### 仓库中保留的内容

- 最终可训练数据：[`data/sft_gpt61/`](data/sft_gpt61/)；
- 数据生成说明和采集命令：[`data/sft_gpt61/GENERATION.md`](data/sft_gpt61/GENERATION.md)；
- 从 raw JSONL 构建训练文件的脚本：[`data/sft_gpt61/generate.py`](data/sft_gpt61/generate.py)；
- 数据校验、哈希和分区信息：[`data/sft_gpt61/metadata.json`](data/sft_gpt61/metadata.json)；
- 训练结果摘要：[`experiments/gpt61/summary.json`](experiments/gpt61/summary.json)；
- 完整流程、Codex 上下文结构和复现入口：[`docs/gpt61-posttraining.md`](docs/gpt61-posttraining.md)。

原始 Codex 事件流、仿真器日志和机器相关审计文件不提交到仓库；采集端的 `campaign.py`、`run.sh` 和 `collect.py` 运行在已登录的采集机器上，仓库保留最终派生数据和可复查的生成脚本。教师数据的采集是多轮“模型决策 → 工具调用 → 仿真观察”循环，不是一次性预测整条操作序列。

## 实验结果与基线对照

所有结果都使用 Qwen3.5-2B、ShopSimulator Reward v3 和固定 200 个评估任务。严格成功指最终状态为 `gold_purchase` 且 `reward_valid=true`；失败和错误任务计入分母。每个模型的 SGLang/vLLM 评估使用相同导出权重。

### 原作者数据

原作者数据完成 3 个 epoch SFT 和 500 步 SGLang GRPO：

| 阶段 | SGLang | vLLM |
|---|---:|---:|
| Base | 0.5%（1/200） | 1.0%（2/200） |
| SFT | 65.0%（130/200） | 69.5%（139/200） |
| GRPO500 | 67.0%（134/200） | 71.0%（142/200） |

### GPT-6.1 Sol 教师数据

新增教师数据由 GPT‑6.1 Sol 通过 Codex CLI 采集，严格按 Reward v3 筛选，最终为 800 条训练数据和 200 条验证数据。Qwen3.5 训练入口在 24576 token 上限下加载丢弃 0 条，训练/验证任务无重叠，原 `data/sft/` 保持不变。

| 阶段 | SGLang | vLLM | 平均奖励（SGLang / vLLM） |
|---|---:|---:|---:|
| SFT | **74.5%（149/200）** | **74.5%（149/200）** | **0.6728 / 0.6674** |
| GRPO500 | **77.5%（155/200）** | **77.5%（155/200）** | **0.7090 / 0.7081** |

相对原作者数据，新教师数据使 SFT 在 SGLang/vLLM 上分别提升 **9.5/5.0 个百分点**，GRPO500 分别提升 **10.5/6.5 个百分点**。这是单次固定 200 任务的实验观察，不能据此推断对所有任务、随机种子或推理后端都普遍成立。

GPT‑6.1 Sol 数据和复现入口：

- 数据：[data/sft_gpt61/](data/sft_gpt61/)
- 后训练说明：[docs/gpt61-posttraining.md](docs/gpt61-posttraining.md)
- 结果摘要：[experiments/gpt61/summary.json](experiments/gpt61/summary.json)
- 使用新 SFT 数据：`SFT_DATA_DIR=$PWD/data/sft_gpt61 bash scripts/sft.sh`

SGLang/vLLM 存在已记录的 143 token 输入计数差异，因此不能把两者成绩差异直接解释为框架优劣。完整协议和结果局限见 [详细报告](docs/reproduction-results.md)。

## 1. 环境与复现范围

工作流：**Baseline → LoRA SFT → 在线 GRPO → 导出 → Evaluation**。

安装会从固定公开来源创建项目自己的 Python、SGLang、PyTorch、CUDA 编译工具和仿真环境。
**不需要 conda、预装 SGLang、系统 Python 开发包或本机 SGLang 源码。**
`.venv` 不继承系统包；`.runtime/python` 保存独立 Python。GPU 入口会清除外部 Python/CUDA 库路径，使用项目自己的 CUDA 工具。

仍需宿主机提供 Linux x86_64（glibc ≥ 2.34）、兼容 CUDA 13 的 NVIDIA 驱动、GPU、Git、curl、tar、gzip、patch、gcc/g++。
系统驱动不属于 Python 虚拟环境。验证机器为两张 96 GB NVIDIA GPU、驱动 595.84；其他硬件配置需另测。
Ubuntu 缺少基础工具时先执行：

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates build-essential patch gzip
```

安装与 GRPO 预检查共用 [configs/runtime.json](configs/runtime.json)，全部传递依赖及文件哈希锁定在
[configs/runtime.lock](configs/runtime.lock)，仿真依赖锁定在 [configs/shopsim.lock](configs/shopsim.lock)。

| 组件 | 固定版本 |
|---|---|
| Python / 仿真 Python | 3.12.13 / 3.10.20 |
| uv | 0.11.30（缺少时自动下载并校验 SHA256） |
| SGLang | 0.5.16，官方 wheel |
| PyTorch / Transformers | 2.11.0 / 5.12.1 |
| NumPy / ModelScope | 1.26.4 / 1.38.1 |
| veRL / PEFT | 0.8.0 / 0.20.0 |
| Ray / TensorDict | 2.56.1 / 0.10.0 |

单步验证用于检查工程流程，不代表训练效果。单步 SFT 不保证已学会购物，GRPO 的有效更新验证需要有能力的完整 SFT checkpoint。
已验证首次/重复安装、非零单步 SFT、一次双卡 GRPO 更新、导出和三种模型的两任务推理；范围详见 [验证说明](docs/validation.md)。
完整训练与正式评估已完成，成绩见上方实验结果。

## 2. 克隆与安装

```bash
git clone git@github.com:hm782178911/ShopAgent-Lab.git
cd ShopAgent-Lab
bash scripts/setup.sh
```

不需要激活 conda。setup 会：

1. 准备固定版本 uv 和两个独立 Python（自带 `Python.h`）。
2. 按哈希锁安装完整依赖（含固定构建工具，禁用隐式构建环境），执行 `uv pip check` 检查声明约束一致性，再安装本项目。
3. 用复制模式安装依赖，对 veRL 应用可重复、带源文件哈希检查的补丁，不修改下载缓存或其他环境。
4. 建立独立 ShopSimulator 环境，校验商品数据并构建搜索索引。

可以重复执行。已有环境继承外部包、使用外部 Python 或是符号链接时会拒绝覆盖，请使用新的克隆目录。
不要复制或移动 `.venv`、`.runtime` 到另一目录；在新目录重跑安装。
本项目使用 requirements 哈希锁，**不要用 `uv sync` 替代 setup**。

### 下载与缓存

优先直连。若 PyPI 较慢，可使用镜像与官方源回退；所有安装包仍必须匹配锁定哈希：

```bash
UV_INDEX=https://mirrors.aliyun.com/pypi/simple \
UV_INDEX_STRATEGY=unsafe-best-match bash scripts/setup.sh
```

这里允许多个源匹配**已固定版本和哈希**，不会自动升级依赖。镜像不保证拥有全部版本。
GitHub 上的 uv/Python 下载若确有网络问题，可临时配置代理；不要把代理地址写进仓库。
下载缓存可复用以节约流量，缓存并非运行时依赖。已有项目 Python 和完整包缓存时可用 `UV_OFFLINE=true` 重复安装；离线缺包会明确报错。

### 运行终端

每个终端进入仓库后设置仿真地址，本地请求绕过代理：

```bash
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export SHOPSIM_BASE_URL=http://127.0.0.1:5701
```

直接使用 `.venv/bin/python` 或本文的 shell 入口，无需手动设置 Python.h、CUDA_HOME 或 PYTHONPATH。

## 3. ModelScope 下载

`configs/model.yaml` 默认选择：

```yaml
source: modelscope
model_id: Qwen/Qwen3.5-2B
revision: c00cc5fd7803c60b7788e053dcce33d0d26b11ef
local_dir: models/Qwen3.5-2B
```

保留文件原有 `serving` 配置，它控制服务名称、端口、上下文、TP 和显存比例。

```bash
.venv/bin/python scripts/download_model.py
.venv/bin/python scripts/download_model.py --local-only
```

完整权重约 4.55 GB，下载后生成包含来源及 SHA256 的 `model-source.json`。
`--local-only` 检查文件存在、分片完整性和来源标记，不重新计算所有哈希，也不下载。
不要用未登记的非空目录作为下载目标。默认已固定不可变 revision；切换模型或 revision 时使用新的下载目录。

## 4. 启动仿真与 Base 推理

终端 A：

```bash
SHOPSIM_PORT=5701 bash scripts/start_environment.sh
```

本文统一使用 5701，其他终端的 `SHOPSIM_BASE_URL` 和 GRPO 的 `--env-url` 必须一致。
环境契约为 Environment v2.1、Reward v3、observation v2、tool schema v2。

终端 B：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_model.sh
```

终端 C：

```bash
curl --fail http://127.0.0.1:8000/v1/models
.venv/bin/python scripts/smoke_shop_env.py --base-url "$SHOPSIM_BASE_URL" \
  --task-id 8187 --output-dir outputs/smoke/environment
```

## 5. 两条任务的评估模板

创建固定子集，不改原始评估集：

```bash
mkdir -p outputs/smoke
head -n 2 data/evaluation/tasks.jsonl > outputs/smoke/tasks.jsonl
EVAL_LABEL=base
mkdir -p "outputs/smoke/$EVAL_LABEL"
.venv/bin/python scripts/evaluate_shop_benchmark.py \
  --benchmark outputs/smoke/tasks.jsonl \
  --output "outputs/smoke/$EVAL_LABEL/trajectories.jsonl" \
  --summary "outputs/smoke/$EVAL_LABEL/summary.json" \
  --base-url "$SHOPSIM_BASE_URL" --model shopping-agent \
  --llm-base-url http://127.0.0.1:8000/v1 --api-key EMPTY
.venv/bin/python scripts/build_eval_report.py --run-dir "outputs/smoke/$EVAL_LABEL"
```

报告为 `outputs/smoke/base/report.html`。SFT、GRPO 复用这段评估命令，修改标签即可。
**标签不会切换服务器上的模型**，必须先启动对应模型。不同运行使用新标签。

上下文计数使用 SGLang `/v1/tokenize`。`completed_tasks` 可能包含 error；需要检查
`status_counts`、轨迹 `error` 和严格成功率。done 率不是购买成功率。

## 6. SFT 与自动合并

先在终端 B 用 Ctrl+C 结束独立模型服务，释放训练显卡。

### 单步验证

验证子集来自 SFT validation，不使用评估任务参与训练或训练期验证。

```bash
mkdir -p outputs/smoke
head -n 2 data/sft/validation.jsonl > outputs/smoke/sft-validation-2.jsonl
CUDA_VISIBLE_DEVICES=0 \
SFT_ADAPTER_DIR="$PWD/outputs/smoke/sft-lora" \
SFT_MERGED_DIR="$PWD/outputs/smoke/sft-merged" \
bash scripts/sft.sh \
  --max-steps 1 --warmup-ratio 0 --train-count 2 \
  --gradient-accumulation-steps 1 --logging-steps 1 \
  --validation outputs/smoke/sft-validation-2.jsonl
```

`warmup-ratio=0` 避免唯一一步学习率为零。脚本自动合并 LoRA，并将 tokenizer 的 EOS 加入
generation config，保留原有 EOS。Qwen3.5 通常得到 `[248044,248046]`，避免原始生成越过对话结束符。
无需手工编辑 EOS，变化记录在 `merge_manifest.json`。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_model.sh outputs/smoke/sft-merged
```

另一个终端按第 5 节评估，设置 `EVAL_LABEL=sft`。

### 完整 SFT（需要正式实验时执行）

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/sft.sh
```

使用 800 条训练数据、200 条验证数据，自动合并到 `outputs/models/sft-merged`。
可以用 `BASE_MODEL` 显式指定本地模型，`MODEL_CONFIG` 指定另一配置。
输出用 `SFT_ADAPTER_DIR` / `SFT_MERGED_DIR` 配置，不要单独传入训练器的 `--output`。

## 7. GRPO 单步验证

停止独立模型服务，保持仿真运行。veRL 自己启动 SGLang rollout，不连接 8000 上的服务。
**GRPO 应从充分训练的 SFT 模型开始**；只训练一步 SFT 可能因无效轨迹过多而无法更新。
完整 SFT checkpoint 不随 Git 仓库提供，需自行训练或明确提供来源。

```bash
SFT_MODEL="$PWD/outputs/models/sft-merged"
GRPO_OUTPUT="$PWD/outputs/smoke/grpo"
CUDA_VISIBLE_DEVICES=0,1 bash scripts/grpo.sh \
  --model "$SFT_MODEL" --output "$GRPO_OUTPUT" --env-url "$SHOPSIM_BASE_URL" -- \
  trainer.total_training_steps=1 trainer.save_freq=1 \
  trainer.test_freq=-1 trainer.val_before_train=false \
  trainer.n_gpus_per_node=2 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  shopping_dynamic_sampling.max_consecutive_skipped_updates=1
```

版本预检查直接读取统一配置，不需要内嵌 Python 修改检查器。`grpo.sh` 设置显式 rank 设备选择、
Python 头文件和 NCCL P2P fallback；P2P 已验证可用的机器可设置 `NCCL_P2P_DISABLE=0`。

确认日志 `training/optimizer_updated:1` 与 `global_step_1/actor`；`SKIPPED` 或服务启动不表示更新成功。
单步验证关闭初始验证与 warmup，一次空更新尝试就停止。换新输出目录重试，不覆盖已有 checkpoint，
不承诺自动续训。

旧版 SFT 合并产物可能未对齐 EOS。复用时先检查配置，必要时由原始模型和 adapter 重新运行
`merge_lora_adapter.py`，生成新的独立产物，保留原 checkpoint。

### 完整 GRPO

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/grpo.sh \
  --model "$SFT_MODEL" --output outputs/models/grpo --env-url "$SHOPSIM_BASE_URL"
```

使用 `configs/grpo.yaml` 默认值：500 步、每 10 步保存、每 50 步验证、开始前验证。
完整训练效果仍需单独实验。

## 8. 自动导出与最终推理

```bash
bash scripts/export_grpo.sh \
  "$GRPO_OUTPUT/global_step_1/actor" outputs/smoke/grpo-final
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_model.sh outputs/smoke/grpo-final
```

导出入口自动执行 **FSDP 分片导出 → 检测 adapter → 合并 LoRA → 发布最终目录**。
不要再对最终模型重复合并同一 adapter。输入必须含 `actor/huggingface/config.json`，输出必须新建或为空。
失败会保留临时目录用于定位，不把半成品放进目标目录；来源记录在 `export_manifest.json`。

另一个终端按第 5 节评估，设置 `EVAL_LABEL=grpo`。完整训练时把 step 1 改成实际保存的步数。

## 9. 正式评估与测试

Base、SFT、GRPO 对比必须使用相同后端、任务集和参数。分别启动对应模型后执行：

```bash
bash scripts/evaluate.sh base
# 换成完整 SFT 模型服务后：
bash scripts/evaluate.sh sft
# 换成 GRPO 最终模型服务后：
bash scripts/evaluate.sh grpo
```

每条命令运行完整 Final-200 Clean，报告在 `outputs/evaluation/<标签>/report.html`。
严格成功要求 `gold_purchase` 终态且 `reward_valid=true`。默认奖励是环境规则，不需要外部 LLM 裁判；
额外 Judge 见 [docs/evaluation.md](docs/evaluation.md)。

pytest 已包含在锁定环境中。

```bash
.venv/bin/python -m pytest \
  tests/test_runtime_installation.py tests/test_runtime_isolation.py \
  tests/test_runtime_environment.py tests/test_export_grpo_cli.py \
  tests/test_merge_lora_adapter.py tests/test_grpo_dynamic_sampling_config.py \
  tests/test_model_source.py tests/test_sglang_token_counter.py \
  tests/test_context_window.py tests/test_environment_manifest.py \
  tests/test_evaluation_dataset.py -q
```

## 10. 常见问题

| 现象 | 检查方向 |
|---|---|
| setup 拒绝旧环境 | 用新的克隆目录；不要继承旧 conda 或软链接 .venv |
| Python.h / CUDA 编译器缺失 | 重跑 setup 修复项目环境，不借用另一个 conda 环境 |
| CPU 内核 / adapter 错误 | 重新运行 setup；补丁会校验哈希，未知源文件会拒绝修改 |
| checkpoint 路径错误 | 使用实际运行目录的 global_step_*/actor，各仓库 outputs 不会自动共享 |
| 输出非空 | 换新目录，不随意删除已有结果 |
| GRPO 无更新 | 检查有效组及 optimizer_updated，单步 SFT 能力可能不足 |
| ContextBudgetError | 轨迹超出输入预算，修改预算后的结果不能混入原协议对比 |
| 仿真无可用槽位 | 检查并发及中断残留 lease，确认无其他任务后再处理 |

运行前后用 `nvidia-smi` 检查 GPU，只停止自己启动的服务。模型权重、输出、虚拟环境和凭据
不提交到 Git。当前安装与运行命令以本文为准。

## 11. 更新依赖锁（维护者）

普通复现只运行 setup。调整 [configs/runtime.in](configs/runtime.in) 后，使用固定 uv 重新解析：

```bash
uv pip compile configs/runtime.in --python-version 3.12 \
  --python-platform x86_64-manylinux_2_34 --generate-hashes -o configs/runtime.lock
uv pip compile configs/build.in --python-version 3.10 \
  --python-platform x86_64-manylinux_2_34 --generate-hashes -o configs/build.lock
uv pip compile configs/shopsim.in --python-version 3.10 \
  --python-platform x86_64-manylinux_2_34 --generate-hashes -o configs/shopsim.lock
```

随后同步 runtime.json，在新目录验证安装、依赖检查、单步训练、导出和推理，再提交锁文件。
不要仅修改版本预检查来跳过不兼容问题。GPU 架构、驱动和随机采样会影响数值结果，本项目不承诺逐位一致。
