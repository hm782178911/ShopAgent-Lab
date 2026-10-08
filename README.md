# ShopAgent-Lab

面向购物 Agent 的后训练实践，涵盖 SFT、GRPO 与评估，基于 SGLang。

本项目基于 [shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon)
复现并进行 SGLang 适配，沿用原作者数据、ShopSimulator 和 Reward v3。
来源与授权状态见 [SOURCES.md](SOURCES.md)。上游历史成绩不代表本实现的效果。

## 1. 当前状态与前提

工作流：**Baseline → LoRA SFT → 在线 GRPO → 导出 → Evaluation**。

脚本已整合环境安装、依赖补丁、版本预检查、EOS 对齐和 GRPO adapter 导出合并。
验证平台为 Linux、两张 96 GB NVIDIA GPU、已有 Python 3.12 conda SGLang 环境。

- 已验证首次安装、重复安装、SFT 一次真实更新、GRPO 一次真实更新、最终模型重载及两条任务评估。
- 单步 SFT 不保证模型已学会购物。本次单步 SFT 直接接 GRPO 未组成有效训练组；GRPO 更新测试采用此前完整训练的 SFT checkpoint。
- 尚未在 SGLang 下完成全量训练与正式 200 题对比，少量测试不能当作效果成绩。
- **setup 复用已有 SGLang 基础环境，不负责从零安装 SGLang、CUDA 或驱动。**

安装与 GRPO 预检查共用 [configs/runtime.json](configs/runtime.json)：

| 组件 | 验证版本 |
|---|---|
| Python | 3.12（验证使用 3.12.13） |
| SGLang | `0.5.15.post2.dev468+g85ac56c82` |
| SGLang 源提交 | `85ac56c82308b8f0e86b3aa8542b61af0de666aa` |
| PyTorch / Transformers | 2.11.0 / 5.12.1 |
| NumPy / ModelScope | 2.3.5 / 1.38.1 |
| veRL / PEFT | 0.8.0 / 0.20.0 |
| Ray / TensorDict | 2.56.1 / 0.10.0 |

SGLang 是源码开发版本，仅创建同名 conda 环境并不足够。PyTorch 允许 CUDA 构建后缀，
SGLang 的提交后缀也会检查。其他 GPU、模型、依赖组合需另测。

## 2. 克隆与安装

需要 Git、uv、patch、gzip 和 Python 开发头文件；ShopSimulator 单独使用 Python 3.10。

```bash
git clone git@github.com:hm782178911/ShopAgent-Lab.git
cd ShopAgent-Lab
conda activate xlm_py312
MAIN_PYTHON="$(command -v python)" bash scripts/setup.sh
```

`xlm_py312` 是已有基础环境的示例名称，`MAIN_PYTHON` 必须指向满足上述版本的解释器。
setup 会先校验基础版本与 `Python.h`，再：

1. 创建独立 `.venv`，继承基础包，按 [requirements-training.txt](configs/requirements-training.txt) 补充依赖。
2. 用复制模式单独安装 veRL，应用并校验动态采样和两个 SGLang 补丁。
3. 创建独立仿真环境，解压并校验商品数据，构建搜索索引。

重复安装可再次执行，不修改原 conda 环境。已有 `.venv` 为符号链接或基于另一解释器时会拒绝覆盖，
请使用新克隆目录。缓存完整时可设置 `UV_OFFLINE=true` 节约下载；离线缺包会报错。

旧的 `uv.lock` 与冲突的训练 extras 已移除。训练环境由 setup 管理，**不要运行 `uv sync`
覆盖这套继承环境**。补充包采用 `--no-deps` 保持验证组合；veRL 的 NumPy、SwanLab 的 protobuf
声明约束与继承版本仍有差异，运行通过不等于依赖元数据完全无冲突。

### 每个运行终端的环境变量

激活基础 conda 环境、进入仓库后执行：

```bash
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY_INCLUDE="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("include"))')"
test -f "$PY_INCLUDE/Python.h"
export C_INCLUDE_PATH="$PY_INCLUDE${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export SHOPSIM_BASE_URL=http://127.0.0.1:5701
```

本机路径通过环境变量传入。下载优先直连，网络确有需要时再配置代理。

## 3. ModelScope 下载

`configs/model.yaml` 默认选择：

```yaml
source: modelscope
model_id: Qwen/Qwen3.5-2B
revision: master
local_dir: models/Qwen3.5-2B
```

保留文件原有 `serving` 配置，它控制服务名称、端口、上下文、TP 和显存比例。

```bash
.venv/bin/python scripts/download_model.py
.venv/bin/python scripts/download_model.py --local-only
```

完整权重约 4.55 GB，下载后生成包含来源及 SHA256 的 `model-source.json`。
`--local-only` 检查文件存在、分片完整性和来源标记，不重新计算所有哈希，也不下载。
不要用未登记的非空目录作为下载目标。`master` 会移动，严格复现应选不可变 revision 并保存哈希清单。

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

测试需要 pytest；基础环境没有时可先执行 `uv pip install --python .venv/bin/python pytest`。

```bash
.venv/bin/python -m pytest \
  tests/test_runtime_installation.py tests/test_export_grpo_cli.py \
  tests/test_merge_lora_adapter.py tests/test_grpo_dynamic_sampling_config.py \
  tests/test_model_source.py tests/test_sglang_token_counter.py \
  tests/test_context_window.py tests/test_environment_manifest.py \
  tests/test_evaluation_dataset.py -q
```

## 10. 常见问题

| 现象 | 检查方向 |
|---|---|
| setup 版本不匹配 | 检查 MAIN_PYTHON 和 runtime.json，不关闭检查来混用环境 |
| Python.h 缺失 | 查看实际解释器 include 路径，仅有目录或 Imaging.h 不够 |
| CPU 内核 / adapter 错误 | 重新运行 setup；补丁会校验哈希，未知源文件会拒绝修改 |
| checkpoint 路径错误 | 使用实际运行目录的 global_step_*/actor，各仓库 outputs 不会自动共享 |
| 输出非空 | 换新目录，不随意删除已有结果 |
| GRPO 无更新 | 检查有效组及 optimizer_updated，单步 SFT 能力可能不足 |
| ContextBudgetError | 轨迹超出输入预算，修改预算后的结果不能混入原协议对比 |
| 仿真无可用槽位 | 检查并发及中断残留 lease，确认无其他任务后再处理 |

运行前后用 `nvidia-smi` 检查 GPU，只停止自己启动的服务。模型权重、输出、虚拟环境和凭据
不提交到 Git。早期文档保留用于追溯，当前安装与运行命令以本文为准。
