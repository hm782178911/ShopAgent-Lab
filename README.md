# ShopAgent-Lab

面向购物 Agent 的后训练实践，涵盖 SFT、GRPO 与评估，基于 SGLang。

本项目基于 [shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon)
复现并进行 SGLang 适配，沿用原作者数据、ShopSimulator 仿真环境及 Reward v3。
模型通过 ModelScope 获取。来源和授权状态见 [SOURCES.md](SOURCES.md)，上游说明见
[docs/upstream-README.md](docs/upstream-README.md)。上游历史成绩不代表本实现的效果。

## 1. 验证范围：先读这里

唯一工作流：**Baseline → LoRA SFT → 在线 GRPO → 导出 → Evaluation**。

在 Linux、两张 96 GB NVIDIA GPU、已有 Python 3.12 conda SGLang 环境上，已从
GitHub 独立克隆，完成完整模型下载、独立仿真环境构建、SFT 单步更新、GRPO 单步更新、
模型导出合并和重载评估。23 项针对模型配置、上下文、数据及环境契约的测试通过。

必须区分：

- SFT 单步测试确认参数发生变化，不代表模型已经学会购物。
- 本次单步 SFT 直接接 GRPO，96 条轨迹未组成有效训练组，更新次数为 0。
- 使用此前完整训练的 SFT checkpoint 后，GRPO 完成一次有效更新，导出的模型完成两条评估任务。
- 只用两条固定评估任务验证接口和闭环；没有在 SGLang 下重跑完整训练或正式 200 题效果实验。

**以下是基于已有 conda SGLang 环境的已验证配置。全新机器从零安装 SGLang 尚未验证。**
`scripts/setup.sh`、`pyproject.toml` 和 `uv.lock` 仍对应旧依赖配置，且 setup 尚未自动应用
两个 SGLang 补丁。走本文流程时不要再运行 `setup.sh` 或 `uv sync` 覆盖环境。

## 2. 克隆与前置环境

需要 Git、uv、patch、gzip、可用 NVIDIA 驱动，以及包含 Python 开发头文件的 Python 3.12。
ShopSimulator 单独使用 Python 3.10。两卡显存预算按每卡约 96 GB 验证，其他硬件需另测。

```bash
git clone git@github.com:hm782178911/ShopAgent-Lab.git
cd ShopAgent-Lab
conda activate xlm_py312
```

`xlm_py312` 是本次使用的已有环境名称；不要仅创建同名空环境就认为依赖已满足。
主要版本如下，SGLang 是源码开发版本：

| 组件 | 验证版本 |
|---|---|
| Python | 3.12.13 |
| SGLang | 0.5.15.post2.dev468+g85ac56c82 |
| SGLang 源提交 | `85ac56c82308b8f0e86b3aa8542b61af0de666aa` |
| PyTorch / Transformers | 2.11.0 / 5.12.1 |
| veRL / PEFT | 0.8.0 / 0.20.0 |
| Ray / TensorDict | 2.56.1 / 0.10.0 |
| NumPy / ModelScope | 2.3.5 / 1.38.1 |

先核对 conda 环境中提供的基础依赖：

```bash
python - <<'PY'
import sys, importlib.metadata as m
print(sys.executable)
for name in ('sglang', 'torch', 'transformers', 'numpy', 'modelscope'):
    print(name, m.version(name))
PY
```

### 2.1 创建独立训练环境

在全新克隆目录执行。`.venv` 不应提交到 Git；不要把它链接到其他项目的环境。

```bash
CONDA_PYTHON="$(command -v python)"
uv venv --system-site-packages --python "$CONDA_PYTHON" .venv
UV_LINK_MODE=copy uv pip install --python .venv/bin/python --no-deps \
  -r docs/requirements-conda-overlay.txt
UV_LINK_MODE=copy uv pip install --python .venv/bin/python --no-deps -e .
```

依赖列表记录了本次向独立环境补充的包，基础包继承 conda。`--no-deps` 用于保持已验证的
基础组合，不是通用的全新环境安装方案；例如 veRL 对 NumPy、SwanLab 对 protobuf 的
声明约束与继承版本并不完全一致，运行验证通过不等于依赖解析完全无冲突。
有本地 uv 缓存时，可在两个安装命令中加 `--offline`，本次验证就是如此安装的。
`UV_LINK_MODE=copy` 避免后续补丁通过硬链接改到缓存或其他环境。

### 2.2 应用 veRL 补丁

动态采样补丁由仓库脚本校验版本和哈希；另两个补丁适配上述 SGLang 开发版本。

```bash
.venv/bin/python scripts/apply_verl_dynamic_sampling_patch.py
SITE="$PWD/.venv/lib/python3.12/site-packages"
patch --batch --forward -d "$SITE" -p1 \
  -i "$PWD/patches/verl-0.8.0-sglang-gpu-import.patch"
patch --batch --forward -d "$SITE" -p1 \
  -i "$PWD/patches/verl-0.8.0-sglang-merged-rollout.patch"
.venv/bin/python scripts/apply_verl_dynamic_sampling_patch.py --check
```

两个 SGLang 补丁分别取消 veRL 错误强制的 CPU 引擎导入、修正合并权重模式下误启用 adapter
的问题。只对新安装、独立复制的 veRL 应用一次；重复执行 patch 可能报告已应用，不要强行重打。

### 2.3 每个运行终端的环境变量

每个新终端先激活 conda 并进入仓库，再执行：

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

`Python.h` 检查失败时先补齐匹配 Python 的开发头文件。目录存在或只有 `Imaging.h` 并不够。
`C_INCLUDE_PATH` 帮助运行时编译器找到头文件，不会安装缺失文件。下载优先直连，网络确实
不通时再按自己的网络环境配置代理；本地服务始终加入 NO_PROXY。

## 3. 安装并启动独立仿真环境

从仓库内压缩商品库解压，核对哈希，再构建搜索索引：

```bash
SHOP_ROOT="$PWD/environments/ShopSimulator"
SHOP_ENV_ROOT="$SHOP_ROOT/shop_env"
uv venv --python 3.10 "$SHOP_ROOT/.venv-shopsim"
UV_LINK_MODE=copy uv pip install --python "$SHOP_ROOT/.venv-shopsim/bin/python" \
  -r "$SHOP_ENV_ROOT/requirements.txt"
gzip -cd "$SHOP_ENV_ROOT/data/fine_items_eval_train_all.json.gz" \
  > "$SHOP_ENV_ROOT/data/items_eval_train.json"
printf '%s  %s\n' \
  57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f \
  "$SHOP_ENV_ROOT/data/items_eval_train.json" | sha256sum -c -
```

**哈希校验成功后**继续构建索引：

```bash
(
  cd "$SHOP_ENV_ROOT"
  PYTHONPATH=. "$SHOP_ROOT/.venv-shopsim/bin/python" scripts/build_index.py
)
```

终端 A 启动环境并保持运行：

```bash
SHOPSIM_PORT=5701 bash scripts/start_environment.sh
```

本文统一使用 5701，避开其他项目可能占用的默认 5700。另一个已配置环境变量的终端验证：

```bash
.venv/bin/python scripts/smoke_shop_env.py \
  --base-url "$SHOPSIM_BASE_URL" --task-id 8187 \
  --output-dir outputs/smoke/environment
```

本项目固定 ShopSimulator Environment v2.1、Reward v3、observation v2、tool schema v2。
奖励来自仿真环境规则；默认流程不需要外部 LLM 裁判或 API Key。

## 4. ModelScope 模型配置与下载

编辑 `configs/model.yaml`：

```yaml
source: modelscope
model_id: Qwen/Qwen3.5-2B
revision: master
local_dir: models/Qwen3.5-2B
```

文件原有 `serving` 配置保留；其中可配置服务名称、端口、上下文、TP 和显存比例。

```bash
.venv/bin/python scripts/download_model.py
BASE_MODEL="$(.venv/bin/python scripts/download_model.py --local-only)"
```

完整权重约 4.55 GB。下载完成会生成 `model-source.json`，记录来源及文件 SHA256；
`--local-only` 会核对文件完整性和来源标记，不隐式下载。不要把未登记的非空模型目录直接
当作下载目标。`master` 是可变版本，严格跨机器复现应选择不可变 revision 并保存哈希清单。

## 5. Base 推理与小规模评估

终端 B 启动推理服务：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_model.sh
```

终端 C 检查模型服务并取固定评估集的前两条任务。这里只创建子集，不修改原数据：

```bash
curl --fail http://127.0.0.1:8000/v1/models
mkdir -p outputs/smoke
head -n 2 data/evaluation/tasks.jsonl > outputs/smoke/tasks.jsonl
EVAL_LABEL=base
mkdir -p "outputs/smoke/$EVAL_LABEL"
.venv/bin/python scripts/evaluate_shop_benchmark.py \
  --benchmark outputs/smoke/tasks.jsonl \
  --output "outputs/smoke/$EVAL_LABEL/trajectories.jsonl" \
  --summary "outputs/smoke/$EVAL_LABEL/summary.json" \
  --base-url "$SHOPSIM_BASE_URL" \
  --model shopping-agent --llm-base-url http://127.0.0.1:8000/v1 --api-key EMPTY
.venv/bin/python scripts/build_eval_report.py --run-dir "outputs/smoke/$EVAL_LABEL"
```

报告为 `outputs/smoke/base/report.html`。默认工具观察与上下文预算保持启用，SGLang
`/v1/tokenize` 用于计数。`completed_tasks` 表示任务已处理，可能包含 error；检查
`status_counts`、轨迹中的 `error` 和 `strict_success_rate` 才能判断执行及任务结果。

需要正式评估时，`bash scripts/baseline.sh` 或 `bash scripts/evaluate.sh 标签`
会运行完整 Final-200 Clean，不属于单步可行性验证。

## 6. SFT：选择单步验证或完整训练

保持仿真环境运行，先在终端 B 用 Ctrl+C 停止独立模型服务，为训练释放 GPU。

### 单步可行性验证

```bash
BASE_MODEL="$(.venv/bin/python scripts/download_model.py --local-only)"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_lora_sft.py \
  --model "$BASE_MODEL" --train data/sft/train.jsonl \
  --output outputs/smoke/sft-lora \
  --max-steps 1 --warmup-ratio 0 --train-count 2 \
  --gradient-accumulation-steps 1 --gradient-checkpointing \
  --attention-implementation sdpa --logging-steps 1
.venv/bin/python scripts/merge_lora_adapter.py \
  --base-model "$BASE_MODEL" --adapter outputs/smoke/sft-lora \
  --output outputs/smoke/sft-merged --bf16
```

`warmup-ratio=0` 避免唯一一步学习率为零。可核对 LoRA B 非零：

```bash
.venv/bin/python - <<'PY'
from safetensors import safe_open
with safe_open('outputs/smoke/sft-lora/adapter_model.safetensors', framework='pt') as f:
    values = [float(f.get_tensor(k).abs().max()) for k in f.keys() if 'lora_B' in k]
assert values and max(values) > 0
print('LoRA B tensors:', len(values), 'max abs:', max(values))
PY
```

### 完整 SFT

如果目标是正式训练，使用 800 条训练数据和 200 条验证数据：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/sft.sh
```

这是完整训练，会自动合并到 `outputs/models/sft-merged`。它与上面的单步测试是二选一的
执行规模，不要误以为单步已经充分训练。显存与时间取决于实际序列长度，LoRA 参数少也仍需反向计算。

## 7. 对齐 SFT 的结束 token

Qwen3.5 合并产物可能只将 248044（`<|endoftext|>`）记为 EOS，而对话结束 token 为
248046（`<|im_end|>`）。在 SGLang 原始生成接口中，这可能导致生成跨越对话边界。
对自己新合并的模型执行以下操作，保留原 EOS 并加入 tokenizer EOS：

```bash
# 单步模型；完整训练时改为 outputs/models/sft-merged
SFT_MODEL="$PWD/outputs/smoke/sft-merged"
.venv/bin/python - "$SFT_MODEL" <<'PY'
import json, sys
from pathlib import Path
from transformers import AutoTokenizer
model = Path(sys.argv[1])
path = model / 'generation_config.json'
config = json.loads(path.read_text())
old = config.get('eos_token_id')
ids = old if isinstance(old, list) else [old]
eos = AutoTokenizer.from_pretrained(model, local_files_only=True).eos_token_id
config['eos_token_id'] = sorted({i for i in [*ids, eos] if i is not None})
path.write_text(json.dumps(config, indent=2) + '\n')
print('EOS:', old, '->', config['eos_token_id'])
PY
```

此步骤修改选定模型的 generation config。若复用既有 checkpoint，先建立独立副本或
独立配置目录，保留源文件。SFT 重载评估时用 `serve_model.sh "$SFT_MODEL"` 启动，
然后复用第 5 节评估命令，将 `EVAL_LABEL` 设为 `sft`。

## 8. GRPO 单步验证

先停止独立 SGLang 推理服务。veRL 会启动自己的 SGLang rollout，不连接 8000 上的模型服务。
仿真环境继续运行。

**使用充分训练的 SFT 模型作为起点。** 若只训练过一步 SFT，允许尝试，但可能因非法动作、
未完成轨迹等问题无法组成有效奖励组。本次就是这种情况；不要关闭有效性筛选来伪造通过。
已有 SFT checkpoint 不包含在 Git 仓库中，需要自行训练或明确提供来源。

```bash
SFT_MODEL="$PWD/outputs/models/sft-merged"
GRPO_OUTPUT="$PWD/outputs/smoke/grpo"
export CUDA_VISIBLE_DEVICES=0,1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export NCCL_P2P_DISABLE=1
```

先完成第 7 节的 EOS 对齐。输出目录必须不存在或为空，不要覆盖已有运行。
无 NVLink 不等于 P2P 一定不可用；本次设置 `NCCL_P2P_DISABLE=1` 是已验证的机器环境选择。

当前 `scripts/check_grpo_runtime.py` 默认版本仍对应旧锁文件，所以此 conda 配置暂时不能
直接用 `bash scripts/grpo.sh` 通过预检查。以下命令复用原参数构建及所有预检查，只显式
设置本次验证的三个依赖版本。它不会写入仓库的 Python 文件：

```bash
.venv/bin/python - \
  --model "$SFT_MODEL" --output "$GRPO_OUTPUT" --env-url "$SHOPSIM_BASE_URL" -- \
  trainer.total_training_steps=1 trainer.save_freq=1 \
  trainer.test_freq=-1 trainer.val_before_train=false \
  trainer.n_gpus_per_node=2 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  shopping_dynamic_sampling.max_consecutive_skipped_updates=1 <<'PY'
import json, os, subprocess, sys
from pathlib import Path
from scripts.train_grpo import parse_args, build_command
from scripts import check_grpo_runtime as check
args = parse_args()
command, environment = build_command(args)
print(json.dumps(command, indent=2))
if args.dry_run:
    raise SystemExit(0)
Path(environment['GRPO_OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
os.environ.update(environment)
check.EXPECTED_VERSIONS.update(
    sglang='0.5.15.post2.dev468', transformers='5.12.1', numpy='2.3.5'
)
sys.argv = ['check_grpo_runtime.py', *command[5:]]
check.main()
raise SystemExit(subprocess.call(command, env=environment))
PY
```

确认日志出现 `training/optimizer_updated:1`，并生成 `global_step_1/actor`。
`SKIPPED` 或只出现 AgentLoopWorker 输出不表示更新成功。单步验证关闭训练前验证并取消
学习率 warmup；一次无效更新尝试就停止，避免长时间空采样。失败后换新的输出目录再试。

完整训练仍使用上面同一个 conda profile 调用方式，将 `--` 后的单步覆盖参数去掉，
并将输出目录改为新的正式目录，即恢复 `configs/grpo.yaml` 的默认训练配置：500 步、
每 10 步保存、每 50 步验证、训练前验证。完整 SGLang 训练尚未由本次测试验证。

## 9. 导出、合并 LoRA、最终评估

从真实存在的 checkpoint 导出。单步测试使用 step 1，完整训练则改为实际保存的步数。

```bash
CHECKPOINT="$GRPO_OUTPUT/global_step_1/actor"
test -d "$CHECKPOINT/huggingface"
bash scripts/export_grpo.sh "$CHECKPOINT" outputs/smoke/grpo-export
.venv/bin/python scripts/merge_lora_adapter.py \
  --base-model outputs/smoke/grpo-export \
  --adapter outputs/smoke/grpo-export/lora_adapter \
  --output outputs/smoke/grpo-final --bf16
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_model.sh outputs/smoke/grpo-final
```

本次 veRL 导出分别保存模型和 `lora_adapter`，**需要再合并 LoRA 后才得到最终推理模型**，
不能只凭导出目录名字含有 merged 就认为适配器已合并。另一个终端复用第 5 节评估命令，
将 `EVAL_LABEL` 设为 `grpo`，生成 `outputs/smoke/grpo/report.html`。

正式对比需分别启动 Base、完整 SFT、GRPO 模型，并保持相同后端、任务集和评估参数：

```bash
# 每次先启动对应模型；标签不会替你切换正在服务的模型。
bash scripts/evaluate.sh base
# 换成完整 SFT 模型服务后：
bash scripts/evaluate.sh sft
# 换成 GRPO 最终模型服务后：
bash scripts/evaluate.sh grpo
```

这些命令各运行完整 200 题，报告位于 `outputs/evaluation/<标签>/report.html`。
严格成功要求完整的 `gold_purchase` 终态且 `reward_valid=true`，不能把 done 率当作成功率。
额外的 LLM Judge 配置见 [docs/evaluation.md](docs/evaluation.md)，不是默认奖励来源。

## 10. 检查与常见问题

```bash
.venv/bin/python -m pytest \
  tests/test_model_source.py tests/test_sglang_token_counter.py \
  tests/test_context_window.py tests/test_environment_manifest.py \
  tests/test_evaluation_dataset.py -q
.venv/bin/python scripts/evaluate_configured.py baseline --dry-run
```

测试需要环境中已有 pytest。运行前后用 `nvidia-smi` 检查 GPU；只停止自己启动的服务。

| 现象 | 检查方法 / 原因 |
|---|---|
| 克隆后 `.venv` 无法创建 | 新版已取消跟踪虚拟环境链接；旧克隆先确认是不是失效符号链接，勿删除真实环境 |
| `Python.h` 找不到 | 核对实际解释器及第 2.3 节头文件路径，conda 激活不保证脚本使用的 Python 正确 |
| CPU KV-cache 内核缺失 | 核对 GPU 导入补丁是否应用到当前独立 veRL |
| `get_hidden_dim` / adapter 请求错误 | 核对 merged-rollout 补丁、SGLang 版本及 `lora.merge=true` 配置 |
| GRPO 版本预检查失败 | 旧锁文件与本文 conda profile 不同；使用第 8 节明确版本的调用，不关闭整个检查 |
| `HFValidationError: Repo id...` | 优先确认本地 checkpoint 路径与 `actor/huggingface/config.json` 真正存在 |
| 输出目录不是空目录 | 换新目录；不承诺自动续训，也不要直接删除可能有用的 checkpoint |
| 单步训练完成但参数没变 | 检查 warmup 和实际学习率，核对 LoRA 权重及 optimizer_updated |
| `ContextBudgetError` | 轨迹已超过输入预算；属于任务失败，不能把放宽预算后的结果混入原协议对比 |
| 仿真没有可用槽位 | 检查并发任务和中断后的 lease，仅在确认无其他任务时处理残留槽位 |

代码修改、模型和数据来源需保留说明；模型权重、训练输出、虚拟环境和凭据不提交到 Git。
早期验证材料见 [docs/validation.md](docs/validation.md)，当前可执行流程以本文为准。
