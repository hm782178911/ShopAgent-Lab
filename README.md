# Shopping GRPO：ModelScope + SGLang

> 2026-09-22：已通过基于现有 xlm_py312 SGLang 环境的单步可行性验证，详见 [验证记录](docs/feasibility-validation.md)。默认依赖锁与安装入口尚未同步到这套环境，请勿将下述 setup 路线视为已验证的新机器一键安装。

从作者现成数据复现 **Baseline → LoRA SFT → 在线 GRPO → Evaluation**。
默认模型 Qwen3.5-2B；模型从 ModelScope 下载。独立推理服务和 veRL rollout
均使用 SGLang。环境保持 ShopSimulator v2.1 / Reward v3 / observation v2。

本仓库是有来源记录的衍生实现，作者数据与历史成绩不能冒充本项目的新结果。
见 [SOURCES.md](SOURCES.md)、[验证记录](docs/validation.md)。GPU 训练须另行实测。

## 1. 安装

目标平台：Linux x86_64、NVIDIA GPU、带开发头文件的 Python 3.12，另需
ShopSimulator Python 3.10、Git、patch、gzip、uv。驱动需支持锁定 PyTorch 的 CUDA。
两卡示例面向每卡约 96 GiB；更换显卡、模型或上下文长度需重新验证显存。

```bash
cd shopping-grpo-sglang
# MAIN_PYTHON 可指定已有 conda 环境的 Python 路径，setup 会建立独立 .venv。
MAIN_PYTHON=3.12 SHOPSIM_PYTHON=3.10 bash scripts/setup.sh
```

setup 用 `uv sync --frozen` 安装，不复用其他项目的 site-packages；解压并验证商品
数据、构建搜索索引，并应用有哈希检查的 veRL 动态采样补丁。不自动开始训练。
`Python.h` 缺失时使用带头文件的 Python 重建本仓库 .venv，不能仅切换 shell 的 conda。

## 2. 配置并下载模型

编辑 `configs/model.yaml`：

```yaml
source: modelscope
model_id: Qwen/Qwen3.5-2B
revision: master
local_dir: models/Qwen3.5-2B
```

```bash
.venv/bin/python scripts/download_model.py
```

下载后生成 `models/Qwen3.5-2B/model-source.json`，记录来源、请求版本和文件 SHA256。
`master` 会移动；跨机器严格复现应在配置中填写 ModelScope 不可变版本，并保存
该 manifest。重复使用本地模型时检查来源和分片完整性，不隐式联网下载。
更换模型 ID/版本请同时更换 local_dir。服务名、端口、TP、显存比例、工具解析器
在同一个 YAML 的 `serving` 中设置。其他模型的模板、LoRA targets 和 veRL 支持
仍需验证；改变模型 ID 不代表所有架构自动兼容。

## 3. 启动环境与 Baseline

终端 A：

```bash
bash scripts/start_environment.sh
```

终端 B：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_model.sh
```

终端 C：

```bash
curl http://127.0.0.1:8000/v1/models
bash scripts/baseline.sh
```

默认环境地址为 5700，SGLang 为 8000。已有原项目占用这些端口时先安排空闲窗口，
不要启动第二个冲突实例；仅在确认环境版本一致时通过 SHOPSIM_BASE_URL 连接现有服务。
评估通过 SGLang `/v1/tokenize` 保留上下文与 observation 预算，不静默关闭截断检查。

## 4. SFT 与合并

先结束占用训练显卡的独立推理服务，然后：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/sft.sh
```

使用 `data/sft/train.jsonl` 800 条、validation 200 条，action-only LoRA SFT。
脚本从 model.yaml 获取已下载的模型；完成后自动合并到 `outputs/models/sft-merged`。
可用 MODEL_CONFIG 指定另一个配置；BASE_MODEL 仍允许显式本地路径覆盖。
不要把历史作者 checkpoint 复制来冒充本次训练结果。

## 5. SGLang GRPO

保持仿真环境运行，先停止 Baseline/SFT 评估的独立 SGLang 服务。veRL 内部会启动
自己的 SGLang rollout，并同步训练后的模型权重，不连接 8000 端口。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/grpo.sh
```

默认 2 卡、response budget 8192、容量 24576、每题 4 条轨迹、500 次更新、每 10 步保存、每 50 步
验证；训练前先做 50 条验证。`configs/grpo.yaml` 管理训练设置。
脚本设置显式 rank 设备选择与 NCCL P2P fallback，并从自己的 Python 查找头文件。
GPU P2P 测试正常的机器可显式设置 NCCL_P2P_DISABLE=0；没有 NVLink 不等于不能用 P2P。

```bash
# 查看启动参数，不开始训练；需要有效的 SFT merged 模型。
bash scripts/grpo.sh --dry-run
# 使用新输出目录；不要删除正在运行的训练目录。
bash scripts/grpo.sh --output outputs/models/grpo-run2
```

`SKIPPED` 表示动态采样未凑齐有效奖励组，`training/optimizer_updated:1` 才表示
更新权重。脚本拒绝非空输出目录，当前不承诺自动续训。GPU 空闲时可做 1 步烟测，
它会额外执行验证并保存模型，必须由使用者主动运行：

```bash
bash scripts/grpo.sh --output outputs/models/grpo-smoke -- \
  trainer.total_training_steps=1 trainer.save_freq=1 trainer.test_freq=-1
```

## 6. 导出与评估

```bash
ls outputs/models/grpo/global_step_*/actor
# 将 100 换成真实存在的保存步数。
bash scripts/export_grpo.sh outputs/models/grpo/global_step_100/actor outputs/models/grpo-merged
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_model.sh outputs/models/grpo-merged
# 另一个终端
bash scripts/evaluate.sh grpo
```

用 `outputs/models/sft-merged` 启动服务、执行 `bash scripts/evaluate.sh sft` 即评估 SFT。
所有模型使用同一 Final-200 Clean。evaluate 生成轨迹与环境奖励报告；额外 DeepSeek
Rubric/Trajectory Judge 需按 docs/evaluation.md 单独配置，不是默认 GRPO 奖励模型。
导出需要实际 checkpoint，完整的 SGLang 训练/保存/导出链以 GPU 实测为准。

## 7. 验证与来源

```bash
.venv/bin/python -m pytest tests/test_model_source.py tests/test_sglang_token_counter.py tests/test_context_window.py -q
.venv/bin/python scripts/evaluate_configured.py baseline --dry-run
```

数据文件和 split 不修改。`experiments/` 与 docs/upstream-README.md 保留的是上游
历史结果，不是这个 SGLang 实现的测量值。详细限制与实际完成的检查见 docs/validation.md。
