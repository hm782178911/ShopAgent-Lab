# SGLang 流程可行性验证（2026-09-22）

## 范围与环境

本次验证使用用户已有的 `xlm_py312` conda 环境中的 SGLang，目标是验证接口、训练更新、保存和模型重载，不进行完整训练或 200 条效果评估。

训练依赖安装在本仓库 `.venv-smoke` 中；它通过 `--system-site-packages` 继承 conda 包。本仓库 `.venv` 现在链接到 `.venv-smoke`，之前无法导入 SGLang 的环境保留为 `.venv-pinned-backup`。原仓库模型仅作为只读输入；原 conda 环境未安装或升级包。

实际核心版本：Python 3.12.13、SGLang `0.5.15.post2.dev468+g85ac56c82`、Transformers 5.12.1、PyTorch 2.11.0、veRL 0.8.0、PEFT 0.20.0、Ray 2.56.1。SGLang 源码提交为 `85ac56c82308b8f0e86b3aa8542b61af0de666aa`。

**重要：本次不是仓库原 `pyproject.toml` / `uv.lock` 中 SGLang 0.5.12 环境的验证。** 旧锁文件和默认检查器的版本列表尚未切换为此次运行配置。不要运行 `uv sync` 覆盖当前验证环境，也不要将本结果表述为“全新机器直接 setup 已通过”。此次 GRPO 驱动使用明确的运行时版本 profile，再执行原有的环境、CUDA、显存预算、生命周期与动态采样补丁检查；并非关闭这些检查。

## 已执行的流程

| 环节 | 验证方法与证据 |
|---|---|
| ModelScope 连接 | 直连下载 Qwen3.5-2B 的 `config.json`；完整权重复用本地文件，未重复下载 |
| Base 推理与工具循环 | SGLang 服务处理两个固定评估任务；聊天、工具调用、`/v1/tokenize` 均实际执行 |
| SFT 单步 | `max_steps=1`、`warmup_ratio=0`、梯度累积 1、训练样本选择 2 条，保存 LoRA |
| SFT 权重检查 | 186 个 LoRA B 张量，最大绝对值约 `1e-4`；已确认不是只跑前向或零学习率空更新 |
| SFT 合并与重载 | `sft-merged-updated`，执行独立 SGLang 少量任务评估 |
| GRPO 单步 | 两卡 SGLang rollout、有效奖励组筛选、一次 optimizer 更新、保存 `global_step_1` |
| GRPO 更新证据 | `training/optimizer_updated=1`，梯度范数约 0.211886，学习率 `1e-6`，训练进程退出码 0 |
| GRPO 权重检查 | 导出的 284 个 LoRA B 张量，最大绝对值约 `9.98e-7` |
| GRPO 导出与合并 | FSDP 分片导出后，再合并 LoRA，得到 `grpo-final` |
| GRPO 最终推理 | SGLang 重载最终模型，两条固定任务均正常结束，无执行错误 |

两条评估 task ID 为 8187 和 17354，来自评估集，不用于训练。少量任务中的失败或非法动作超限反映模型行为，不等于推理服务崩溃；不能从两条任务推导模型成功率。

实际评估终态：Base 两条均为 `invalid_action_limit`；非零更新后的单步 SFT 两条分别执行 29、33 步后触发 `ContextBudgetError`（输入 23693、23833 tokens，超过 23552 的输入预算）；GRPO 最终模型两条均为 `done`、各 8 步、无执行错误。单步 SFT 已验证模型重载、连续生成与工具调用，但未通过完整任务完成验证。上下文保护正常拦截了超限请求，没有为了通过测试放宽评估限制。

### 必须区分的两个训练起点

SFT 测试从本地 Qwen3.5-2B 开始，只更新一步。GRPO 最终成功测试使用原仓库已完整训练的 `sft-merged` 权重作为起点，而不是宣称单步 SFT 已具备完整购物能力。

因此本次结论是：**SFT 单步训练链和 GRPO 单步训练链都可运行，GRPO 保存后的模型也可再次推理。** 它不等于从全新 Base 仅训练一步 SFT、一步 GRPO 就取得可用任务效果。

## 遇到的问题与处理

### 1. 单步 SFT 被 warmup 占用

第一次默认 warmup 测试完成了 backward，但学习率为 0，LoRA B 仍全零，未将其算作有效参数更新。随后显式设置 `--warmup-ratio 0`，保存到独立的 `sft-lora-updated` 并确认权重变化。两次产物分开保留。

### 2. veRL 强制使用 CPU 引擎导入 SGLang

veRL 的 SGLang loader 强制设置 `SGLANG_USE_CPU_ENGINE=1`，导致当前 GPU 版内核包缺少 `copy_all_layer_kv_cache_cpu`。该赋值发生在代码内部，无法用启动命令上的同名环境变量覆盖。

修复保存在 `patches/verl-0.8.0-sglang-gpu-import.patch`，仅修改隔离环境里的 veRL，取消错误的 CPU 引擎强制设置。

### 3. 合并权重模式被错误当成 adapter 模式

配置 `actor_rollout_ref.model.lora.merge=true` 时，veRL 仍按 `lora_rank>0` 开启推理 adapter 模式，并在请求中添加 adapter 名称。先出现 `get_hidden_dim not implemented for a/e`，随后出现请求未启用的 adapter 错误。

修复保存在 `patches/verl-0.8.0-sglang-merged-rollout.patch`：启动和生成请求都使用 `lora_as_adapter` 判断。训练仍采用 LoRA；推理接收合并后的完整权重，没有改变奖励或绕过有效性筛选。

注意 uv 安装可能使用硬链接。修改依赖文件时必须采用复制/原子替换，避免把缓存或其他环境一起改掉。本次最终补丁文件已经与缓存断开硬链接，并核对原仓库相关依赖文件恢复为原始内容。

### 4. 停止 token 不一致

原 SFT 产物的 generation config 为 `eos_token_id=248044`（`<|endoftext|>`），tokenizer 的对话结束 token 为 248046（`<|im_end|>`）。SGLang 的原始生成接口会继续生成越过对话结束符，导致多工具调用或提前结束等无效轨迹。

通过记录原始生成内容确认该问题后，在 `sft-eos-aligned` 中创建独立配置：

```json
{"eos_token_id": [248044, 248046]}
```

其余模型文件通过链接只读复用，权重未改动，原仓库文件未改动。修正后，实际停止 token 变为 248046，工具循环和成功购买恢复，随后完成 GRPO 更新。最终 `grpo-final/generation_config.json` 保留两个结束 token。

### 5. 中断测试后的仿真槽位

中断的 Ray 任务可能来不及释放仿真 lease。出现无可用环境资源后，确认没有其他训练/评估，再调用环境的 release API 释放残留槽位；没有重启服务或更改数据。

## 产物与可核验证据

全部位于本仓库 `outputs/feasibility-20260922/`：

- `runtime-audit.json`：实际解释器、依赖版本、SGLang 提交与补丁哈希。
- `grpo-runtime-profile.json`：GRPO 测试采用的版本检查 profile。
- `commands/`：此次调用的验证脚本。包含本机绝对路径，属于运行记录，不是可移植的公共安装入口。
- `modelscope.log`：ModelScope 小文件直连验证。
- `sft-update.log`、`sft-update-verification.json`：非零学习率 SFT 更新。
- `sft-lora-updated/`、`sft-merged-updated/`：单步 SFT 产物。
- `sft-updated-summary.json`、`sft-updated-trajectories.jsonl`：该产物的重载评估。
- `sft-eos-aligned/eos-adjustment.json`：GRPO 起点的结束符修正记录。
- `grpo-eos-run.log`：成功的 GRPO 单步运行日志。
- `grpo-step1-eos/global_step_1/`：实际 checkpoint 与优化器状态。
- `grpo-update-verification.json`：GRPO adapter 非零权重证据。
- `grpo-export/`、`grpo-final/`：导出及最终合并模型。
- `grpo-summary.json`、`grpo-trajectories.jsonl`：最终模型的两条任务推理。

调试时临时增加的原始文本打印已撤回，必要兼容补丁保留。失败尝试日志保留用于追溯，不混入成功产物。

## 本次没有验证的事项

- 全量 ModelScope 权重下载：为节约流量，复用了已有模型。
- 新建 ShopSimulator 环境/索引：复用 5700 上已有的同版本仿真服务。
- 从零安装的默认锁文件流程：当前验证使用 conda 继承环境与补充依赖。
- SGLang 全量 SFT、500 步 GRPO 和 200 条正式效果评估。

后续完整训练前，应先将本次实际版本 profile、依赖补丁和 EOS 配置步骤纳入正式安装/启动说明，再做完整实验。此次单步结果不能替代 vLLM 标准复现的正式成绩。
