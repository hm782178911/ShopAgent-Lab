# 可行性验证方法

本项目只支持 Baseline → SFT → GRPO → Evaluation。详细可执行命令见
[README](../README.md)，当前验证范围见 [validation.md](validation.md)。

## 安装隔离

在新克隆目录运行 setup。脚本准备固定 uv、项目内 Python 和两套独立虚拟环境，
安装完整哈希锁并检查依赖一致性。已有继承环境、外部 Python 或环境软链接会被拒绝。
GPU 入口使用项目内 CUDA 编译器、头文件和库，不依赖 `/usr/local/cuda`。
宿主机仍须提供 Linux、NVIDIA 驱动、GPU 和 README 列出的基础系统工具。

主环境可执行以下检查：

```bash
.venv/bin/python -m shopping_grpo.runtime
cat .venv/pyvenv.cfg
```

`include-system-site-packages` 应为 false。Python 基础目录应位于项目 `.runtime/python`。
setup 会为 NVIDIA CUDA wheel 补齐 `lib64` 和 `libcudart.so` 的相对链接，
供 FlashInfer JIT 链接使用；只操作项目环境，已有异常链接会报错。

## 参数更新验收

SFT：只运行一步时显式关闭 warmup，使用训练/验证数据子集，不使用最终评估集。
确认 optimizer 执行、LoRA B 张量非零、adapter 保存，合并模型的结束符包含 tokenizer EOS。

GRPO：使用具备购物能力的 SFT 模型，最多验证一次真实更新。
确认轨迹产生有效奖励差异、日志中 `training/optimizer_updated=1`、学习率非零，
并保存 `global_step_1/actor`。仅启动服务、完成 rollout 或出现 SKIPPED 不算更新成功。
所有重试使用新的输出目录，不覆盖已有 checkpoint。

## 导出和推理验收

通过 `export_grpo.sh` 导出 FSDP checkpoint；脚本检测并合并 adapter，保留来源清单。
用 SGLang 重载最终模型，检查 `/v1/models`、聊天、工具调用和 `/v1/tokenize`。
两条固定评估任务只检查运行流程；必须查看 status_counts 和轨迹 error，不能把 completed 当作购买成功。
Base、SFT、GRPO 使用同一任务集和推理参数，每次切换实际加载的模型。

## 依赖适配边界

- veRL 的动态采样补丁和两个 SGLang 补丁由 setup 自动执行，检查源文件哈希并可重复运行。
- GPU 引擎补丁取消 veRL 内部强制 CPU 导入；adapter 补丁统一使用 lora_as_adapter 判定。
- cachetools 是 veRL AgentLoop 实际需要但未声明的依赖，已显式加入 runtime.in。
- 构建工具也有独立哈希锁；禁用隐式构建环境，避免旧版 Gym 使用未来变化的 setuptools。
- EOS 对齐发生在合并阶段，保留原模型 EOS 并加入 tokenizer 的对话结束符。

上述适配不改变数据切分、环境契约、Reward v3、成功判定或评估预算。
不要移除版本/哈希检查来混用未经验证的依赖。
