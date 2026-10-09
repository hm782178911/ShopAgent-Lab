# 验证范围

当前安装与运行命令以 [README](../README.md) 为准。环境来自固定公开发行包，
不继承 conda、系统 site-packages 或本地 SGLang 源码。

已在 Linux x86_64、两张 96 GB NVIDIA GPU、驱动 595.84 上验证：

- 新目录首次安装、重复安装、缺少 uv 时自动下载并核对 SHA256。
- Python 3.12.13 / 仿真 Python 3.10.20 均在项目目录内，包含开发头文件。
- 完整依赖哈希锁、主环境与仿真环境的 `uv pip check`。
- Gym 源码包在固定 setuptools/wheel/packaging 下重新构建。
- 核心包来源与 CUDA 编译器均属于项目环境。
- ModelScope 固定 revision 小文件下载、已有完整模型逐文件 SHA256 对照。
- 单步 SFT 的非零权重更新、合并和结束符对齐。
- 双卡 GRPO 的一次有效 optimizer 更新、checkpoint 保存与导出合并。
- Base、单步 SFT、GRPO 最终模型的 SGLang 重载和每种两条固定任务评估。

验证使用新的 HOME 与干净 PATH，实际触发过 FlashInfer 冷启动编译。
没有把预先存在的用户包、SGLang 源码或编译缓存当作必要条件。
模型是显式输入，模型下载缓存可复用；完整权重来源经公开 revision 文件哈希核对。

单步 SFT 只验证工程链路。GRPO 有效更新验证使用一个已完成完整 SFT 的 checkpoint，
并未宣称一步 SFT 加一步 GRPO 就能得到可用购物能力。
两条任务只检查调用流程，不能推导成功率或训练收益；模型行为错误也需如实保留。
后续已完成该独立环境的全量训练与正式 200 题双后端对比，见 [完整复现结果](reproduction-results.md)。
下文小样本结果保留作为安装与工程流程的验证记录；跨硬件验证尚未进行。

验收方法见 [可行性验证](feasibility-validation.md)。运行日志、模型与报告保留在 outputs，
不提交到 Git，也不依赖其他仓库的输出目录。

## 小样本实际结果

同一组两条固定任务：Base 为 1 条 done、1 条 invalid_action_limit；单步 SFT 的两条均触发
ContextBudgetError（长轨迹超过上下文预算）；GRPO 最终模型两条均 done，其中 1 条严格成功。
这验证了服务重载和工具调用链路，不代表单步 SFT 已经学会任务，也不是正式效果对比。
