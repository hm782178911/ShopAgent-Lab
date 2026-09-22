# 来源与授权状态

- 本仓库衍生自 https://github.com/YYHDBL/shopping-grpo-longhorizon 。
  精确源提交及本地修复列于 UPSTREAM.json；原 README 保留在 docs/upstream-README.md。
- ShopSimulator 来源见 environments/ShopSimulator/EMBEDDED_SOURCE.json，
  上游为 https://github.com/ShopAgent-Team/ShopSimulator 。内嵌的是项目修改版。
- SFT 800/200、GRPO 1000/50 和 Final-200 Clean 200 条直接沿用原作者数据。
  采集模型、哈希和任务划分见 data/*/metadata.json；没有重新宣称自行采集。
- Reward v3 是原项目规则代码；正式额外 LLM Judge 的配置见 docs/evaluation.md。
- 本次新增：ModelScope 配置/下载与来源记录、SGLang serving/rollout/tokenize
  适配、独立依赖锁和复现入口。现有 SFT、环境、奖励和动态采样方法沿用上游。
- 检查时两个上游均未发现明确的项目/数据许可证。此处仅建立本地衍生仓库，
  不为上游代码或数据擅自添加 MIT/Apache 授权。公开发布前需确认再分发许可。
- veRL、SGLang、ModelScope、Qwen 等第三方软件/模型遵守各自许可证；模型下载
  来源也不替代模型本身的许可证。密钥、模型权重和训练输出不纳入 Git。
