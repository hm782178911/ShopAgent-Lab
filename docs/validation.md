# 验证记录

## 2026-09-22 更新

已使用 xlm_py312 的 SGLang 完成单步可行性验证：非零 SFT 更新、GRPO 更新、checkpoint 导出合并、模型重载及少量任务评估。详情与限制见 [可行性验证记录](feasibility-validation.md)。默认 uv.lock 仍为旧环境配置，不代表该安装路线已经验证通过。下面保留的是此前检查记录。

本文件记录实际执行的检查，不能把静态检查当作 GPU 端到端训练通过。

- ModelScope 配置/路径/来源检查、完整权重分片检查：6 项初始测试通过。
- SGLang 0.5.12 上游源码确认支持 Qwen3 coder 工具解析和 /v1/tokenize 的
  messages/tools、prompt、count 协议。
- 原仓库 observation projection 的部分旧格式测试已有失败；继续保留测试，
  不将它们跳过或删除以制造全绿结果。最终统计在本文件末尾更新。
- 后台原仓库训练占用两张 GPU，本仓库尚未执行 GPU serving、GRPO optimizer、
  checkpoint export 或全量评估。上述能力必须在空闲窗口进行 smoke 验证。
