# ModelScope + SGLang implementation plan

Approved design: independent repository; reuse existing datasets, ShopSimulator v2.1,
Reward v3, action-only LoRA SFT, veRL GRPO, and Final-200 Clean evaluation.
No runtime environments, credentials, checkpoints or generated logs copied.

- [ ] Add configs/model.yaml and a model resolver/download entrypoint. Test local
  path resolution from a different cwd, invalid config, incomplete models and cache reuse.
- [ ] Replace serving with SGLang; route evaluation token counting to SGLang's
  tokenizer endpoint without disabling observation/context budgets. Test requests.
- [ ] Pin compatible SGLang/torch/transformers/veRL dependencies in uv.lock;
  install in this repository's own environment and verify actual imports.
- [ ] Select SGLang rollout, adapt runtime preflight, preserve dynamic sampling
  patch and validate Hydra config. Keep response budget 8192 and capacity 24576.
- [ ] Document setup, ModelScope download, baseline, SFT/merge, GRPO/export,
  evaluation, provenance, limitations and validation status. Test shell syntax,
  focused regression tests, data split checks and command construction.
- [ ] GPU rollout and optimizer smoke require free GPUs; do not interrupt the
  original repository's running training or claim unexecuted GPU verification.
