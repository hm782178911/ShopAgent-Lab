# Full SGLang reproduction results

The standalone installation completed the full Baseline -> SFT -> GRPO -> export -> evaluation workflow. Training used the original author dataset and Qwen3.5-2B from the pinned ModelScope revision in configs/model.yaml. No new teacher dataset was used.

## Protocol

- Source: commit `62a9938567a59d11fa048d1f6a15dee0ddc93c5e`.
- Hardware: two NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs (96 GB each); driver 595.84.
- Independent runtime: SGLang 0.5.16, PyTorch 2.11.0, Transformers 5.12.1; dependency locks in configs.
- SFT: single GPU, 3 epochs, 300 steps; training took about 2 h 1 min.
- GRPO: two GPUs, SGLang rollout, final step 500; 7 h 23 min. The final skip counter is 51; 500 scheduled steps must not be described as 500 optimizer updates.
- All six runs evaluated the same fixed 200 tasks using Reward v3. Model-file hashes and evaluation protocol match within each backend pair.
- Temperature 0, top_p 1, max_steps 35, max_tokens 512, context window 24576. Full metrics and hashes: [JSON](results/sglang-reproduction.json).
- SGLang used the independent runtime. vLLM 0.25.1 comparison used an existing separate installation; it is not a second standalone-installation validation. Serving configurations and software stacks differ.

## Results

| Model | Serving backend | Strict success | Mean reward | Mean steps | Evaluation minutes |
|---|---|---:|---:|---:|---:|
| base | sglang | 0.5% (1/200) | -0.2364 | 6.79 | 9.92 |
| base | vllm | 1.0% (2/200) | -0.1486 | 5.55 | 12.69 |
| sft | sglang | 65.0% (130/200) | 0.5637 | 9.40 | 5.81 |
| sft | vllm | 69.5% (139/200) | 0.6128 | 9.24 | 7.05 |
| grpo500 | sglang | 67.0% (134/200) | 0.5728 | 10.40 | 6.35 |
| grpo500 | vllm | 71.0% (142/200) | 0.6137 | 10.97 | 8.97 |

## Interpretation and limits

SGLang serving improves from 65.0% after SFT to 67.0% after GRPO (+2.0 percentage points). vLLM serving of the same weights improves from 69.5% to 71.0% (+1.5 points). The final model has 129 shared successes, 5 SGLang-only successes and 13 vLLM-only successes. Exact paired McNemar p=0.0963 does not establish general backend superiority from this single run.

SGLang final evaluation takes 6.35 minutes versus vLLM 8.97 minutes. This includes simulator/client overhead and different generated trajectories; it is not a fixed-token throughput benchmark. Only SGLang GRPO was trained in this experiment, so these results do not isolate the effect of the training rollout backend.

Completed tasks include failed episodes. Final SGLang status: 191 done, 7 error, 2 invalid_action_limit; vLLM: 197 done, 2 error, 1 invalid_action_limit. No missing tasks. Error outcomes remain in the denominator.

## Reproduce and retain evidence

Follow the repository [README](../README.md) for installation, fixed model download, baseline, full SFT, full GRPO, export and evaluation. Use a fresh output directory for each model/backend; never silently reuse an old summary. Preserve the fixed evaluation task file, configuration, commands, model hashes, raw trajectories and summaries. Do not train on evaluation tasks.

The full run artifacts remain outside Git: final SFT adapter and merged model, final GRPO model, resumable checkpoints at steps 100/200/300/400/500, all six raw trajectory files, HTML reports and logs. A verified evidence archive also retains the experiment launcher and evaluator adapter. Machine-specific launchers and model binaries are not repository dependencies. The archive digest is included in the result JSON.

This full run supplements the earlier [small-sample feasibility checks](validation.md). Cross-hardware reproducibility and repeated-seed effect estimates remain untested.
