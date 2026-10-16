# LSL 复现与同步诊断计划

跨平台与现有AMD环境部署命令见 [AMD_ROCM_DEPLOYMENT.md](AMD_ROCM_DEPLOYMENT.md)；GPU验证仍须在服务器完成。

所有模型/数据准备和正式实验转移到 Linux GPU 服务器。本地不再下载。本轮只增加配置、准备脚本和文档，不修改原始训练、GMM、Gate 或既有 Diagnostics 算法。原 baseline tag、main 与研究分支保留。

顺序：服务器环境与资源验证 → CUDA Off/Basic/Extended 一致性 → 1.5B smoke → 7B smoke → 单任务 → 三阶段 LSL/LoRA 对照。正式配置每个运行123/456/789三个项目指定种子；论文没有公开种子 IDs。

每次正式训练同步 Basic Diagnostics；Extended 须通过一致性和开销检查。P2 使用独立 `python -m diagnostics.probe_runner`，不插入训练。固定 Probe 需保存 input_ids、attention_mask、sample_ids、预处理及 tokenizer revision，跨阶段复用相同文件。

配置在 reproduction/configs/，包含单任务三个、7B持续学习LSL/LoRA、1.5B/7B smoke、microbatch2适配。`paper_exact` 仅指公开超参数匹配，不表示未公开的子集等细节已匹配。修改 microbatch 保留 global batch 但标记 hardware_adapted。

每个 runs/<experiment_id>/ 保存 launch.json、资源锁副本、日志、完成标记；experiment/ 保存 reproduction、diagnostics、checkpoints。实际结果入口 original_metrics.jsonl；以 Diagnostics metadata 的 experiment ID/stage/epoch 和 launch.start_time 对齐原日志。

collect_results.py 只接受完整阶段/epoch 的完成实验，按配置与资源版本隔离，输出 stage_task_matrix.csv 和 seed_summary.json；三个不同种子才可报告三种子均值和样本标准差。results_template.json 是空模板，未填虚构结果。诊断比例按分子/分母聚合，不平均 batch 比率。

CL State 保存 adapters/JL/GMM 与侧车配置，足以恢复评测；不含 optimizer/scheduler/RNG/EMA/KV，不能声称完整恢复训练。GPU parity 的合成两阶段、小 EM 预算也不等同百万 token 正式实验。

当前只达到 CPU 与静态准备层面；CUDA、FA2、7B、完整正式评测和硬件峰值均待服务器验证。详见 GPU_LAUNCH_CHECKLIST.md 与 REPRODUCTION_ISSUES.md。
# First 1.5B continual run: A10, Seed 123

Default server entry: `bash scripts/run_continual_1p5b_seed123.sh` in an activated,
accepted Full environment. Set `LSL_PARITY_RECEIPT` to the real passing receipt,
`LSL_CACHE_ROOT` to the existing model/data cache, and optionally
`LSL_EVALUATOR_AUDIT` / `LSL_OUTPUT_ROOT`. No downloads or seed sweeps occur.

The entry selects `continual_1p5b_lsl_a10`, seed 123 only. It checks GPU inventory,
all nine datasets and the model (SHA256), prints a Dry Run, performs strict
preflight (real import paths, dependencies, memory, idle GPU, matching receipt),
then launches exactly one run. Receipt must match rank128/JL256/K16/32 and model
revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. A rank4/JL8 Smoke receipt is
insufficient. Parity comparison rules and core hashes are unchanged; this script
does not automatically rerun Parity. `--preflight-only` on run_reproduction.py
performs all checks without creating the run directory or starting training.

Both new plans preserve order Igbo -> Chemistry -> Cybersecurity, epochs 3/3/1,
rank/alpha 128/256, LR 1e-4, sequence 1024, global batch 32, positive/negative K
16/32, JL256, smoothing .1. Hardware differences are explicit in each JSON:
1.5B instead of paper 7B, microbatch 1 with accumulation 32, task eval batch 1,
LM eval batch 1 instead of hardcoded 256/541. Only the dedicated plan changes
HFLM/simple_evaluate batching; task order, prompts, chat template, fewshot,
generation settings, limit 300, filters and scoring are unchanged. Batching can
change kernel/padding numerics; generation equivalence to large batches has not
been measured. These results must not be labeled paper-exact 7B reproduction.

`continual_1p5b_lsl.json` retains 1,000,000 GMM tokens as a reference.
`continual_1p5b_lsl_a10.json` explicitly uses 65,536 for BOTH positive and negative
fits. This changes the support sample, including the original budget-dependent
Negative preprocessing cap; it is a paper deviation, not just an implementation
optimization. EM, Gate, Temporal Smoothing, freezing, optimizer and loss remain
the original implementations. 24GB launches reject the million-token plan.

Planning geometry H1536/I8960/L28/V151936, query heads12/KV heads2, tied embeddings:
FP32 base 5.75 GiB, three adapters 1.65, gradients .55, Adam 1.10, JL .36,
GMM model/vectors .034; persistent lower bound 9.45 GiB. Million-token hidden
states alone need 5.72 GiB; estimated training/fit planning range 17.08--25.08 GiB.
65,536-token hidden states need .375 GiB; estimated range 10.12--18.12 GiB.
These are analytical estimates, NOT peak bounds/OOM guarantees (EM, RoPE, KV,
allocator workspace are incomplete). Launch recalculates with actual cached model
config. Original batch256 at length8192 would need about 56 GiB just for BF16 KV.

Existing Extended Diagnostics includes Basic metrics, sampled every 1000 forwards,
at most 2 batches/context and 64 valid tokens/batch. Basic and Extended share this
low-frequency cadence; no new schedule or research metric is introduced. Gate,
margin, coactivation, adapter updates and cross-layer summaries use existing
observers; no extra counterfactual forward or repeated training occurs.

All 7 epoch CL states plus sidecars are retained under experiment/checkpoints.
They reconstruct LoRA/JL/GMM for independent probes, NOT exact training resumption:
base weights, optimizer, scheduler, RNG, data cursor, EMA and KV are not included.
Original save_cl_state copies are also enabled, increasing disk use; launcher still
requires at least 60 GiB free. Default run path is
`/mnt/workspace/lsl_runs/continual_1p5b_lsl_a10_seed123`; existing output is refused.
An advisory lock prevents concurrent launches through this entry.

Terminal and process.log stream Stage/Epoch, `[####----] Step x/y`, Loss, GMM/layer
fitting tqdm, evaluation tqdm, checkpoint paths and cumulative CUDA peak.
progress.jsonl retains structured progress and GMM hit/active-owner statistics.
Failures preserve logs, traceback and existing results and return nonzero.
Success additionally requires three complete Stage x Task epoch metric sets,
7 nonempty checkpoints with matching sidecars, the CSV matrix, each stage's
Gate/GMM/Adapter diagnostics, one negative and seven positive GMM fits. Disabled
GMM owners remain governed by original skip-on-fail, but strict completion refuses
to label such a run fully accepted. Large checkpoint tensors are not reloaded by
this file/sidecar check; it is not a complete serialized-content integrity audit.

Windows CPU tests do not certify real A10 memory peaks, evaluation batching or
Linux CUDA/ROCm training. Run the server command explicitly to begin training.
