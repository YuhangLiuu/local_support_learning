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
