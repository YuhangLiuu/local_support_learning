# 论文配置审计

来源：[论文 v1 Appendix D Tables 2/3、Figure 5](https://arxiv.org/html/2610.02126v1)；代码来源见 PROVENANCE.md。

|配置|batch|rank|epochs|K+ / K-|smoothing|
|---|---:|---:|---|---|---:|
|single_igbo|32|128|3|16 / 32|.3|
|single_chemistry|32|128|3|32 / 32|.1|
|single_cybersecurity|16|512|1|32 / 32|.3|
|continual_7b_lsl|32|128|3 / 3 / 1|16 / 32|.1|

共同：Qwen2.5-7B-Instruct；AdamW LR1e-4，wd.1，betas(.9,.95)，clip1；WSD warmup.1/stable.8/decay.1，最低LR1e-6；seq1024，JL256，diagonal GMM，gradient checkpointing；LoRA alpha=2r。持续学习顺序 Igbo→Chemistry→Cyber，benchmark上限300，三个 seeds。这些值落在 reproduction/configs/*.json 的 overrides。

论文未公开种子 IDs；123/456/789 是项目选择。benchmark 限额与 held-out 划分的精确结合未公开：wrapper 保持原划分后取 test 前300，不改变 SecEval500划分。原代码百万 token GMM 预算、正seed0/负seed1、训练 cap/blend 是实现设定，不能全部归为论文明确报告。

论文描述阶段间 merge LoRA；官方代码保留冻结并行 adapters 求和。实数算术可以等价，浮点舍入、计算、存储不同。LoRA对照保留代码行为，标记 LORA_NOT_MERGED，不擅自修复。

Figure4 raw benchmark平均与 Figure7 retention比例须区分。工具以 code_total_retention 标注均值之比，保留各任务比例；不宣称 aggregate 定义已获作者确认。`paper_exact` 的 exactness 字段明确上述边界。

原 load_model 将基础权重转 FP32，forward 使用 BF16 autocast。仅按 BF16 权重估算显存会低估；microbatch 调整属于适配，不是严格数值等价保证。
