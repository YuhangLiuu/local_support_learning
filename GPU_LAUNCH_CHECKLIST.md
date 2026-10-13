# 服务器启动清单

只在服务器下载/执行。本地仅 --dry-run。先阅读问题登记；不要通过重做 inventory 静默更新版本。

```bash
python scripts/prefetch_resources.py --relocate --cache-root /srv/lsl/resources --models --datasets --verify-sha256 --max-gib 40
python scripts/verify_datasets.py --cache-root /srv/lsl/resources --sample-limit 0
python scripts/verify_evaluators.py --cache-root /srv/lsl/resources
python scripts/check_gpu_env.py --config continual_7b_lsl --cache-root /srv/lsl/resources --require-gpu
python scripts/gpu_parity_test.py --config smoke_1p5b --cache-root /srv/lsl/resources --output runs/parity_1p5b
bash scripts/smoke_test.sh 1p5b --cache-root /srv/lsl/resources --parity-receipt runs/parity_1p5b/receipt.json --acknowledge-issues
python scripts/gpu_parity_test.py --config smoke_7b --cache-root /srv/lsl/resources --output runs/parity_7b
bash scripts/smoke_test.sh 7b --cache-root /srv/lsl/resources --parity-receipt runs/parity_7b/receipt.json --acknowledge-issues
python scripts/gpu_parity_test.py --config continual_7b_lsl --cache-root /srv/lsl/resources --output runs/parity_continual
for seed in 123 456 789; do
  bash scripts/run_continual.sh --seed "$seed" --experiment-id "continual_lsl_$seed" --cache-root /srv/lsl/resources --parity-receipt runs/parity_continual/receipt.json --acknowledge-issues
done
```

单任务 run_single_task.sh igbo|chemistry|cybersecurity；对照 run_lora_baseline.sh。分别先生成相应config的parity receipt，不能用rank4 smoke替代rank128/512。receipt核对模型、软件、架构、LoRA/GMM/Gate几何。seed loop仅在全部前检通过后执行。

--microbatch 2 属硬件适配，须测量梯度累积影响。harness自己的生成batch不会自动跟随eval_batch_size缩小，OOM时停止调查。原baseline/关闭/Basic/Extended四模式已有CPU数值验证；CUDA脚本目前比较研究分支Off/Basic/Extended，两者范围不可混淆。

CUDA验证含logits/loss/grad/step/Gate/EMA/RNG、小EM、CL往返、unpadded KV decode；默认零容差。非零容差必须说明原因。正式评测一致性、长序列padding/cache、百万token EM、峰值/吞吐尚须实测，不得提前勾选。

Checkpoint独立评测使用 run_reproduction.py --config <原配置> --eval-checkpoint <实际pt路径> 加新的experiment-id与对应receipt。不能声称恢复optimizer续训。P2调用 diagnostics.probe_runner --help；输入固定probe文件，sidecar stage/epoch完全匹配，experiment-id取原Diagnostics metadata；默认eager FP32须记录与正式FA2/BF16差异。

```bash
python scripts/collect_results.py --runs runs/continual_lsl_123 runs/continual_lsl_456 runs/continual_lsl_789 --output runs/summary_lsl
```

禁止覆盖已有ID；不提交数据、权重、密钥或运行输出。GPU脚本未实际执行，当前结论为代码准备完成待服务器验证，不是正式训练许可已全部通过。
