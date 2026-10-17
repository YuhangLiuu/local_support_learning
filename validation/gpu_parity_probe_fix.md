# GPU Parity Probe 修复记录

Parity worker 原先只构造 32 条文本。`fit_cl_memory` 在小预算下重复模式很多，且更高预算无法从这样的小池稳定取得独立样本。服务器失败样本的 2048 x 256 激活只有 12 个 unique rows（duplicate ratio 0.9941），因此这次先修测试输入并加闸门，不改 GMM EM。

`scripts/gpu_parity_test.py` 现在使用固定 Seed 的 1024+ 条、变长、带样本 ID/变体和确定性词序的文本池。Off/Basic/Extended 每个独立 worker 以同一 Seed 重建同一池和 batch 顺序；没有噪声注入，也没有真实任务数据依赖。`fit_layer_gmms` 在 Parity 路径被只读包装，使用实际 `_pending_x`（JL 后、Padding 已由原逻辑排除）检查形状、有效行数、unique rows、重复比例、有限性、近零方差维度和 K。`unique < K` 时输出 `GPU PARITY TEST DATA INVALID`，worker 非零退出，父进程不会写 `passed=true` Receipt。

独立诊断命令（服务器存在失败张量时）：

```bash
python scripts/reproduce_gmm_failure.py /mnt/workspace/lsl_failed_gmm_input.pt --components 32 --seeds 0 1 2
```

它仅调用原 Pomegranate 初始化和 `fit`，报告各 Seed 是否稳定失败，并用固定随机高斯控制输入对照；不会替换生产初始化或修改模型。Windows 没有该服务器张量时保留为待验证。

服务器执行新的 Rank128/JL256/K16/32 Parity（不覆盖旧失败目录）：

```bash
cd /mnt/workspace/local_support_learning
git pull --ff-only origin research/lsl-observability
test "$(git rev-parse --short HEAD)" = "$(git rev-parse --short origin/research/lsl-observability)"
export LSL_CACHE_ROOT=/mnt/workspace/local_support_learning/.cache/lsl
export PARITY_OUT=/mnt/workspace/lsl_parity/continual_1p5b_lsl_a10_seed123_probe_fixed_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$PARITY_OUT"
python scripts/check_gpu_env.py --config continual_1p5b_lsl_a10 --cache-root "$LSL_CACHE_ROOT" --require-gpu --output "$PARITY_OUT/gpu_env.json"
python scripts/prefetch_resources.py --config continual_1p5b_lsl_a10 --cache-root "$LSL_CACHE_ROOT" --models --datasets --check-only --verify-sha256
python scripts/gpu_parity_test.py --config continual_1p5b_lsl_a10 --cache-root "$LSL_CACHE_ROOT" --output "$PARITY_OUT" --fit-tokens 65536
python - "$PARITY_OUT/receipt.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); r=json.loads(p.read_text())
expected={'activate_cl_method':True,'lora_r':128,'lora_alpha':256,'jl_k':256,'gmm_n_pos_comps':16,'gmm_n_neg_comps':32,'smoothing_alpha':0.1,'use_grad_checkpointing':True}
assert r.get('passed') is True, r
assert r.get('errors',{}).get('max_abs') == 0 and r.get('errors',{}).get('max_rel') == 0, r
assert r.get('geometry') == expected, r.get('geometry')
assert r.get('model_revision') == '989aa7980e4cf806f80c7fef2b1adb7bc71aa306', r.get('model_revision')
print('GPU PARITY PASS')
PY
```

若 Probe diversity、Pomegranate、worker、Receipt 任何一步失败，命令以非零退出并保留每个 mode 的日志；不会启动七个 Epoch 的正式训练。Parity 通过后再设置 `LSL_PARITY_RECEIPT=$PARITY_OUT/receipt.json` 执行 `scripts/run_continual_1p5b_seed123.sh`。
