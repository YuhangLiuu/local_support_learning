# 按需恢复 LSL 环境（默认入口）

在具备可用 PyTorch、GPU 和 FlashAttention 的 Linux 实例上运行：

```bash
cd /mnt/workspace/local_support_learning
git pull --ff-only origin research/lsl-observability
bash scripts/setup_lsl.sh full
```

最小 Smoke 使用 `bash scripts/setup_lsl.sh minimal`，仅允许 `skip_lm_eval=True`。
Full 包含 `lm-eval==0.4.10` 和 NLTK 资源检查，正式评测必须使用 Full。
两种入口均不会下载模型、Benchmark 数据或启动训练。

默认在线 Pip、正常 Pip 缓存；不需要预构建 Wheelhouse。
可选 `--index-url https://your-trusted-index/simple` 或 `LSL_PIP_INDEX_URL`。
非标准系统解释器可用 `LSL_SYSTEM_PYTHON=/actual/base/python bash scripts/setup_lsl.sh full`。
默认环境和报告保存于 `/mnt/workspace/lsl_environment`，模型/数据缓存目录为
`/mnt/workspace/lsl_cache`，可通过 `--root`、`--cache-root` 覆盖。

流程：检测真实 base Python、ABI、实例和 GPU 栈 → 创建/验证当前实例专用
system-site-packages venv → 核对应用版本和实际导入 → 实时显示 Pip 解析计划 →
拒绝 GPU 核心包安装 → 仅安装批准的应用包 → 新 Python 进程严格验收。
关键固定版本：Transformers 4.56.2、Tokenizers 0.22.2、HF Hub 0.36.2、
Pomegranate 1.1.2；应用约束沿用 `reproduction/environment/profiles.json`。
兼容的系统应用包可复用，但验收同时核对其实际导入文件及依赖闭包；
系统 Transformers 5.x 会被 venv 内的 4.56.2 遮蔽，不卸载系统包。

旧环境绑定不匹配或解释器失效时创建新目录，保留旧目录，不删除源码或缓存。
再次执行会重新验收，依赖版本、导入和闭包已正确则不执行 Pip 安装。
安装计划或安装失败立即停止；只有真实核心导入、随机 Tiny LSL、GMM、
BF16 与 FlashAttention 前后向及缓存访问均通过才输出 `LSL ENVIRONMENT PASS`。
JSON 报告写在 `reports/`，失败不会留下当前恢复尝试的 PASS。
验收不是实验 GPU Parity，也不证明 7B 训练可用。

成功后按终端提示 `source .../bin/activate`，再独立执行资源校验、Parity 和训练。
`current-pip-full.json` / `current-pip-minimal.json` 记录已通过环境的解释器和报告。
现有正式启动器会对该 venv 的 `bootstrap.json` 和验收报告执行门槛检查。

无网络且缺少依赖时明确失败，不声称自动恢复成功。完全离线部署仍可显式使用
[旧 Wheelhouse 高级路线](ENVIRONMENT_BOOTSTRAP.md)，它不是默认前置条件。

Windows 只运行 CPU/模拟测试；真实 CUDA/ROCm 安装、系统包复用和 GPU Kernel
验收必须在目标 Linux 实例执行。本次不更改算法、Diagnostics 或实验配置。
