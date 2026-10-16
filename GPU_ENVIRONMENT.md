> 默认环境恢复已改为 [setup_lsl.sh 按需 Pip 路线](SETUP_LSL.md)：`bash scripts/setup_lsl.sh minimal` 或 `bash scripts/setup_lsl.sh full`。下文 Wheelhouse/旧安装步骤仅供显式选择的高级部署参考，不是默认前置条件。

# GPU 服务器环境

更换云实例的环境恢复使用 [ENVIRONMENT_BOOTSTRAP.md](ENVIRONMENT_BOOTSTRAP.md)：Persistent Wheelhouse、Minimal/Full Profile、当前实例专用venv及严格验收。下文为早期候选安装路线，不能直接复用持久化旧venv。

AMD现有环境部署优先阅读 [AMD_ROCM_DEPLOYMENT.md](AMD_ROCM_DEPLOYMENT.md)。下文PyTorch2.9/CUDA安装是原候选NVIDIA路线，不能用于覆盖已可用的ROCm2.12环境；ROCm不要求NVIDIA工具或CUDA runtime字段非空。

目标 Linux / Python3.11。setup_gpu_env.sh 创建新 venv，拒绝覆盖。PyTorch2.9.0 CUDA wheel 在 cu126/cu128/cu130 中按驱动、GPU架构、nvcc选择；FlashAttention2.8.3 是未实测候选。requirements-gpu.txt 固定 Transformers4.56.2、Pomegranate1.1.2、lm_eval0.4.10 等；CPU验证不等于 CUDA 环境认证。

原 Dockerfile NVIDIA25.10保持原样；新增 Dockerfile.gpu 要求显式 BASE_IMAGE。不假定所有服务器支持CUDA13。参考 [PyTorch官方版本](https://pytorch.org/get-started/previous-versions/) 和 [FlashAttention官方要求](https://github.com/Dao-AILab/flash-attention)。

```bash
nvidia-smi
nvcc --version
bash scripts/setup_gpu_env.sh --venv .venv-gpu --torch-index https://download.pytorch.org/whl/cu128 --dry-run
# cu128仅为示例；服务器核对后去掉dry-run安装并激活。
source .venv-gpu/bin/activate
python scripts/check_gpu_env.py --config continual_7b_lsl --require-gpu
```

7B基础7,615,616,512参数，FP32约28.37GiB；rank128三阶段adapter3.61GiB、梯度1.20GiB、Adam2.41GiB，加JL/GMM持久下界约36.42GiB。默认micro32粗估训练/拟合58–82.5GiB，非实测峰值或OOM保证。百万token隐状态13.35GiB；24GiB卡不能容纳原FP32基础权重。优先80GiB级设备再实测；micro2保留global32但拟合峰值仍需测量。Cyber rank512单独估计和smoke。

rank128三阶段CL State单份估计4.71GiB；所有epoch、三种子、两种方法及parity快照应按数百GiB规划。launcher最低60GiB空闲不替代总容量预算。

check_nccl.py只验证通信。原rank-local GMM和sampler行为尚未验证多卡算法等价，正式准备入口目前拒绝nproc>1。未实现ZeRO/FSDP，不能据此承诺小显存多卡可运行。
