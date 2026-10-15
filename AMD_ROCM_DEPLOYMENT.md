# Windows 开发与 AMD ROCm 服务器部署

本轮只修改路径、资源预检、下载入口、环境检测和错误处理；原模型、LoRA、GMM、Gate、平滑、loss/optimizer/scheduler、实验配置和 Diagnostics 指标定义保持不变。baseline/main/CPU tag 均不移动。本地没有下载权重或数据，没有安装环境，没有执行真实 ROCm 计算。

## 路径与资源约定

缓存优先级：CLI `--cache-root` → `LSL_CACHE_ROOT` → manifest.cache_root → 项目 `.cache/lsl`。相对路径相对仓库根目录；旧反斜杠/混合分隔符受控规范化。旧机器的绝对路径只有提供当前机器缓存覆盖后才可忽略；Linux 不接受 Windows 盘符/UNC 根目录，不将其伪装成 Linux 相对目录。

Git 中仍为 schema1，local_path 已统一 POSIX 相对路径，全部 repo/revision/files 保持不变。运行路径由 repo/kind 和缓存根推导；不信任旧 status。拒绝父目录穿越、非法 repo、文件绝对路径、盘符/ADS、资源身份不匹配及 symlink/junction 逃逸。

预检检查需要的 config/tokenizer/全部权重分片或数据文件及其大小。revision 必须有匹配 `.lsl-resource.json`，或旧缓存逐文件 Hub download metadata；缺少 metadata 时，只有每个必需文件都匹配 pinned SHA256 才能接受。单独目录存在、旧 status=downloaded、手写目录名均不足以通过。`--verify-sha256` 校验有公开 SHA 的文件；没有 SHA 的文件使用固定 revision 与大小，不能称所有文件均有密码学校验。

复用现有 prefetch：`--config` 选择模型，只有加 `--datasets` 才准备该任务、负数据与已启用的评测数据。最小smoke_1p5b已设置skip_lm_eval=true、run_cl_eval=true；--models --datasets 合计一个模型、两个数据集（Igbo与Negatives），仍训练LSL、拟合GMM、评测Igbo并采集Diagnostics，不要求通用Benchmark预热报告。正式配置保持完整评测。`--dry-run` 不访问网络；`--check-only` 只检查；失败可重跑相同命令，成功缓存跳过。`--relocate` 在内存迁移旧路径，机器状态只写缓存侧车，不再改 tracked lock；部署禁止 inventory repin。

## ROCm 环境

用户报告：Ubuntu22.04、ROCm7.2.53211、约192GB GPU、Python3.12、PyTorch2.12.0 ROCm、Transformers4.56.2、FlashAttention2.8.3、Pomegranate1.1.2、lm_eval0.4.10；环境 `/mnt/workspace/lsl-venv`。BF16与FA前后向的服务器测试由用户提供，本地未重验。

按 [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html)，HIP复用torch.cuda接口；保留 `.to('cuda')`、autocast('cuda')及相关原算法调用。用torch.version.hip识别ROCm，记录HIP版本/gcnArchName，不把cuda runtime=None或NVIDIA capability阈值当ROCm失败。rocm-smi/rocminfo/hipcc只是可选信息，不要求nvidia-smi或nvcc。NCCL字符串在ROCm对应RCCL接口，多卡算法仍未验证，正式入口仍只接受单GPU。

不运行安装脚本覆盖现有环境。setup_gpu_env.sh --backend rocm 拒绝安装；--backend rocm --check-existing 只检查当前Python。CUDA安装路线也拒绝已有目标目录、活动venv/Conda或检测到的ROCm PyTorch。不注入CUDA专有的CUBLAS/allocator设置到ROCm；已有用户环境变量保留，不偷偷调整GPU计算策略。

用户报告系统vLLM与Transformers依赖冲突，但没有提供vLLM精确版本/约束。check_gpu_env.py只读取当前环境distribution metadata，记录torch/Transformers约束冲突，不import/修复vLLM。LSL不依赖vLLM，不为它升级Transformers；系统环境与lsl-venv检查结果可能不同。

## 可直接执行的服务器命令

先确认GPU服务器有足够空闲磁盘。Parity保存大量真实数值与CL快照；--output以.json结尾时，receipt是该文件，快照/日志在同名.artifacts目录。已有输出不会覆盖，重跑须选择新名字。旧脚本失败后可能留下名为lsl_gpu_parity.json的目录，因此下面采用新的portable文件名。

```bash
set -euo pipefail
cd /mnt/workspace/local_support_learning
git switch research/lsl-observability
git pull --ff-only origin research/lsl-observability
source /mnt/workspace/lsl-venv/bin/activate
export LSL_CACHE_ROOT=/mnt/workspace/lsl_cache

# 只计划并下载1.5B，固定原锁定revision。
python scripts/prefetch_resources.py --config reproduction/configs/smoke_1p5b.json --models --relocate --dry-run
python scripts/prefetch_resources.py --config reproduction/configs/smoke_1p5b.json --models --relocate --verify-sha256
python scripts/prefetch_resources.py --config reproduction/configs/smoke_1p5b.json --models --check-only --verify-sha256
python scripts/check_gpu_env.py --config reproduction/configs/smoke_1p5b.json --require-gpu --output /mnt/workspace/lsl_gpu_environment.json

python scripts/gpu_parity_test.py --config reproduction/configs/smoke_1p5b.json --cache-root /mnt/workspace/lsl_cache --output /mnt/workspace/lsl_gpu_parity_portable.json

# 最小Smoke只需要Igbo任务及负样本；显式请求，不下载7B或通用Benchmark。
python scripts/prefetch_resources.py --config reproduction/configs/smoke_1p5b.json --datasets --verify-sha256
python scripts/prefetch_resources.py --config reproduction/configs/smoke_1p5b.json --models --datasets --check-only
bash scripts/smoke_test.sh 1p5b --cache-root /mnt/workspace/lsl_cache --parity-receipt /mnt/workspace/lsl_gpu_parity_portable.json --experiment-id smoke_1p5b_rocm --acknowledge-issues
```

Parity缓存缺失时在Transformers之前报repo/revision/路径/预取建议。Worker失败时报对应off/basic/extended日志完整位置及末尾异常，不产生通过receipt。原Off/Basic/Extended比较规则和默认零容差未修改；原baseline对比继续由CPU回归覆盖，不宣称GPU增加了baseline worker。

仍待服务器：PyTorch2.12/Python3.12真实加载、两阶段EM/FA2/BF16/梯度检查点/KV/保存加载数值一致性、任务harness预热与Smoke、峰值显存/吞吐、正式7B及百万token拟合。用户先前BF16/FA单项通过不替代这些项目。非确定性内核错误应定位，不自动放宽容差或替换算法。

## 本地验证记录（2026-10-15）

完整CPU回归144 passed、0 failed、0 skipped；原101项保留，新增43项。三个warning来自既有源文件字符串中的无效转义提示。额外通过pip check、compileall、git diff --check；所有锁定revision/files及原核心/Diagnostics/实验配置相对44b2119不变。本地模型缓存不存在，真实ROCm计算未执行。

路径测试执行真实临时文件操作；Windows无symlink权限时使用junction验证目录逃逸。下载重试通过mock Hub边界、真实文件与SHA验证，不访问Hub。Worker失败测试实际执行一个抛错的Python子进程并检查日志根异常。NVIDIA/ROCm为模拟硬件metadata，不能替代GPU数值验证。
