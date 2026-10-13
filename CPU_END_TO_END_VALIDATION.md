# LSL CPU 端到端集成验证

## Material Passport

- 日期：2026-10-13，Asia/Shanghai。
- 任务：验证已有 Diagnostics，不开展算法创新或正式论文实验。
- Verification Status：**VERIFIED，限本文的 CPU FP32/eager、Tiny 模型与合成数据路径**。
- 数据：本地生成 Token，随机初始化模型；没有下载模型权重或真实数据集。
- 证据：[汇总 JSON](validation/cpu_end_to_end_2026_10_13.json)、[环境锁定文件](validation/requirements-cpu-e2e.lock.txt)、本地 JUnit 和逐模式输出。
- 研究解释边界：没有性能显著性检验、算法效果比较或遗忘效果推断；随机 Tiny 的 Loss 仅证明执行正常。

## 1. 结论

**LSL Diagnostics 已通过本轮 CPU 端到端集成验证，可以进入 GPU 正式复现准备阶段。GPU 正式复现验证仍未执行。**

本轮使用项目完整 `Qwen2ForCausalLM` 和实际 Pomegranate，执行两阶段 LoRA 训练、真实正负 GMM EM、门控、CL State 保存/重载、Cache 推理、同步 Basic/Extended 观测和独立固定 Probe。

A=原 Tag、B=Off、C=Basic、D=Extended 的已检查张量均逐位相同，最大绝对/相对误差均为 **0**。没有为四模式对比放宽容差，也没有修改原算法或 Diagnostics 生产代码来通过测试。

验证层级：

| 层级 | 状态 |
|---|---|
| CPU 核心模块 | 已通过，上一轮结果保留 |
| CPU 完整 Tiny 模型 + 真实 EM + 两阶段集成 | **本轮通过** |
| GPU/7B、论文数据和正式评测 | **未执行** |

## 2. 环境与安装

隔离 venv：`D:/FDU/Projects/10-LSL/.venv-lsl-cpu`。它位于仓库目录外，不启用 system-site-packages。

| 软件 | 实际版本 |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.9.0+cpu |
| Transformers | 4.56.2（与仓库 requirements 一致） |
| Pomegranate | 1.1.2 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.1 |
| tokenizers | 0.22.2 |
| loguru | 0.7.3 |
| pytest | 9.1.1 |

完整传递依赖见锁定文件。`pip check`：`No broken requirements found`。没有安装 torchvision、CUDA、FlashAttention 或 GPU 版 PyTorch。

成功安装所用命令：

```powershell
& 'C:/Users/yuhangliu/miniconda3/envs/torchlink/python.exe' -m venv 'D:/FDU/Projects/10-LSL/.venv-lsl-cpu'
& 'D:/FDU/Projects/10-LSL/.venv-lsl-cpu/Scripts/python.exe' -m pip install 'torch==2.9.0+cpu' --index-url https://download.pytorch.org/whl/cpu --disable-pip-version-check
& 'D:/FDU/Projects/10-LSL/.venv-lsl-cpu/Scripts/python.exe' -m pip install 'torch==2.9.0+cpu' 'transformers==4.56.2' 'pomegranate==1.1.2' 'loguru==0.7.3' 'pytest==9.1.1' --disable-pip-version-check
```

原 `torchlink` 复查仍为 torch 2.14.0+cpu、Transformers 5.14.0，仍无 Pomegranate/loguru，没有对它安装、卸载或升级包。

上次导入故障的环境原因：缺少 loguru/Pomegranate；torchvision 0.20.1 的安装元数据要求 **torch==2.5.1**，实际 torch 为 2.14.0+cpu，导入时报 `torchvision::nms` 不存在。Transformers 也不是仓库指定版本。新的隔离环境按仓库版本安装文本模型依赖，不安装非必需的 torchvision，完整导入成功；没有打装饰器补丁、屏蔽视觉模块或更改模型源码。

CPU wheel 来源：[PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)；真实混合模型 API 参考：[Pomegranate GeneralMixtureModel 文档](https://pomegranate.readthedocs.io/en/latest/tutorials/B_Model_Tutorial_2_General_Mixture_Models.html)。版本/API 可用性还通过实际执行确认。

## 3. 模型、数据与真实调用范围

Tiny 配置：vocab=256、hidden=64、intermediate=128、layers=2、attention heads=4、KV heads=2、max positions=128、attention dropout=0.1；CPU float32，eager attention。LoRA rank=4、alpha=8；JL dimension=8、seed=17；smoothing alpha=0.3；正/负 GMM 各 2 个对角分量，random_state=0。

完整流程 seed=314，CPU intra/inter-op threads 均为 1，开启 `torch.use_deterministic_algorithms(True)`。不加载预训练权重，LoRA 的 A/B 使用原始初始化（B=0），所有七类矩阵均使用真实 wrapper。

合成 tokenizer 使用本地 WordLevel `t0…t255`，padding=t0，左 padding；共享前缀 `[1,2]`。每个任务 16 个样本，长度 12–16，batch=4：

- Stage 0：prefix 后的 Token 从 `[10,90)` 采样，数据 RNG seed=1000。
- Stage 1：从 `[50,130)` 采样，seed=1001。
- Negative：从 `[130,230)` 采样，seed=1009。
- 每次 EM 预算=128 个 non-pad tokens，原函数保留完整批次，因此实际收集 **165 tokens/3 batches**。这是原批次停止规则，不是截断到精确 128。

Padding labels=-100，使用真实 HF `model(..., labels=...).loss`。每阶段 3 次 AdamW 更新，lr=1e-3、betas=(0.9,0.95)、weight_decay=0.1；这些是独立 Tiny 测试设置，不改生产训练配置。未运行正式 scheduler、官方数据加载器或评测器。

真实代码路径：

```text
Qwen2Config → Qwen2ForCausalLM → apply_conditional_lora_wrapper → add_lora
  → fit_cl_memory(neg, Stage 0 only)
    → Qwen2Model(full_dataset_input)
      → 真实 decoder 逐层/逐批 Forward → fit_layer_gmms
        → GMMClassifierGPU.fit_neg → Pomegranate EM
  → model(labels=...) → 原 HF CE → backward → AdamW
  → fit_cl_memory(pos) → fit_layer_gmms → GMMClassifierGPU.fit_pos → 真实 EM
  → eval → is_in_gmm / predict → Temporal Smoothing → Σ conditional LoRA updates
  → save_cl_state → 相同随机 frozen base → load_cl_state → 对照输出与 Gate
  → 从 Stage 0 重载模型进入 Stage 1，新增 Adapter、冻结历史 Adapter
```

测试自有部分仅包括小型配置、合成 Token/tokenizer、训练驱动、只读捕获和断言。**本轮没有使用 AST 抽取、Mock GMM、手动拟合参数容器或替代 Attention。** 原有核心测试中的降级夹具仍保留，作为补充回归。

Baseline 每个进程通过 `git show baseline-lsl-original:<file>` 获取 modeling/configuration/cl_memory/ddp 完整文件，原样导入。A/B/C/D 各自新进程，避免类补丁、全局 Pomegranate cache-floor patch 和 RNG 相互污染。CPU/离线环境变量在子进程中设置。

## 4. 测试结果

最终命令（仓库根目录）：

```powershell
& 'D:/FDU/Projects/10-LSL/.venv-lsl-cpu/Scripts/python.exe' -X utf8 -m pytest tests -q --basetemp=tmp/cpu-e2e-release --junitxml=tmp/cpu-e2e-release.xml --tb=short
```

**73 passed，0 skipped，0 failed，0 errors，54.75 秒。** 两个 warning 是原源码 docstring 的 invalid escape sequence。

- 原有 65 项全部通过，原先缺 Pomegranate 而跳过的真实 EM 测试已实际执行。
- 新增 8 项完整模型集成断言：四模式对照 3 项、EM/冻结/roundtrip/调用计数 1 项、Basic/Extended 语义 2 项、跨层/共激活/Cosine/Probe 1 项、真实 Probe Drift oracle 1 项。
- `cpu_integration`、`real_em` marker 区分新增测试；GPU 没有测试通过记录。

本地完整证据：

- `tmp/cpu-e2e-release.xml`：JUnit。
- `tmp/cpu-e2e-release/full-qwen-real-em0/{baseline,off,basic,extended}/process.log`：真实进程输出。
- 同目录 `summary.json`、`snapshots.pt`、`parity.json`（B/C/D）、`cl_stage{0,1}.pt`。
- C/D 的 `diagnostics/`、独立 `probe/`、`probe_history.npz`、`probe_snapshots.pt`。
- 小型持久汇总和源文件 SHA256：[validation/cpu_end_to_end_2026_10_13.json](validation/cpu_end_to_end_2026_10_13.json)。大型合成张量/Checkpoint 按原 gitignore 不提交。

### 两阶段结果（四模式完全相同）

| Stage | 三步训练 CE | 固定 Probe eval CE | 真实 EM | Active Owners |
|---|---|---:|---|---:|
| 0 | 5.58492136 / 5.60468674 / 5.54359865 | 5.56252670 | 8 negative + 8 positive | 8/8 |
| 1 | 5.60724068 / 5.57157087 / 5.56304169 | 5.56043100 | 8 positive，沿用 negative | 8/8 |

各步是不同合成 batch，不能把这些数字当成效果曲线。每阶段只有 28 个 trainable tensors（14 matrices × A/B），基础模型和历史 LoRA 均不变，当前 Adapter 有实际更新；Stage 1 没有重拟合 negative。

GMM 参数全部有限，没有依赖 skip-on-fail 掩盖拟合失败；测试配置明确设为 fail-fast。四模式 Stage 1 scoring covariance 的最小值均为 0.0006384688895。原 GMM covariance-floor 和失败策略没有修改。

## 5. 四模式数值一致性

最大相对误差按逐元素 `|a-b| / max(|a|,|b|,1e-12)` 计算。所有比较先检查结构/shape/dtype，再检查 `torch.equal`；RNG 的 NumPy/Python 非张量状态也比较。

| 项目 | A vs B | A vs C | A vs D | Max Abs / Max Rel |
|---|---|---|---|---|
| 完整 logits、真实 HF Loss | bitwise | bitwise | bitwise | 0 / 0 |
| 每步 LoRA A/B Gradient，包括 None | bitwise | bitwise | bitwise | 0 / 0 |
| 每步 AdamW 后所有模型参数 | bitwise | bitwise | bitwise | 0 / 0 |
| 真实 GMM mean/cov/prior、vectorized 参数 | bitwise | bitwise | bitwise | 0 / 0 |
| Raw Gate、Smoothed Gate | bitwise | bitwise | bitwise | 0 / 0 |
| EMA、prefill/两步 decode 的真实 HF KV Cache | bitwise | bitwise | bitwise | 0 / 0 |
| Torch / NumPy / Python RNG | exact | exact | exact | 0 / 0 |
| 两次 CL State round-trip 输出、loss、gate | bitwise | bitwise | bitwise | 0 / 0 |

原向量化 likelihood 与 **Pomegranate 自身** `log_probability` 是另一项 API 交叉检查，不是四模式差异：在固定测试网格上，Max Abs=0.000244140625，Max Rel=5.2224208957e-7；assertion 为 atol=1e-3、rtol=1e-5。两条实现的浮点运算顺序不同，部分外插点 log-likelihood 绝对值大。这个容差只用于 API 交叉检查，没有用于放宽任何 A/B/C/D 或 Checkpoint 比较。

### 没有额外 inline 模型调用或拟合

每模式正常两阶段/round-trip/Cache 验证计数相同：

| 调用 | A/B/C/D 每个模式 |
|---|---:|
| 完整 model Forward | 13 |
| Backbone Forward（含拟合 sweep） | 16 |
| 正 GMM fit | 16 |
| 负 GMM fit | 8 |
| GMM predict（含显式重载评分核对） | 300 |

Basic/Extended 之后才运行独立 P2 Probe：各额外 8 次 model/backbone Forward、98 次 predict，**没有 EM fit**；单独记入 `independent_probe_counts`，没有混入 inline 计数。关闭诊断没有挂载 Collector 或生成诊断目录。

## 6. 真实模型 Diagnostics 正确性

全部记录保留 Stage/Epoch/Dataset/Layer/Adapter/Owner；正常 eval 两行 non-pad tokens=25，排除两行各两个 prefix 后 denominator=**21**。

第一层 q_proj 的同一次 Forward 独立对照：

| Stage / Adapter | Raw hits | Smoothed hits | Mean Margin | Mean Applied Update L2 |
|---|---:|---:|---:|---:|
| 0 / 0 | 17/21 | 15/21 | 0.74998075 | 0.02715502 |
| 1 / 0 | 17/21 | 15/21 | 0.74998075 | 0.02715502 |
| 1 / 1 | 11/21 | 9/21 | -0.12430532 | 0.01693231 |

验证方式与状态：

- Gate Activation：与真实 captured smoothed masks 的计数相同，padding/prefix 排除规则按固定输入独立确认。
- Positive/Negative Likelihood、Margin：从真实 fitted 参数和相同投影独立评分，与 Collector mean margin 对照；单独对照 Pomegranate API，所有记录有限。
- Coactivation/all-closed：从真实所有 matrix × adapter masks 重建 token active count，逐矩阵核对分子。
- Update L2：用捕获的 q input 和真实 A/B/mask 重算，与 Collector 对照；多 Adapter 更新求和与实际 q output 逐位相同。
- Update Cosine：真实 Stage 1 两 Adapter applied updates 的 NumPy dot/norm oracle 与记录 sum/count 对照，并确认存在非零有效分母。
- Cross-Layer：对四类 owners 的相邻层真实 masks，按同一 token positions 核对每个 Adapter disagreement 分子/分母。
- Training–Inference：当前 Adapter 的 forced_on_rate=1，score_observed_rate=0；不把强制开启误当成已经计算的 GMM score。历史 Adapter 使用真实 gate。
- Owner 去重：每层七个 matrix、四个 owner；两层 Stage 0 为 14 matrix / 8 owner，Stage 1 为 28 / 16 adapter activation records，两个 scope 不混加。
- JSONL：重读、有效 sample_count、按计数聚合验证；多 rank 边界继续由原有模拟记录测试覆盖，未启动真实 DDP。
- 独立固定 Probe：在两个已保存的真实模型上运行 full/base/all-gated-no-smoothing/current-forced；参数、EMA、RNG 和已有梯度保持，Probe 结束不改变模型状态。
- Probe Drift：从保存的实际 bounded snapshots 独立计算 representation norm、Gate Flip、margin/positive/negative score shifts，核对 first/previous 两种记录和 Stage/Epoch；历史仅匹配 Adapter 0，Stage 1 新 Adapter 不伪造历史参考。NPZ 保存/加载后继续比较成功。

这些指标仅描述随机 Tiny 的代码执行，不是论文复现结果。未验证 7B 诊断成本、GPU 数值/内存行为或正式任务指标。

## 7. Checkpoint 和恢复范围

Stage 0 保存原始 `save_cl_state`，新建相同随机 frozen base，`load_cl_state`，核对 logits/loss/raw gate/smoothed gate；随后从**重载模型**进入 Stage 1，训练后再次保存/重载核对。真实 Pomegranate priors/means/covs 和重新 vectorize 后的 scores 均逐位相同。

测试配置在 summary/metadata JSON 中保存并重读，Probe history 使用实际 NPZ load。冻结的 base 不在 CL State 中，测试通过相同配置和保存的初始 base state 重建；没有声称 CL 文件独立包含整个基础模型。

**未测试完整训练断点续训**：Optimizer、Scheduler、RNG、数据位置、EMA 和 KV 不由原 CL State 保存。Stage 1 是新阶段、新 Optimizer；Checkpoint 对照使用新的完整 prefill，不能据此推断途中 decode 或中途 optimizer resume 能恢复。

## 8. 问题与未验证风险

### 环境问题

已通过隔离安装解决 loguru/Pomegranate 缺失和 Transformers/torchvision 兼容问题。原 `torchlink` 不变。

原 `fit_cl_memory` 硬编码 CUDA autocast；CPU PyTorch 会警告并自动禁用，实际执行仍为 CPU，`torch.cuda.empty_cache()` 在此为 no-op。没有 monkeypatch 或修改这段原逻辑以获得通过。

### 原 LSL 问题

此前首次 T=1 且无 EMA prefill 的 KeyError 继续记录，未修复。完整模型 incremental decode 将 full attention mask 传给 token gate 时存在 padding 对齐风险，本轮 Cache 用无 padding prefill 避开该上游边界；不把这部分计为已经覆盖的 padded generation。Padding/prefix 的普通 Forward 与统计分母已覆盖。

### Diagnostics 问题

本轮没有发现需要修复生产 Diagnostics 的新错误；没有新增科研指标或更改既有算法。上一轮三个修复及回归保留。

### 正式 GPU 实验前仍必须补验

1. 目标 CUDA/PyTorch/Transformers/FlashAttention 组合和生产 dtype/autocast 下四模式一致性；先要求 bitwise，任何非零容差要有实测内核依据。
2. 真实 7B checkpoint/tokenizer revisions、官方数据顺序、Loss/Optimizer/Scheduler 的生产驱动和全部原评测器。当前测试没有执行 `train_continual.main()` 或下载论文数据。
3. 长序列、大 GMM 组件/JL、beam/batched/padded generation、梯度检查点与真实多 rank DDP。
4. 长时间显存/主存、CPU transfer、I/O、吞吐开销；Basic 优先，Extended 的预算单独评估。
5. 如需断点续训，另验证完整训练 checkpoint；不能使用本轮 CL State 验证替代。

## 9. Git 状态和提交

- 分支：`research/lsl-observability`，本地使用，没有推送或新增 Remote。
- 原始基线：`baseline-lsl-original` → `fb9d2612fe3b51514f7703f250c3c218883d29f9`，main 未改变。
- 新 Annotated Tag：`lsl-observability-cpu-validated` → `badb29c`，明确只表示上一轮 CPU 核心测试通过，未冒用为 GPU 或端到端 Tag。
- `b1529cb`：真实 Tiny worker、EM/CL State round-trip、环境锁定。
- `bafdafc`：完整四模式/诊断/固定 Probe 集成测试及 pytest markers。
- 本报告与持久汇总在随后单独文档 Commit；最终 HEAD 用 `git log -1 --oneline` 查询，交付前工作树干净。

执行测试时新测试文件尚未提交，原始诊断记录保留 `badb29c` + dirty=true，不重写 provenance；随后提交的文件就是执行过的内容，汇总 JSON 用 SHA256 固定源字节和对应 implementation commit。

本轮 `modeling_qwen2.py`、`cl_memory.py`、`configuration_qwen2.py`、`train_continual.py`、配置及 `diagnostics/` 相对 `badb29c` 全部无修改。
