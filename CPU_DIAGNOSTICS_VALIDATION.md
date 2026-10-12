# LSL Diagnostics 纯 CPU 正确性验证报告

日期：2026-10-12（Asia/Shanghai）。本轮仅验证和修复已有 Diagnostics，没有新增科研指标、安装依赖、下载模型/数据或运行正式 LSL 实验。

## 结论与准入范围

**已通过本报告定义的 CPU 核心计算与统计正确性检查；尚不能据此批准真实 GPU/7B 正式复现实验直接开启 Diagnostics。**

在已执行的 CPU 测试路径上，没有发现 Diagnostics 改变 LSL 的数值行为。A/B/C/D 四模式逐位相同，所有一致性比较均使用零容差。但完整生产模型、真实 Pomegranate EM、生产评测器和 GPU 精度/内核尚未验证。下一步应在目标环境完成下文必检项，通过后再同步开启 Basic；Extended 还需确认开销预算。

“通过”指下列明确输入和断言的通过，不是对全部输入、全部模型版本或硬件的证明。

## 1. 基线、环境和证据

- 基线读取自本地不可变 Tag `baseline-lsl-original`，Commit `fb9d2612fe3b51514f7703f250c3c218883d29f9`。
- 研究分支：`research/lsl-observability`；本轮测试实现 Commit `8c8592d`。
- 本轮没有修改 `modeling_qwen2.py`、`cl_memory.py`、`train_continual.py` 或原训练配置；原先的观测插点保持不变。
- Python 3.11.15；PyTorch 2.14.0+cpu；NumPy 2.4.6；pytest 9.1.1；Transformers 5.14.0；torchvision 0.20.1；CUDA 不可用。
- 缺少 `loguru` 和 Pomegranate；仓库要求 Transformers **4.56.2**，现有版本不同。

最终执行命令（PowerShell，仓库根目录）：

```powershell
& 'C:/Users/yuhangliu/miniconda3/envs/torchlink/python.exe' -X utf8 -m pytest tests -q --basetemp=tmp/cpu-audit-final --junitxml=tmp/cpu-audit-final.xml --tb=short
```

结果：**65 collected，64 passed，1 skipped，0 failed，0 errors，26.84 秒**。两个 warning 均为原源码 docstring 的 `invalid escape sequence '\ '`，没有为消除警告修改原代码。

原始 JUnit 证据：[tmp/cpu-audit-final.xml](tmp/cpu-audit-final.xml)（本地忽略文件）。汇总证据：[validation/cpu_2026_10_12.json](validation/cpu_2026_10_12.json)。

| 测试文件 | 通过 | 跳过 | 作用 |
|---|---:|---:|---|
| `tests/test_baseline_contract.py` | 2 | 0 | 原训练参数和流程结构约束，属于补充证据 |
| `tests/test_torch_consistency.py` | 11 | 1 | 既有真实 Linear/评分器计算；EM 因缺依赖跳过 |
| `tests/test_cpu_integration.py` | 18 | 0 | 四模式、多阶段、真实 decoder 核心、Cache、异常、Probe、checkpoint 重算 |
| `tests/test_cpu_metric_oracles.py` | 15 | 0 | 已知答案、实际 Collector、计数/存储/聚合数值 |
| `tests/test_diagnostics.py` | 8 | 0 | 原基础统计、Schema、固定 Probe、窗口、NPZ |
| `tests/test_storage_regressions.py` | 5 | 0 | 实验隔离和聚合上下文修复 |
| `tests/test_state_restoration_regressions.py` | 2 | 0 | 共享列表和采样预算异常恢复 |
| `tests/test_drift_history_regressions.py` | 3 | 0 | 空历史、替换加载、失败原子性 |

## 2. Tiny 集成测试的真实范围

直接导入完整仓库模型首先因缺少 `loguru` 失败；进一步检查 Transformers 模型依赖时，torchvision 注册出现 `RuntimeError: operator torchvision::nms does not exist`。没有通过安装/卸载包或修改生产源码规避这些问题。

因此采用降级夹具 `tests/cpu_lsl_fixture.py`：从 Git Tag 和当前源码分别加载定义，再**实际执行函数体**。AST 只用于避开不可用的导入和抽取源码，不作为数值等价的证明。

执行的真实代码包括：

- `ConditionalLoRALinear` 初始化、增量 Adapter、冻结、Forward、GMM membership 和 Temporal Smoothing。
- `_diag_gmm_logprob`、`GMMClassifierGPU.predict`、`is_in_gmm`、参数 vectorization/load/state serialization。
- 官方 Qwen `Qwen2Attention`、eager attention、RoPE 应用、`Qwen2MLP`、`Qwen2RMSNorm`、`Qwen2DecoderLayer`。
- `apply_conditional_lora_wrapper`、JL 构造、Q/K/V 与 Gate/Up 共享、`add_lora`、`save_cl_state/load_cl_state`。
- 研究分支实际 `DiagnosticsCollector`、`ProbeRunner` 和所有既有统计模块。

降级边界：外层为测试自有 embedding/head、位置编码及 causal mask 组装；HF 文档/弃用装饰器用恒等装饰器；GradientCheckpointingLayer 的宿主用 `nn.Module`。GMM 的 Pomegranate 对象使用**只存参数、没有 EM 或评分功能**的容器，使真实参数序列化代码可执行。门控、LoRA、attention、MLP、RMSNorm 和评分公式没有重写或 Mock。

模型大小：hidden=64，layers=2，heads=4，KV heads=2，intermediate=128，LoRA rank=4，JL=8，vocab=32，sequence=16。CPU FP32、eager attention；非零 B，attention dropout=0.15，使梯度和 RNG 检查不因零更新而退化。

这属于真实核心计算集成测试，**不是完整 `Qwen2ForCausalLM`、生产模型加载或正式评测验证**。

## 3. A/B/C/D 四模式一致性

模式：A=Tag 原始实现；B=研究分支关闭诊断；C=Basic；D=Extended。

种子为 0、42、2026，分别测试 train/eval，共 6 组四模式比较；每个模型有 3 个 Adapter 阶段。每组用相同种子重建模型，先确认全部注册参数一致，再恢复相同 RNG 执行一次合成训练步。测试使用 AdamW(lr=0.001)，这是一次性数值检验，不是更改正式实验设置。

输入固定为两行 16 tokens：

```text
[0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
[0,1,2,15,16,17,18,19,20,21,22,23,24,25,26,27]
```

0 为 padding；匹配到的共享前缀为 `[1,2]`，使用官方 `drop_shared_prefix_mask`；诊断预算 11 个 valid tokens。测试 loss 是明确的 shifted cross entropy，不声称执行了完整 HF/生产 loss 路径。

| 对比项 | A=B=C=D 的结果 | 容差 |
|---|---|---|
| Forward logits / loss | 逐位一致，最大绝对差 0 | 0 |
| LoRA/所有注册参数梯度，包括 None 状态 | 逐位一致，且存在非零梯度 | 0 |
| 一次 AdamW 更新后参数 | 逐位一致 | 0 |
| 实际 raw gate、smoothed gate | 逐位一致 | 0 |
| EMA、prefix mask、原始 hit counters | 逐位一致 | 0 |
| 原算法 accumulation `_pending_x` | 张量及形状一致 | 0 |
| 正负 GMM scoring 参数 | 逐位一致，没有被观测改写 | 0 |
| Torch、NumPy、Python RNG | 状态一致 | 精确比较 |

另外，B/C/D 分别与 A 比较真实 `DynamicCache` 的 prefill、连续两个单 token decode：logits、Gate、EMA、每层 Key/Value 和 Cache 长度均逐位一致。Forward 人为在相同位置失败时，四模式留下的原算法参数、EMA、accumulation/prefix 状态和 RNG 仍一致；诊断临时绑定清理完毕。

没有因为失败而放宽一致性容差。下述统计 oracle 的容差是 NumPy FP64 参考和 FP32 模型统计之间的舍入差，不能用于掩盖模型 A/B/C/D 差异。

## 4. 指标逐项已知答案

### Gate、GMM、Adapter 和训练/推理语义

真实小线性网络的输入为 `x=[(0,1),(.5,1),(1,1),(2,1)]`，base 为单位矩阵，两个 Adapter 的 A 均为单位矩阵，B 分别为 diag(3,4)、diag(4,3)，scaling=1。

正高斯 N((0,0),I)，负高斯 N((2,0),I)。独立概率公式得到：

```text
log p+(x) = -log(2π) - ||x||²/2
log p-(x) = -log(2π) - ||x-(2,0)||²/2
margin = [2,1,0,-2]
raw gate = [1,1,0,0]   # 严格 >0，边界 0 关闭
```

smoothing α=.3，从零开始得到 s=[.3,.51,.357,.2499]，applied gate=[0,1,0,0]。

| 情况 | 历史 Adapter | 当前 Adapter | 共激活 | 全关 |
|---|---|---|---:|---:|
| train | raw gate，2/4 激活 | 强制开启，4/4 | 2/4 | 0/4 |
| eval | smoothed gate，1/4 | smoothed gate，1/4 | 1/4 | 3/4 |

当前 Adapter 在 train 中没有执行 GMM 评分，所以 `score_observed_rate=0`，不伪造 margin/raw gate。train 无 EMA 更新，eval raw/applied disagreement=1/4。

实际 Collector 的输出与独立计算 `base + Σ mask_i*(x A_iᵀ B_iᵀ)` 一致。update L2、update/base 相对范数和两个实际 applied update 的 cosine 均从这些固定矩阵另算；零范数 token 不进入 cosine 分母。均值误差容限 3e-7，cosine 1e-7，margin/update correlation 的 sxy 容限 3e-6。实际张量输出没有容差差异。

额外 Gate 矩阵 oracle 覆盖非连续有效 token：activation=3/6、raw activation=3/6、raw/applied disagreement=4/6、switch=2/3、coactivation=1/6、all-closed=1/6；padding 空行不进入 sequence 分母，跨无效位置不算相邻切换。

GMM 双分量独立概率空间 oracle：prior=(.25,.75)，mean=(-1,1)，variance=(.5,2)，x=(-1,0,1,2)。log(sum prior*density) 和归一化 density 的 responsibility 与 FP64 实现误差 ≤1e-14。实际 Collector 的双分量 occupancy、hard assignment、entropy 另用 2D 概率公式核对，误差 ≤1e-7。

直方图输入 [-2,-.1,0,.1,2,NaN,+Inf]，内边界 [-1,0,1]：桶计数 [1,1,2,1]、finite count=5、nonfinite=2、near-zero=3、sum=0、sum_sq=8.02。

GMM 分块以 chunk_m=1 强制跨四块采样；保留 flat positions [0,2] 时，记录 margin [2,0]、分母 2；中间 padding 不产生 token switch。

### Cross-Layer、Drift、Scalable

- Cross-Layer：两层反向 GMM，排除首 token 后的 gates 为 [1,0,0] 与 [0,0,1]，实际 Collector disagreement=2/3，peer module 精确对应相邻层。
- Drift：固定位置 [1,3]；z 从全零到 [(3,4),(0,0)] 再到 [(6,8),(0,0)]。第三阶段相对 first 的 representation L2 sum/count=10/2，相对 previous=5/2；gate flips 分别 2/2 和 1/2；margin shifts 分别 8/2 和 4/2；正/负 score shift 与已知数组分别核对。
- Probe ID、内容、顺序、位置维度必须匹配；新增 Adapter 不与尚不存在的历史 Adapter 比较。实际两层 ProbeRunner 跨 Stage 1→2 只对 Adapter 0/1 和层输出做历史匹配。
- 独立 ProbeRunner 实际执行各模式并检查参数/已有梯度、混合 Train/Eval 状态、EMA、RNG恢复；KL 用 NumPy FP64 独立公式核对（3e-7），answer NLL 用全有效 labels 独立公式核对（1e-6）。
- Scalable：2 层、7 矩阵、3 阶段、rank=4，应有 24,576 个 Adapter 参数、98,304 bytes；JL 为 20,480 bytes。K=2,D=8 的测试 GMM 参数模型每个 136 bytes，共 32 个独立正/负模型对象、4,352 bytes；scoring vectors 共 6,528 bytes，总参数存储 10,880 bytes。
- 本夹具每阶段显式 reload 负分布，形成不同 scoring vectors；因此上面的 scoring bytes 针对实际存储分配，不代表生产环境每次必然分配同样数量。共享负 GMM 对象仍去重。缓存/workspace 不计入参数 bytes，未测峰值内存或 GPU性能。
- Q/K/V、Gate/Up 使用真实共享函数：每阶段 14 个 matrix records、8 个 owner records；3 阶段分别 42、24，不能相加。相应存储使用实际共享指针去重。

## 5. 多阶段、落盘与副作用检查

Stage 0→1→2 真实 add_lora、backward、SGD step；检查新增前的所有旧参数逐位不变、历史 Adapter 冻结，新增非零 B 保证更新检查有效；多 Adapter 输出按固定矩阵 oracle 验证求和。

各 Stage/Epoch/Dataset/Adapter/Layer/Owner 分开记录。真实 save_cl_state/load_cl_state 函数在参数容器条件下执行后：同一随机 frozen base、重建 adapters/JL/GMM/scalings 后的 logits 和 Gate/GMM/Adapter 诊断记录一致（去除 forward_id）。**这不是实际 Pomegranate roundtrip，也不是 optimizer/RNG/EMA 的精确训练恢复。**

落盘检查包含严格 JSON、重新读取、sample_count=denominator、零分母 null、不同 rank 和 batch 的充分统计加权聚合。四批计数 (1/2,0/1,8/10,0/0) 的正确结果为 **9/13**，不是批次比例平均；不同实验、stage、参考阶段、比较模式、owner/matrix scopes 不混合。

副作用检查包含：

- disabled 无 hook、模块诊断属性或输出目录；Basic/Extended 与 baseline 相同 Torch linear 次数、GMM score 次数，root forward 均为 1。
- 一次训练 Forward 的历史评分为 28 次（14 matrices × pos/neg），未添加反事实 Forward；Extended 只在 CPU 用既有投影样本算责任统计。
- 递归检查 pending 张量全部 detached、无 grad_fn；采样后 pending/score bindings 清空；正常 Basic/Extended 不保留 last_snapshot。
- 连续三个 activation-checkpoint forward/backward，仅采样三个 root forward，反向重算不再收集；输入 IDs/mask 没有原地修改。
- 实际 Cache 逐项对照；Probe 强制 use_cache=False；混合模块 mode、参数/已有梯度、共享列表别名、RNG 在正常及异常路径恢复。
- 异常后 hook 清理、close 幂等、可再次 attach；诊断配额跨嵌套 eval 保持。

CPU 张量释放/图引用断言通过不等于测过 CUDA 显存峰值或长时间稳定性。Basic/Extended 的 CPU 汇总和 I/O 仍有运行时间开销，未声明“零开销”。

## 6. 本轮发现的问题与独立修复

每组均先运行新增回归测试复现失败，再修复并重新执行；未放宽原算法一致性断言。

| Commit | 修复 | 修复前证据 |
|---|---|---|
| `a62ecc9` | 聚合保持 reference/stage/epoch、comparison mode、probe identity 等维度；拒绝不支持的 occupancy/error 结构化 reducer；阻止不同实验共用输出目录 | 5 个回归用例失败 |
| `e814451` | 独立 Probe 异常恢复共享列表内容并保留对象别名；恢复采样预算，flush 异常也恢复 context | 2 个回归用例失败 |
| `6445b82` | NPZ 保留空 Probe 身份；加载替换原历史；验证失败前不提交新历史 | 3 个回归用例失败 |
| `8c8592d` | 本轮 CPU 集成与已知答案测试，无新增科研指标 | 所有测试纳入最终全套执行 |

原始 LSL 限制另行确认：没有 prefill 的 T=1 首次 Forward，在 smoothing<1 时直接访问缺失 `_smoothing_state[i]`，原 Tag 与研究分支均抛 KeyError(0)。测试记录其原有行为，**未修复或改变 Gate/EMA 算法**。其余原代码潜在问题仍以 OBSERVABILITY.md 的说明为准。

## 7. 尚未验证及正式实验前必检

1. 修复/准备目标环境的依赖兼容性，按仓库版本配置；本轮没有更改当前环境。完整随机 Tiny Qwen2ForCausalLM 直接 import/instantiate，执行四模式全模型测试。
2. 实际 Pomegranate 正负 GMM EM：相同数据、初始化、RNG，比较拟合参数、likelihood、gate、失败策略以及真实 CL checkpoint roundtrip。当前 EM 是 skipped，不能视作通过。
3. 在目标 GPU 上用生产 dtype、autocast、FlashAttention、gradient checkpointing、prefill/decode、目标 batch 等再次比较 A/B/C/D 的 logits/loss/梯度/step参数/Gate/EMA/KV/RNG。优先 bitwise；若内核本身非确定性，先测重复运行误差并说明每项容差与依据。
4. 使用原始数据顺序/样本、分布式 sampler 和完整原评测路径，验证 Accuracy/PPL/遗忘指标/GMM 命中率及原结果镜像落盘一致。本轮 synthetic CE/NLL/KL 不是正式评测结果。
5. 多 rank 真正并行验证：rank-local 收集、DDP hooks、异常清理、Checkpoint rank0、文件隔离与汇总；本轮只有离线模拟 rank records，未启动 DDP。
6. 生产保存/重载 Adapter/GMM/JL/disabled 状态，固定 base/tokenizer revisions，检查整个阶段间 Probe 重建。CL State/sidecar 缺 optimizer/scheduler/RNG/EMA/KV/data cursor，不能当成 exact resume checkpoint。
7. 实际长时训练和评测的显存、主存、CPU transfer、同步/I/O 时延测量。先启 Basic；Extended 需在相同采样预算下单独确认吞吐与内存开销。P2 继续仅在独立 Probe 路径执行。

上述必检项未完成前，建议将本次状态标记为 **CPU 核心路径验证通过，正式 LSL 复现准入待验证**，而不是“真实 GPU/7B 验证通过”。
