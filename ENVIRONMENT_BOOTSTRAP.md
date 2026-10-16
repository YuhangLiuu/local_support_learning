> 默认环境恢复已改为 [setup_lsl.sh 按需 Pip 路线](SETUP_LSL.md)：`bash scripts/setup_lsl.sh minimal` 或 `bash scripts/setup_lsl.sh full`。下文 Wheelhouse/旧安装步骤仅供显式选择的高级部署参考，不是默认前置条件。

# Persistent Wheelhouse 与实例环境恢复

目标是恢复应用环境，不安装或替换系统GPU栈。Windows只开发、CPU测试和读取官方制品元数据；没有在Windows下载Linux Wheels、预训练模型、数据或NLTK包，也没有创建新的训练环境。

## 依赖审计与两种 Profile

原 requirements.txt 含未固定依赖，Dockerfile针对NVIDIA镜像并在线构建FA；reproduction/requirements-gpu.txt是旧候选清单，传递依赖不完整。它们保留为来源记录，不作为新恢复系统的锁。没有使用污染环境的全量pip freeze生成唯一锁。

新锁位于 reproduction/environment/：profiles.json 声明应用根依赖及其固定传递闭包；artifacts.json 保存官方PyPI指定版本的Linux/Python通用Wheel及sdist文件名、大小、URL、SHA256、依赖元数据，以及固定NLTK Git revision的官方SHA256。元数据审计覆盖Linux x86_64、CPython3.11/3.12；实际Wheel标签和底层GPU约束仍由目标服务器验证。

|层/Profile|策略|
|---|---|
|GPU底层|复用当前实例base Python中的PyTorch、FlashAttention、Triton及CUDA/ROCm依赖；记录版本和真实导入路径；绝不通过应用安装器下载/覆盖|
|Minimal|73个固定应用/传递包；Transformers4.56.2、Tokenizers0.22.2、Hub0.36.2、Pomegranate1.1.2；不要求lm-eval/NLTK，只允许运行skip_lm_eval=True的实验|
|Full|97个固定应用/传递包；包含Minimal及lm-eval0.4.10、evaluate、NLTK、PEFT等；打包并验证punkt/punkt_tab；正式实验仍需独立数据预检及GPU Parity|

锁基于明确根依赖、已有CPU验证过的关键版本和官方依赖元数据审计，不等于完整GPU软件栈已验证。审计发现W&B0.22.3不接受Protobuf7.x，因此锁定6.33.5。GPU基线必须自行满足其依赖约束；不满足则FAIL，不强制升级系统PyTorch。

## 首次准备 Wheelhouse（只在GPU Linux服务器）

入口自动寻找当前实例的base GPU Python，绕开PATH中的旧venv；非标准路径可使用入口首位参数 `--python /actual/base/python` 或设置 LSL_SYSTEM_PYTHON。不要指向旧lsl-venv/bin/python。可设置LSL_BASE_IMAGE_ID为平台提供的不可变镜像标识；未提供时记录observed OS-release fingerprint，明确不声称知道registry镜像digest。

```bash
cd /mnt/workspace/local_support_learning
bash scripts/bootstrap_environment.sh --check --root /mnt/workspace/lsl_environment

# 只生成计划：不联网、不下载、不安装。
bash scripts/bootstrap_environment.sh --prepare --profile minimal --allow-build --root /mnt/workspace/lsl_environment
# 审阅终端/plan.json中的来源、大小、SHA256和源码构建项后，显式执行。
bash scripts/bootstrap_environment.sh --prepare --profile minimal --allow-build --download --root /mnt/workspace/lsl_environment --max-gib 8

bash scripts/bootstrap_environment.sh --prepare --profile full --allow-build --root /mnt/workspace/lsl_environment
bash scripts/bootstrap_environment.sh --prepare --profile full --allow-build --download --root /mnt/workspace/lsl_environment --max-gib 8
```

prepare要求当前Linux GPU base可用；--check只是底层清单检查，不代表应用验收PASS。计划保存于wheelhouses/<profile>/<stack-and-lock-id>/plan.json。预留至少下载量三倍加5GiB；--max-gib只控制下载预算，不是完整实验磁盘预算。

缺少匹配Wheel时，默认FAIL；--allow-build显式授权从已锁定SHA256的官方sdist构建应用Wheel。先创建临时build环境，离线安装固定pip25.2/setuptools80.9.0/wheel0.45.1/packaging26.3；构建使用--no-index --no-deps --no-build-isolation，不自动安装系统编译器、GPU依赖或未锁定构建依赖。缺少工具则明确失败。源码构建后的Wheel记录自身SHA256、来源SHA256和构建工具版本，后续恢复只使用这些缓存Wheel。

Full的NLTK包来自固定官方Git revision和index.xml中的SHA256，显式prepare --download时下载；Minimal不下载或验证NLTK。所有下载只来自已计划的官方files.pythonhosted.org/raw.githubusercontent.com制品。失败可重跑，无公网回退；完整manifest通过后才可用于恢复。

## 新实例一键离线恢复与验收

持久化Wheelhouse已经准备且软件栈兼容时：

```bash
# Minimal一键入口
bash scripts/bootstrap_environment.sh --install --profile minimal --root /mnt/workspace/lsl_environment --cache-root /mnt/workspace/lsl_cache

# Full一键入口（不自动启动训练）
bash scripts/bootstrap_environment.sh --install --profile full --root /mnt/workspace/lsl_environment --cache-root /mnt/workspace/lsl_cache
```

--install自动检测、校验全部Wheel哈希/标签/依赖闭包、创建当前实例专用venv、离线安装并验收。强制--no-index --require-hashes --no-deps --ignore-installed，应用包安装到venv，不卸载系统包。venv使用--system-site-packages共享GPU底层，但逐包检查所有锁定应用的metadata路径与关键模块真实导入路径/版本，拒绝系统Transformers5.x混入。Full还复制已验证的NLTK资源到venv/share/nltk_data。

环境ID包含当前base解释器路径/哈希、Python构建信息、实例身份、GPU软件栈和Wheelhouse绑定。实例身份组合boot ID、hostname、mount namespace、PID1启动时间/cgroup及可选LSL_INSTANCE_ID，避免不同容器共享宿主机boot ID时误复用。旧实例venv不直接复用；同一标识的PASS环境重用前也重新验收。无效旧目录保留，重建使用新后缀；不递归删除用户环境。只在PASS后原子发布current-minimal.json/current-full.json。安装中断或任何检查失败都不会发布新的ready指针。

Python ABI、架构、libc、GPU后端/版本/架构、镜像标识或应用锁不匹配会明确FAIL。不能将cp311二进制Wheel用于cp312，也不能将CUDA准备结果冒充ROCm。没有匹配缓存时不会联网修复，需针对当前实例明确执行prepare。缺少有效系统PyTorch/FA时选择合适的GPU镜像；本系统不负责强行构建GPU栈。

bash子进程不能激活父终端；PASS时会输出实际 `source <env>/bin/activate` 路径。current-<profile>.json也保存env/report路径。激活该新路径后才运行后续实验。

## 验收范围与启动门槛

--verify重新验收current指针对应环境；check_environment.py也可用目标venv Python独立执行。报告保存到持久化root/reports，输出ENVIRONMENT PASS或ENVIRONMENT FAIL与JSON。报告检查解释器/链接/ABI、所有应用pin及实际导入路径、底层GPU版本/真实导入路径、Pomegranate/lm-eval、原Qwen/GMM/Diagnostics导入、CPU GMM评分、随机Tiny Qwen+LoRA+Basic Diagnostics前后向、BF16 GPU前后向、FlashAttention真实Kernel前后向、缓存目录访问、Full的NLTK资源哈希与可发现性。Full还使用内存Tokenizer和随机CPU Tiny Qwen执行一次真实HFLM模型调用，不加载任何Benchmark或预训练权重。

缓存检查仅验证目录可访问，不声称模型/数据已下载。随机Tiny CPU自检不是7B验证，也不是两阶段真实EM的实验级Parity。GPU核心检查使用torch.cuda API，同时支持HIP；不依赖nvidia-smi/nvcc判定ROCm。

受管环境启动run_reproduction.py时自动读取bootstrap绑定的PASS报告；也可对其他环境显式传 --environment-report <report.json>。报告必须匹配当前解释器、实例、GPU栈、导入路径、锁和验收代码。Minimal不能启动启用通用评测的正式配置。未受管的旧环境保留兼容入口，但也执行关键版本检查，不允许Transformers5.x/不兼容Hub混入。

## lm-eval 按需导入

HFLM及原HFWithExternalCache包装类移动到 reproduction/lm_evaluation.py 的延迟加载函数；实际评测之前才加载，缓存/return_dummy分支提前返回。skip_lm_eval=True不加载lm_eval、其数据集或NLTK，不生成虚假的非零分数。原有零占位结果仍只是占位，不是已测Benchmark。

Full在启动器和worker的pre-model/pre-seed边界完成依赖预检/加载，保留原评测batch、few-shot、生成方式、cache wrapper及指标计算。直接原训练入口也提前检查Full依赖；硬件报告校验安排在原pre-CUDA DataLoader准备之后，避免引入fork-after-CUDA行为。缺少lm-eval0.4.10/NLTK时明确失败。现有formal JSON、LoRA/GMM/Gate/Optimizer/Loss/Diagnostics均未修改。

本轮train_continual.py因按需导入而发生文件变化，旧Parity Receipt的core_hash可能失效。保持原Receipt门槛，不伪造兼容；请在新环境中单独重新运行实验级GPU Parity。bootstrap绝不会自动运行它，也不会每次恢复都启动完整Parity/正式训练。Checkpoint和Diagnostics状态格式不变。

## 验证边界

本地只执行CPU/Mock/Synthetic验证：真实临时Wheel ZIP/SHA、ABI匹配、路径隔离、依赖闭包、网络失败、部分安装、旧环境/重复恢复，以及阻断lm_eval的真实训练模块导入。真实HFLM包装行为和原版/新版正式harness的调用参数、指标汇总通过受控后端运行对照；没有下载Benchmark或模型。

Linux离线pip安装、sdist构建、真实GPU BF16/FA Kernel、新venv的真实系统包解析顺序及CUDA/ROCm完整恢复仍需目标实例实测。本地CPU通过不得替代这些验收；脚本在检查失败时只输出FAIL，不会将Mock结果写成真实GPU PASS。

最终CPU回归230 passed、0 failed、0 skipped，原184项保留、新增46项。7个warning来自既有字符串转义提示。测试临时复用已有Hub0.36.2包以避开本地2.2.0与Transformers的冲突，没有安装或替换环境。记录见validation/environment_bootstrap.json。
