# Verified Local Import：复用已有 1.5B 模型

接口复用 scripts/prefetch_resources.py：`--import-local --source modelscope --verify-sha256`，必须同时指定 `--config` 和 `--models`；不能与下载数据、tokenizer-only、inventory 或 check-only 混用。全程不联网、不下载、不改模型内容，只在全部验证成功后原子写入缓存侧车。Windows/Linux 均可导入及 check-only，真实下载仍仅允许服务器执行。

## 官方证据

reproduction/local_import_evidence.json 绑定 Qwen/Qwen2.5-1.5B-Instruct / `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。没有修改 resources.lock.json 的 revision 或文件清单。

2026-10-15 使用 Hugging Face 官方 SDK 查询 [指定 revision 的官方文件元数据](https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct/revision/989aa7980e4cf806f80c7fef2b1adb7bc71aa306?blobs=true)，确认返回 commit SHA、文件大小、Git blob IDs 和 LFS SHA256。权重大小3087467144，SHA256为 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`，来自本次独立API核对，而非只信任用户提供值。

其余六个小文件从官方 /resolve/<revision>/<filename> HTTPS 地址读取到内存，计算 `SHA1("blob " + byte_length + NUL + content)` 与官方 Git blob ID 比较，再计算 SHA256。未下载权重，未持久化小文件正文；Git仅提交哈希与来源证据。证据覆盖 config.json、generation_config.json、tokenizer_config.json、tokenizer.json、vocab.json、merges.txt、model.safetensors 共七个实际必需文件。

服务器按可信证据逐文件核对存在、大小和完整SHA256。锁中已有SHA也必须一致；缺失证据时拒绝，不退化为仅大小或“可以加载”。额外special_tokens_map等可能覆盖运行行为的未验证文件也会拒绝；ModelScope的configuration.json等独立记账文件不冒充官方文件。

## 真实来源与兼容性

缓存 `.lsl-resource.json` schema2使用 provenance_type=verified_local_import，记录 source_provider、source_revision、verified_against、verification_method、verified_files、verified_at。ModelScope版本未提供时source_revision=null/status=unknown；显式提供时只是declared_unconfirmed，master绝不会变成HF目标commit。source_provider是用户声明的实际提供方，内容验证不证明传输来源。

临时文件写入、flush/fsync后os.replace；失败不留下新的有效认证，已有Sidecar保持原样。重复导入重新哈希后可复用原认证。既有HF下载Sidecar兼容。对本地导入的Sidecar，共享Validator在普通Parity预检中仍逐文件重新SHA256，不仅信任Sidecar或文件大小，因此需要读取约3GB权重，但不会下载或加载模型来做认证。

只为本次1.5B指定revision提交了官方证据；其他模型/版本无证据会拒绝导入。没有修改GPU Parity、原算法、Diagnostics、实验配置或比较容差。

## CUDA 服务器命令

以下假设origin访问已恢复。已有输出不会覆盖；若同名文件或.artifacts目录已存在，需选择新的输出名称。

```bash
set -euo pipefail
cd /mnt/workspace/local_support_learning
git switch research/lsl-observability
git pull --ff-only origin research/lsl-observability
source /mnt/workspace/lsl-venv/bin/activate

python scripts/prefetch_resources.py \
  --config smoke_1p5b --models \
  --cache-root /mnt/workspace/lsl_cache \
  --import-local --source modelscope --verify-sha256

python scripts/prefetch_resources.py \
  --config smoke_1p5b --models \
  --cache-root /mnt/workspace/lsl_cache \
  --check-only --verify-sha256

python scripts/gpu_parity_test.py \
  --config reproduction/configs/smoke_1p5b.json \
  --cache-root /mnt/workspace/lsl_cache \
  --output /mnt/workspace/lsl_gpu_parity_cuda24g_verified.json
```

本地没有服务器模型，不能预先声称ModelScope的六个配置/Tokenizer文件匹配；任一不一致都列出失败文件并停止，不能手写认证绕过。真实CUDA/24GB Parity尚待服务器执行；模型可加载和单次推理通过不等同完整数值一致性通过。

本地最终验证：171 passed、0 failed、0 skipped，包含原144项与新增27项；3个warning来自既有源文件转义。pip check、compileall与git diff --check通过。测试使用真实临时文件/SHA256、原子替换失败模拟及实际CLI子进程；没有下载权重或安装环境。
