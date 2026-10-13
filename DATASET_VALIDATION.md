# 数据审计

resources.lock.json 固定两个模型及九个数据集 revision。当前全部 not_cached；本地下载及生成缓存已移入回收站。保留 validation/dataset_audit.json 为清理前已执行 CPU 审计的元数据，不代表服务器缓存存在。

|原函数处理后任务|训练|测试|
|---|---:|---:|
|Igbo|60000|1192|
|ChemData + ChemBench train|23709|1000|
|Cyber + SecEval train|53883|1000|
|ChemBench|3709|300|
|SecEval|1682|500|
|prepared negatives|3506|400|

负仓库 native test400 被原函数忽略，从 train3906 另划400。ChemBench val45 为 few-shot 来源，默认few-shot0。持续学习benchmark限额不改变上述原始划分。

已执行 schema/数量、任务内 train/test 精确整行重叠（0）、固定seed fingerprint、真实tokenizer模板/padding、三个训练集全量mask审计。跨benchmark精确内容匹配的新增检查尚未执行；语义泄漏未验证。

|seq1024 mask审计|扫描|左截断/偏移|全部答案被屏蔽|损失答案tokens|
|---|---:|---:|---:|---:|
|Igbo|60000|14|2|1416|
|Chemistry|23709|219|173|8449|
|Cyber|53883|0|0|0|

Chemistry空prompt20；Cyber空prompt1、空answer1。原训练/PPL未扣除左截断token数，非诊断引入；本轮不修改原代码。可选补丁及对照见 reproduction/patches/README.md。

服务器重新执行 verify_datasets.py --sample-limit 0，并用 verify_evaluators.py 预热/验证 GSM抽取、HumanEval canonical code、IFEval约束。HumanEval会执行 canonical Python，使用受限运行环境。实际 harness、模型生成、正式指标一致性尚待服务器，不能将脚本存在视为验证通过。
