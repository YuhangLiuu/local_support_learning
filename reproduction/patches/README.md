# 未应用的可选 label mask 修复

left_truncation_mask.patch 仅供后续独立对照，不应用到本研究分支正式路径。

保留长度 L=min(full_length, seq_limit)，移除数量 R=max(0,full_length-L)，保留prompt长度=max(0,prompt_length-R)。原实现直接使用完整prompt_length，导致部分答案标签被屏蔽。

构造答案已知例子：full=12，prompt=8，limit=6。原mask屏蔽全部6个token；修复mask屏蔽2个，保留4个答案token。full=5,prompt=3,limit=6时两者相同。额外完整tokenization有CPU开销，不能称为无副作用的诊断修改。

未来比较应使用同一初始化/seed/数据顺序，分别原版与补丁版，报告有效label计数、loss、PPL和任务结果差异；另做无截断控制组。补丁会改变训练目标，因此属于独立算法/数据处理对照，不能混入原论文复现。
