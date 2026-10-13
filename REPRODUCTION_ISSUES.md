# 问题与验证边界

|性质|问题|处理|
|---|---|---|
|原代码可确认|左截断prompt mask偏移|保留原代码，未应用补丁另存|
|论文未公开|seed IDs、精确评测子集|记录项目选择，不宣称作者相同|
|论文/代码差异|merge描述与并行冻结LoRA|保持官方行为、标记对照差异|
|原代码|负数据重划分与训练cap/blend|固定变换、记录数量|
|原代码|rank-local GMM / sampler未set_epoch|暂禁正式多rank入口|
|原代码|FP32基础权重、harness硬编码batch|按实际行为估显存并实测|
|原代码边界|初始单token平滑、padding增量cache风险|见CPU报告，GPU补验，不静默改算法|
|状态保存|CL State没有optimizer/scheduler/RNG/EMA/KV|用于恢复评测，不是完整续训|
|论文定义|aggregate retention不唯一明确|单列code_total_retention与任务比率|
|本轮wrapper修复|evaluator预热与launcher外部缓存不一致|统一cache_root|
|本轮wrapper修复|空verified_files可能漏检|拒绝空校验记录|
|本轮wrapper修复|异常退出可能遗留hooks|finally关闭observer|

GPU OOM、kernel非确定性及作者未公开选择仍是风险/未知，不称已验证故障。当前没有CUDA/7B/正式论文指标；CPU测试与dry-run不能代替目标GPU验收。资源锁不是本地缓存：全部not_cached，下载进程已停止，指定本轮资源已移入回收站，不表示永久清除或空间已释放。
