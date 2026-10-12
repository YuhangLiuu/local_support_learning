"""Dedicated profiling only. Never called from normal training instrumentation."""

import time


def storage_inventory(model):
    from modeling_qwen2 import ConditionalLoRALinear
    seen = set()
    gmm_objects = set()
    counts = dict(adapter_parameters=0, adapter_bytes=0, gmm_parameter_bytes=0,
                  gmm_scoring_bytes=0, gmm_model_parameter_bytes=0,
                  jl_bytes=0, gmm_count=0, adapter_count=0)
    def size(t):
        key = (t.device.type, t.device.index, t.data_ptr())
        if key in seen:
            return 0
        seen.add(key)
        return t.numel() * t.element_size()
    for m in model.modules():
        if not isinstance(m, ConditionalLoRALinear):
            continue
        counts["adapter_count"] += m.num_loras
        for p in list(m.lora_A) + list(m.lora_B):
            counts["adapter_parameters"] += p.numel()
            counts["adapter_bytes"] += size(p)
        if m.jl_P is not None:
            counts["jl_bytes"] += size(m.jl_P)
        if m.is_cl_memory_owner:
            for clf in m.cl_gmm:
                for gmm in (clf.pos_gmm, clf.neg_gmm):
                    if gmm is not None and id(gmm) not in gmm_objects:
                        gmm_objects.add(id(gmm))
                        counts["gmm_count"] += 1
                        counts["gmm_model_parameter_bytes"] += sum(size(t) for t in gmm.parameters())
                for params in clf.params_vectorized.values():
                    if params is None:
                        continue
                    counts["gmm_scoring_bytes"] += sum(size(t) for t in params)
    counts["gmm_parameter_bytes"] = counts["gmm_scoring_bytes"] + counts["gmm_model_parameter_bytes"]
    counts["gmm_parameter_bytes_note"] = "deduplicated model parameters and vectorized scoring tensors; excludes pomegranate buffer caches/workspace"
    counts["adapter_count_note"] = "matrix-adapter pairs, not learning phases"
    return counts


def profile_call(call, device="cpu", repeats=10, warmup=2):
    """Caller supplies an isolated/restored forward, prefill or decode workload.

    No per-token synchronization. CUDA events enclose whole calls; one sync per
    measured workload. Allocator peak is process-wide and reset only here.
    """
    if repeats <= 0 or warmup < 0:
        raise ValueError("invalid profiling repeat count")
    import torch
    is_cuda = torch.device(device).type == "cuda"
    for _ in range(warmup):
        call()
    if is_cuda:
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                call()
            end.record()
            end.synchronize()
            return dict(milliseconds=start.elapsed_time(end)/repeats,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        repeats=repeats, timer="cuda_events")
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    return dict(milliseconds=1000*(time.perf_counter()-start)/repeats,
                repeats=repeats, peak_allocated_bytes=None, timer="cpu_wall")


def profile_workloads(workloads, writer, context, device="cpu", repeats=10, diagnostics_on=False):
    """Named isolated callables: prefill/decode/gate/lora; never inferred from training.

    Caller must reset KV/EMA per repeat and use identical shapes/weights for on/off.
    Gate/LoRA callables can be microbenchmarks; report them separately from end-to-end.
    """
    results = {}
    for name, call in workloads.items():
        result = profile_call(call, device, repeats)
        results[name] = result
        writer.write("efficiency_metrics", context, metric_name=name + "_milliseconds",
                     numerator=result["milliseconds"] * repeats, denominator=repeats,
                     diagnostics_on=diagnostics_on, workload=name, **{k:v for k,v in result.items() if k != "repeats"})
    return results
