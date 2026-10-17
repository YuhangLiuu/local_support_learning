# Three-stage 1.5B Pilot, seed 123

This is a Hardware-Adapted Research Pilot, **not paper-exact reproduction**.
The formal A10/7B plans, training functions, model, GMM, gate, smoothing and
Diagnostics definitions are unchanged. The existing Parity geometry and core
hash remain applicable; the launcher still checks actual environment, revision,
geometry and GMM failure policy. A Rank4 Smoke receipt is rejected.

## Parameters and scale

- Qwen/Qwen2.5-1.5B-Instruct, revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306.
- Igbo Translation -> Chemistry -> Cybersecurity; seed 123 only; epochs 1/1/1.
- At most 2000 training rows per task, 100 held-out rows per task and paired benchmark.
- Rank128, alpha256, LR1e-4, global batch32, microbatch1, accumulation32.
- Train/eval maximum length1024; FP32 base/BF16 autocast; FA2/checkpointing unchanged.
- Positive K16, negative K32, JL256, smoothing0.1; positive/negative fit budget16384.
- Original skip-on-failure policy remains True. Zero active owners aborts Pilot.
- Extended (includes Basic), every100 forwards, at most2 batches/context, 64 tokens/batch.
- Every epoch CL State saved; W&B disabled; general LM evaluation omitted entirely.
- At most6000 training microbatches, 189 optimizer steps, 3 CL checkpoints;
  1 negative fit and 3 positive fits. Actual fit tokens can exceed the budget by
  the last batch or fall short if the selected training split is exhausted.

## Selection and provenance

`--check` processes already cached, pinned data using the **unchanged upstream**
loaders. Original caps, seeded train/test partitions, cleaning and benchmark
train blending run before Pilot selection. Recursive benchmark loads are not
capped while building the parent's legal mixed train pool. Each task/split uses
its own local RNG derived from seed123 and its name; global RNG is untouched.
The Igbo PPL and chrF evaluations use the same held-out indices.

`sampling_manifest.json` records repo IDs/revisions/source splits, original legal
pool sizes, selected indices/counts, index SHA256, pool fingerprints and selected
row content hashes. Indices are **positions in the upstream processed legal
pool**, not raw Hub row offsets. The upstream transforms/caps/blending are needed
to reconstruct that pool. Original text, collate, chat templates and label masks
remain unchanged. Selected train/eval content overlap is rejected. No test rows
are moved into training.

The check prepares `/mnt/workspace/lsl_pilot_samples/seed123.json` by default.
An existing file must replay exactly; changed sources/pools/content/indices are
rejected. The run copies the verified selection into its experiment directory.
A later ordinary LoRA plan should use the identical task/data controls and pass
`--sample-manifest` pointing to this file; method changes do not affect selection.
No new LoRA experiment is launched by this task.

Positive GMM loaders receive exactly the selected task training pools. Negative
sampling and reuse follow upstream unchanged. `progress.jsonl` adds actual fit
tokens, active-owner ratio and disabled owner/adapter identities in Pilot only;
existing Gate/GMM metrics are saved by Diagnostics. Partial nonzero survival is
accepted and reported, not described as 100% fit success. Formal completion
acceptance is unchanged.

## Separate server commands

Run in the validated LSL Python environment; preserve the server's actual cache
and receipt paths in `LSL_CACHE_ROOT` and `LSL_PARITY_RECEIPT`.

```bash
# A: inspect/preflight/dry-run and prepare/recheck fixed samples; never train
cd /mnt/workspace/local_support_learning
git pull --ff-only origin research/lsl-observability
bash scripts/run_continual_1p5b_pilot.sh --check

# B: only after reviewing A; same selection and seed, no seed sweep
bash scripts/run_continual_1p5b_pilot.sh --run
```

Output defaults to `/mnt/workspace/lsl_runs/continual_1p5b_pilot_seed123`;
conflicting outputs are refused. `LSL_OUTPUT_ROOT` changes the parent directory,
`LSL_SAMPLE_MANIFEST` the selection file. Logs, progress, original Stage x Task
scores, diagnostics and epoch CL States share this experiment ID. The original
cross-task evaluator also keeps future-task baseline evaluation. Igbo uses chrF;
Chemistry uses ChemBench MCQ accuracy; Cybersecurity uses SecEval set-F1, with
original answer-PPL evaluations retained. There are no GSM8K/HumanEval/IFEval
placeholder scores in Pilot metrics.

## Optional isolated speed check (GPU only)

After A, an explicit command runs about100 microbatches under a separate ID:

```bash
python -u scripts/benchmark_pilot.py --sample-manifest "$LSL_SAMPLE_MANIFEST"   --parity-receipt "$LSL_PARITY_RECEIPT" --cache-root "$LSL_CACHE_ROOT"
```

If using the default selection path, set
`LSL_SAMPLE_MANIFEST=/mnt/workspace/lsl_pilot_samples/seed123.json` first.
This uses the real selected Igbo data, real model/LoRA, production label-mask and
loss normalization semantics, microbatch1, length1024 padding, FP32/BF16 and
checkpointing. It reports forward/backward timings, wall throughput, peak CUDA
allocation and a6000-microbatch training-only estimate. It does not save Pilot
checkpoints or alter the shared selection file. It measures stage0 only, not
later adapter gating; GMM, task evaluation and checkpoint I/O need additional
time. It uses a fixed LR for timing rather than claiming a scientific training
result. No GPU benchmark is automatically run by setup/check.

## Diagnostics limits and acceptance

Low-frequency Extended reuses existing forwards. Cross-layer and multi-adapter
statistics are available when the existing collector observes the relevant
paths. Historical representation/score drift and counterfactual comparisons
that require fixed probes remain separate existing Probe workflows; they are
not promised as automatic live metrics. CL State includes adapters/GMM/gates,
not optimizer/scheduler/RNG resume state. No extra diagnostic training run is
required.

The CPU tests cover real Dataset selection, original chemistry splitting and
mixing, the actual dataloader construction, replay, no-overlap guards, GMM fit
progress hooks (including exception cleanup), dry-run and artifact acceptance.
Windows CPU tests do not validate A10 runtime, memory/throughput, live three-stage
metrics or the locked remote dataset contents. Server check, optional speed
check, and the first actual Pilot remain necessary.
