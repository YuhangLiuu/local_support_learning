# LSL reproduction and observability

## Contract and baseline

The official source snapshot is the root commit tagged `baseline-lsl-original`.
All research changes are on `research/lsl-observability`; `main` stays at baseline.
See [PROVENANCE.md](PROVENANCE.md). No remote is configured.

Train once: original training/evaluation and low-frequency P0/P1 observation run
together. There are no extra model forwards, GMM re-fits, LoRA matmuls, dropout
calls, random samples, changed gates, loss terms or optimizer steps in the inline
observer. All logits/LoRA/GMM arithmetic keeps the original precision. Diagnostic
reductions operate on detached, bounded samples. Original known issues remain.

Expensive base/adapter counterfactuals, output KL, fixed-probe history and complete
layer-output comparisons are independent checkpoint workflows. Never invoke
ProbeRunner or profile functions from a training forward. Checkpoints allow later
analysis without retraining. They are **not complete optimizer-resume checkpoints**.

## Running one experiment

In a prepared Linux/GPU environment, from this repository:

```bash
export LSL_DIAGNOSTICS=true
export LSL_DIAGNOSTICS_LEVEL=basic
torchrun --nproc_per_node=1 --master_port=12929 train_continual.py
```

Default is off. Only two new environment variables above control observability.
`extended` adds bounded update norms, applied-update cosine, margin/norm sufficient
statistics, component occupancy/entropy and adjacent-layer disagreements.
`probe` and `profile` are for independent workflows, not the normal experiment.
Normal training accepts only `basic`/`extended` levels to prevent accidental P2 use.

Additional diagnostic-only Configuration fields:

| Field | Default | Meaning |
|---|---:|---|
| diagnostics_every_n_forwards | 100 | sample root forwards 1, 101, ... per context |
| diagnostics_max_batches | 4 | cap sampled forwards per dataset/mode/stage/epoch context |
| diagnostics_max_tokens | 512 | first valid positions across sampled rows; deterministic |
| diagnostics_output_dir | empty | default results/experiment ID |
| diagnostics_save_checkpoints | True | save all adapter/support states per epoch when diagnostics on |

Checkpoint saving adds disk I/O and a bounded pause, not a training update. Set the
last flag False only if an equivalent complete adapter/GMM snapshot is retained.
The observer does not iterate any loader or change sampler ordering.

## Outputs and alignment

```text
results/<timestamp>_<run_name>/
  config.json
  metadata.json
  reproduction/
    original_metrics.jsonl
    task_results.csv
  diagnostics/
    metadata.rank<R>.json
    gate_metrics.rank<R>.jsonl
    gmm_metrics.rank<R>.jsonl
    adapter_metrics.rank<R>.jsonl           # extended
    original_gmm_hit_rates.rank<R>.jsonl
    drift_metrics.rank<R>.jsonl             # independent fixed probes
    probe_metrics.rank<R>.jsonl             # independent fixed probes
    efficiency_metrics.rank<R>.jsonl        # explicit profiler
  checkpoints/
    cl_state_stage<S>_<task>_epoch<E>.pt
    cl_state_stage<S>_<task>_epoch<E>.json
```

Original `output/`, `results/..._best/cl_results.csv` and W&B output remain intact.
The reproduction JSONL mirrors original stage results, with index 0 zero-shot and
index e+1 trained epoch e. Stage/epoch are zero-based; initial evaluation uses -1.
It does not invent a new forgetting metric or reinterpret original hit rates.
Forgetting can be computed offline from the unchanged stage/task matrix with an
explicitly declared definition.

Every metric row has experiment_id, git_commit/git_dirty, rank/world_size,
dataset, stage, epoch, mode, adapter_id, layer_id, module, owner, metric_name,
metric_value, numerator, denominator and sample_count. Not-applicable dimensions
are null. Undefined averages are null, never fabricated zero measurements.
Metadata keeps complete configuration/model configuration, seeds, package versions,
and resolved model/tokenizer revision where exposed. Upstream dataset revisions
are not pinned: record/cache exact HF dataset revisions before formal reproduction.

## Metrics and interpretation

| Direction | Inline observations | Independent additions |
|---|---|---|
| Gate/support | per-stage/adapter/layer/matrix/owner activation, sequence-any activation, raw/applied disagreement, switches, coactivation, all-closed | fixed-probe coverage and mode comparison |
| Function-aware | actual applied update L2 and base-output norm ratio; applied-update cosine; margin/norm sufficient statistics | KL(base || LSL), KL(full || mode), update-mode difference and answer NLL/PPL |
| Cross-layer | adjacent-layer disagreement for matching owner types/adapters; optional bounded token traces | full layer output change vs base and drift across saved stages |
| Drift robust | stage-labeled original scores and gates | fixed IDs/order/preprocessing; projected input, positive/negative score, margin and decision drift vs first/previous checkpoint |
| Adaptive support | original positive/negative scores, margin histogram, near-zero proportion, fit disabled verdict | coverage for fixed ID/OOD/generic probes |
| Scalable | no timing inserted in training | deduplicated parameter inventory; separate prefill/decode/gate/LoRA workload timers |
| Online/task-free | ordered stage/epoch contexts; optional sampled gate token traces | WindowTracker consumes explicitly ordered offline windows; no detector or adapter creation |
| Training–inference | actual forced-on rate, observed raw/applied gate mismatch, mode labels | current-forced vs all-gated/no-smoothing vs native smoothing; optional adapter-off |

No margin is recomputed in basic. `GMMClassifierGPU.predict` exposes its existing
pos/neg tensors. On a forced-on, disabled or unfitted branch, scores may not exist:
`score_observed_rate` documents availability. There is no hidden additional score
forward. Extended posterior responsibilities are CPU fp64 diagnostics on sampled
**already projected inputs**, not authoritative routing scores.

Update cosine compares **applied** update vectors. Zero-norm pairs are excluded.
This is a conflict proxy, not proof of functional conflict. Norm ratio is mean of
per-token ratios with denominator norm floored at 1e-12. Correlation records store
n, sx, sy, sxx, syy and sxy; do not average per-batch correlations.

Padding and matching chat prefix are excluded from diagnostic denominators.
Prefix filtering is computed from this forward's token IDs independently of the
model's mutable prefix mask. It does not change the original gate. Token sampling
is bounded and biased toward early valid positions; these rates are not full-data
estimates without assessing that bias. Sequence metrics mean sampled rows of a
forward. Decode rows are not reconstructed full sequences. Switches count only
adjacent valid positions in that forward, not transitions across batches/calls.

Q/K/V share a GMM; gate/up share another. Matrix-scope rows describe actual updates;
owner-scope rows count each GMM owner once. **Never sum the two scopes.**
Coactivation counts valid tokens once per owner or matrix, not once per adapter.
Adjacent-layer comparisons match q-to-q, o-to-o, gate-to-gate, down-to-down.
No cosine compares hidden vectors of different dimensions.

`report.summarize_gate` groups owner-token exposures by layer/adapter or other
dimensions. This is a weighted mean, not an 'any matrix in layer active' rate.
Token traces for heatmaps require explicit `DiagnosticsConfig(save_token_traces=True)`
and are capped by max_trace_rows. No full-layer/full-sequence activations are saved
by normal inline basic/extended observation.

## Rank semantics

Every rank writes its own file; no collective is inserted into normal forwards.
`report.pool_records` pools **numerators and denominators**, never rank means.
It refuses histogram/correlation/probe/original-ratio records requiring specialized
reducers. Histograms pool bin counts with identical edges; component occupancy
requires matching fitted models/components and therefore should normally stay local.

Original GMM fits may differ by rank. Pooled results describe the mixture of those
rank-local models and shards, not one synchronized global gate. DistributedSampler
padding may duplicate samples: pooling is token-exposure weighted, not a unique-ID
global estimate. Use fixed IDs and explicit shard/dedup rules for global probes.

Original gmm_eval ratios are mirrored with unavailable denominator/sample_count
set null, marked not poolable. Original aggregate hit-rate semantics are preserved.

## Independent fixed probes, no retraining

Prepare fixed tokenized diagnostic/evaluation samples once, including stable IDs,
exact order, tokenizer revision, template, truncation and padding rules:

```json
{
  "preprocessing": {"version": "v1", "tokenizer_revision": "EXACT_HF_COMMIT", "max_length": 1024},
  "probes": [{"probe_id": "heldout-domain", "sample_ids": ["row-001"],
              "input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]],
              "labels": [[-100, -100, 3]]}]
}
```

The IDs above are synthetic format examples, not Qwen prompts. Use real fixed
tokenization when available. Labels are optional and must already mask prompts.
Only evaluation/diagnostic data is allowed; the runner does not feed probes back
into training. Fingerprints cover IDs, order, token IDs, masks and preprocessing.

```bash
python -m diagnostics.probe_runner \
  --base-model /local/Qwen2.5-7B-Instruct \
  --cl-state results/RUN/checkpoints/cl_state_stage0_igbo_translation_epoch2.pt \
  --probe-file /local/fixed_probes.json \
  --output results/RUN/diagnostics \
  --experiment-id RUN --stage 0 --epoch 2 \
  --history results/RUN/diagnostics/probe_history.npz \
  --device cuda --attention-implementation flash_attention_2
```

Both model and tokenizer are loaded with local_files_only=True. Use the same base
revision and backend as reproduction for comparable results. No downloads occur.
Repeat for later checkpoints using the same history and probe definitions;
the runner compares only historical adapters present in both checkpoints.
Optional `--adapter-off N` adds that counterfactual. First-pass base and full logits
are bounded to sampled tokens before CPU retention. Complete layer *coverage* is
available, but per-layer retained token count remains bounded.

Each independent mode uses eval dropout, fresh no-cache forwards and zeroed local
EMA. The current-forced mode changes only Linear wrapper training flags, not
dropout. Modes, RNG, shared disabled lists, EMA/prefix, pending buffers and hit-rate
state are restored even on exceptions. No parameter/gradient is modified. These
controlled modes are not the stochastic training forward or formal eval.

Checkpoint sidecars list what is absent: base weights, optimizer, scheduler, RNG,
data cursor, EMA and KV. Load by the independent runner, not the upstream training
load_cl_state_path path, which may continue training. Smoothing comes from the
saved configuration; base/tokenizer assets must be retained separately.

## Performance and overhead

Basic keeps small sampled masks/scores until the root forward ends, transfers
bounded samples and aggregates detached CPU scalars. It adds synchronization on
**sampled forwards**, not every token. Extended also transfers sampled update/base
vectors and owner features and computes CPU responsibilities: appreciably higher
overhead. No numeric overhead estimate is claimed without GPU measurement.

Use `efficiency_metrics.storage_inventory` and `profile_workloads` in a separate
process. Supply isolated prefill/decode/gate/LoRA callables; each repeat must reset
its own KV/EMA. CUDA timing uses events and one completion synchronization per
workload; CPU timing uses perf_counter. Profile diagnostics-off and diagnostics-on
separately with identical workload shapes. Storage inventory deduplicates JL,
Pomegranate parameters and shared scoring tensors, reports matrix-adapter pairs,
and excludes Pomegranate buffer caches/workspace. It is not total process GPU memory.

## Tests and mandatory launch gates

Existing environment: torchlink, PyTorch 2.14.0+cpu, no CUDA, no Pomegranate.
No packages were installed.

```powershell
New-Item -ItemType Directory -Path tmp -Force | Out-Null
& C:/Users/yuhangliu/miniconda3/envs/torchlink/python.exe -m pytest tests -q --basetemp=tmp/diagnostics-tests-NEW
```

Use a new basetemp name when Windows permissions prevent pytest temp cleanup.
CPU tests extract the actual immutable baseline and research Linear/GMM functions
via AST. The GMM constructor is replaced with deterministic synthetic fitted
parameters because Pomegranate is unavailable; predict/gate/LoRA arithmetic remains
actual source. Tiny-model tests require bitwise equal logits, loss, gradients,
AdamW-step parameters, RNG, EMA and scores for basic/extended train/eval. They also
check owner dedup, counts, prefix/padding, no graph retention, probe restoration,
schema and persisted history. This does not validate actual EM or Qwen attention.

Validation on 2026-10-12: **21 passed, 1 skipped** (Pomegranate unavailable).
All 18 new/modified Python files parse. `git diff --check` passes. Every original
source blob in the baseline matches the official source commit (only .gitignore
and PROVENANCE.md differ/add); main has exactly one root commit. Two parser warnings
come from an unchanged upstream docstring containing an invalid escape sequence.
No model weights, datasets, environment packages or CUDA workloads were downloaded/run.

Before formal reproduction, all items below must pass:

1. Actual Qwen/FlashAttention baseline vs diagnostics-off vs basic-on, same initial
   weights, LoRA/JL seed, batch and RNG: logits, loss, gates, gradients, one step.
2. Real negative/positive EM fits: identical sampled activations, fitted parameters,
   disabled outcomes and predictions. Optional Pomegranate CPU test runs when present.
3. Full prefill plus cached decode, padding/prefix, batch-size changes and EMA/KV
   state equality. Original T==1 initialization and mask-alignment risks remain.
4. Same held-out formal evaluations and cache policy: original metrics unchanged.
5. CUDA RNG and multi-rank counts, checkpoint round-trip, original forward hooks
   under gradient checkpointing; ensure backward recomputation is not sampled twice.
6. GPU memory/latency with diagnostics off/on; inspect no retained computation graph.

`validation.compare_training_step` and `compare_outputs` provide callable gates for
prepared disposable models/fit/eval routines. Default tolerance is zero. For
nondeterministic kernels, investigate the difference first, record a kernel-specific
reason and explicit atol/rtol; never silently weaken tolerances. Formal runs must
not start with unexplained differences.

## Paper settings and unresolved upstream issues

No original defaults were changed. Current defaults are not automatically the
paper's optimal settings: Table 3 uses chemistry K+=32/alpha_s=.1; Igbo K+=16/alpha_s=.3;
cybersecurity batch=16/rank=512/K+=32/alpha_s=.3. Multi-stage uses batch=32/rank=128,
K+=16/K-=32/alpha_s=.1, epochs 3/3/1, three seeds and benchmark cap 300.
Prepare exact manifests for the desired paper experiment rather than silently
normalizing defaults. This branch adds only diagnostic options.

Upstream risks retained: left-truncated prompt masking, Windows fork/SIGQUIT,
empty W&B key override, batch divisibility, per-rank unsynchronized GMM fits,
cached decoding EMA assumptions, checkpoint-load training semantics, and incompletely
pinned dependencies/datasets. Fixes require a separate proposal and baseline-aware
branch; none is bundled into observability.
