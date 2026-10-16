#!/usr/bin/env bash
# Single explicit A10 experiment; no downloads, seed sweep or automatic Parity run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
P="${PYTHON:-python}"
CACHE="${LSL_CACHE_ROOT:-$ROOT/.cache/lsl}"
RECEIPT="${LSL_PARITY_RECEIPT:?Set LSL_PARITY_RECEIPT to your passing rank128/JL256 1.5B receipt JSON}"
AUDIT="${LSL_EVALUATOR_AUDIT:-$ROOT/validation/evaluator_audit.json}"
ID="continual_1p5b_lsl_a10_seed123"
OUT="${LSL_OUTPUT_ROOT:-/mnt/workspace/lsl_runs}"
if [[ $# -ne 0 ]]; then echo 'No extra arguments allowed; this entry launches only A10 seed 123.' >&2; exit 2; fi
mkdir -p "$OUT"
exec 9>"$OUT/.lsl-seed123.lock"
flock -n 9 || { echo 'Another seed123 launch holds the lock.' >&2; exit 2; }
"$P" scripts/check_gpu_env.py --config continual_1p5b_lsl_a10 --cache-root "$CACHE" --require-gpu
"$P" scripts/prefetch_resources.py --config continual_1p5b_lsl_a10 --cache-root "$CACHE" --models --datasets --check-only --verify-sha256
ARGS=(--config continual_1p5b_lsl_a10 --seed 123 --cache-root "$CACHE" --output-root "$OUT" --experiment-id "$ID" --parity-receipt "$RECEIPT" --evaluator-audit "$AUDIT" --acknowledge-issues)
"$P" scripts/run_reproduction.py "${ARGS[@]}" --dry-run
"$P" scripts/run_reproduction.py "${ARGS[@]}" --preflight-only
exec "$P" scripts/run_reproduction.py "${ARGS[@]}"
