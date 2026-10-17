#!/usr/bin/env bash
# Two independent modes: --check never trains; --run explicitly starts one Pilot.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:---check}"
[[ $# -le 1 && ( "$MODE" == --check || "$MODE" == --run ) ]] || { echo 'Use --check or --run' >&2; exit 2; }
P="${PYTHON:-python}"
CACHE="${LSL_CACHE_ROOT:-$ROOT/.cache/lsl}"
OUT="${LSL_OUTPUT_ROOT:-/mnt/workspace/lsl_runs}"
SAMPLES="${LSL_SAMPLE_MANIFEST:-/mnt/workspace/lsl_pilot_samples/seed123.json}"
RECEIPT="${LSL_PARITY_RECEIPT:?Set the passing Rank128/JL256/K16/32 receipt}"
ID="continual_1p5b_pilot_seed123"
mkdir -p "$OUT"
exec 9>"$OUT/.lsl-pilot-seed123.lock"
flock -n 9 || { echo 'Another Pilot command holds the lock' >&2; exit 2; }
"$P" scripts/check_gpu_env.py --config continual_1p5b_pilot_seed123 --cache-root "$CACHE" --require-gpu
"$P" scripts/prefetch_resources.py --config continual_1p5b_pilot_seed123 --cache-root "$CACHE" --models --datasets --check-only --verify-sha256
ARGS=(--config continual_1p5b_pilot_seed123 --seed 123 --cache-root "$CACHE" --output-root "$OUT" --experiment-id "$ID" --parity-receipt "$RECEIPT" --acknowledge-issues)
"$P" scripts/run_reproduction.py "${ARGS[@]}" --dry-run
"$P" scripts/run_reproduction.py "${ARGS[@]}" --preflight-only
if [[ "$MODE" == --check ]]; then
    "$P" scripts/prepare_pilot_samples.py --cache-root "$CACHE" --output "$SAMPLES"
    echo 'PILOT CHECK PASS; no training started'
else
    [[ -f "$SAMPLES" ]] || { echo 'Run --check to prepare the fixed sample manifest first' >&2; exit 2; }
    exec "$P" scripts/run_reproduction.py "${ARGS[@]}" --sample-manifest "$SAMPLES"
fi
