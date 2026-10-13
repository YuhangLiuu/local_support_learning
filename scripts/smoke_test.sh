#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_SIZE="${1:-1p5b}"
shift || true
case "$MODEL_SIZE" in 1p5b|7b) ;; *) echo 'Use 1p5b or 7b' >&2; exit 2;; esac
exec "${PYTHON:-python}" "$ROOT/scripts/run_reproduction.py" --config "smoke_$MODEL_SIZE" "$@"
