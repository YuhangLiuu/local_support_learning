#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:?Choose igbo, chemistry or cybersecurity}"
shift
case "$TASK" in igbo|chemistry|cybersecurity) ;; *) echo 'Unknown task' >&2; exit 2;; esac
exec "${PYTHON:-python}" "$ROOT/scripts/run_reproduction.py" --config "single_$TASK" "$@"
