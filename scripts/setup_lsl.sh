#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P="${LSL_SYSTEM_PYTHON:-}"
if [[ "${1:-}" == --python ]]; then P="${2:?Specify current instance base Python}"; shift 2; fi
if [[ -z "$P" ]]; then
  # Ignore a stale venv at the front of PATH. Prefer an actual GPU base Python.
  for candidate in /opt/conda/bin/python /usr/local/bin/python3 /usr/bin/python3; do
    if [[ -x "$candidate" ]] && "$candidate" -I -c 'import sys,torch,flash_attn; assert sys.prefix == sys.base_prefix and torch.cuda.is_available()' >/dev/null 2>&1; then P="$candidate"; break; fi
  done
fi
if [[ -z "$P" ]] && command -v python3 >/dev/null 2>&1; then
  if candidate="$(python3 -I -c 'import sys; print(sys._base_executable)' 2>/dev/null)"; then
    if [[ -x "$candidate" ]] && "$candidate" -I -c 'import sys,torch,flash_attn; assert sys.prefix == sys.base_prefix and torch.cuda.is_available()' >/dev/null 2>&1; then P="$candidate"; fi
  fi
fi
if [[ -z "$P" ]]; then
  echo 'LSL ENVIRONMENT FAIL: no usable base GPU Python found. Set LSL_SYSTEM_PYTHON or use --python /actual/base/python.' >&2
  exit 2
fi
exec "$P" -I "$ROOT/scripts/setup_lsl.py" "$@"
