#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == --python ]]; then
  P="${2:?Specify base Python}"; shift 2
  exec bash "$ROOT/scripts/bootstrap_environment.sh" --python "$P" --prepare "$@"
fi
exec bash "$ROOT/scripts/bootstrap_environment.sh" --prepare "$@"
