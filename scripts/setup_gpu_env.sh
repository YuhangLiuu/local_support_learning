#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT/.venv-gpu"
TORCH_INDEX=""
FLASH_VERSION="2.8.3"
DRY=0
while (($#)); do
  case "$1" in
    --venv) ENV_DIR="$2"; shift 2;;
    --torch-index) TORCH_INDEX="$2"; shift 2;;
    --flash-version) FLASH_VERSION="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
case "$TORCH_INDEX" in
  https://download.pytorch.org/whl/cu126|https://download.pytorch.org/whl/cu128|https://download.pytorch.org/whl/cu130) ;;
  *) echo 'Choose an official CUDA wheel index after GPU/driver inspection (--torch-index).' >&2; exit 2;;
esac
if ((DRY)); then
  echo "Create isolated venv: $ENV_DIR"
  echo "Install torch==2.9.0 from $TORCH_INDEX; GPU requirements; flash-attn==$FLASH_VERSION (build pending)"
  exit 0
fi
[[ "$(uname -s)" == Linux ]] || { echo 'Linux required' >&2; exit 2; }
nvidia-smi
nvcc --version
[[ ! -e "$ENV_DIR" ]] || { echo 'Refusing to modify an existing environment' >&2; exit 2; }
python3.11 -m venv "$ENV_DIR"
P="$ENV_DIR/bin/python"
"$P" -m pip install 'torch==2.9.0' --index-url "$TORCH_INDEX"
"$P" -m pip freeze > "$ENV_DIR/torch-before.txt"
"$P" -m pip install -r "$ROOT/reproduction/requirements-gpu.txt" 'torch==2.9.0' packaging psutil ninja
MAX_JOBS="${MAX_JOBS:-4}" "$P" -m pip install "flash-attn==$FLASH_VERSION" --no-build-isolation
"$P" -m pip check
"$P" -m nltk.downloader punkt_tab punkt
"$P" -m pip freeze > "$ENV_DIR/requirements-installed.txt"
echo 'Installation finished; run check_gpu_env.sh and real-model parity. No GPU validation has been inferred.'
