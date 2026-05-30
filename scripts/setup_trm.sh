#!/usr/bin/env bash
# Phase 2 setup: fetch the TRM ARC-AGI-1 reproduction checkpoint from HF Hub.
#
# Source: Sanjin2024/TinyRecursiveModels-ARC-AGI-1 (reproduction of
# SamsungSAILMontreal/TinyRecursiveModels). Reported test pass@2 = 41.00%,
# which clears Phase 2's ≥40% baseline gate.
set -euo pipefail

REPO_ID="${TRM_HF_REPO_ID:-Sanjin2024/TinyRecursiveModels-ARC-AGI-1}"
DEST_DIR="${TRM_CKPT_DIR:-$(dirname "$(readlink -f "$0")")/../third_party/checkpoints/arc-agi-1}"

mkdir -p "$DEST_DIR"
echo "[setup_trm] downloading $REPO_ID -> $DEST_DIR"

export REPO_ID DEST_DIR
/usr/bin/python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id=os.environ["REPO_ID"],
    local_dir=os.environ["DEST_DIR"],
    allow_patterns=["*.yaml", "*.py", "README*", "step_*", "*.safetensors", "*.pt"],
)
print("[setup_trm] downloaded to:", path)
PY

echo "[setup_trm] contents:"
ls -lh "$DEST_DIR"
