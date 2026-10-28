#!/usr/bin/env bash
# Rebuild the synderm working environment on a fresh GPU pod whose /workspace volume survived.
# Everything large (model weights, data, LoRAs, executed vignettes, backups) already lives on
# /workspace; this script only restores what sat on the pod's ephemeral root disk.
#
#   bash bootstrap_pod.sh            # from a clone of the repo, or from /workspace/backups
set -euo pipefail

export HF_HOME=/workspace/.cache/huggingface
export SYNDERM_DATA_DIR=/workspace/data
export SYNDERM_WORK_DIR=/workspace/synderm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
grep -q SYNDERM_WORK_DIR ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<'RC'
export HF_HOME=/workspace/.cache/huggingface
export SYNDERM_DATA_DIR=/workspace/data
export SYNDERM_WORK_DIR=/workspace/synderm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RC

# 1. Source tree: prefer the GitHub branch, else the git bundle in /workspace/backups.
REPO=${REPO:-/root/synthetic-derm}
if [ ! -d "$REPO/.git" ]; then
  BUNDLE=$(ls -t /workspace/backups/synthetic-derm-*.bundle 2>/dev/null | head -1 || true)
  if git ls-remote --exit-code --heads https://github.com/manrai/synthetic-derm modernize >/dev/null 2>&1; then
    git clone -b modernize https://github.com/manrai/synthetic-derm "$REPO"
  elif [ -n "$BUNDLE" ]; then
    git clone -b modernize "$BUNDLE" "$REPO"
  else
    echo "no repo: push the modernize branch or restore /workspace/backups/*.bundle" >&2; exit 1
  fi
fi
cd "$REPO"

# 2. Python environment. torch must match the pod's CUDA driver; the lock file records the
#    exact versions that ran Stage A (torch 2.8.0+cu128, diffusers at a pinned GitHub commit).
python -m pip install --upgrade pip
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "torch already present: $(python -c 'import torch; print(torch.__version__)')"
else
  python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
fi
python -m pip install -r requirements-lock.txt --no-deps 2>&1 | tail -2 || {
  echo "lock file failed (new CUDA/driver?): falling back to requirements.txt" >&2
  python -m pip install -r requirements.txt
}
python -m pip install -e . --no-deps

# 2b. The /workspace FUSE mount (geesefs) drops symlinks on remount, which turns every file in
#     the HF cache's snapshots/ into an empty file while the blobs survive. Rebuild the links.
python repair_hf_cache.py || echo "WARNING: some cached model files could not be relinked; see above" >&2
python -m pip install -q openai google-genai  # proprietary-model screen (paper_experiments/2026)

# 3. Sanity checks: GPU, model cache, token, data.
python - <<'PY'
import os, torch, synderm
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("synderm", synderm.__version__, synderm.__file__)
hub = os.path.join(os.environ["HF_HOME"], "hub")
print("cached models:", sorted(d for d in os.listdir(hub) if d.startswith("models--")) if os.path.isdir(hub) else "NONE")
print("hf token:", "present" if os.path.exists(os.path.join(os.environ["HF_HOME"], "token")) else "MISSING: run `hf auth login`")
for p in ("/workspace/data", "/workspace/data/fitzpatrick17k/finalfitz17k", "/workspace/synderm/vignettes/executed", "/workspace/synderm/lora"):
    print(p, "ok" if os.path.isdir(p) else "MISSING")
PY
python -m pytest tests/test_cpu.py -q 2>&1 | tail -1
echo "done. Run one GPU job at a time: jupyter lab synderm/docs/examples/"
