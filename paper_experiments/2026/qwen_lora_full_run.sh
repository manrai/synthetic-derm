#!/usr/bin/env bash
# Steps 3 and 4 of QWEN_LORA.md on two GPUs. Decisions from Steps 1-2: text-to-image uses a per-class LoRA
# (rank 16, lr 1e-4, 512 px, 2000 steps, checkpoint 1500, single prompt, seed-only variety); edit uses untuned
# Qwen-Image-2.1 with the 10 varied prompts (no LoRA).
#
#   bash paper_experiments/2026/qwen_lora_full_run.sh train-t2i <gpu> "<label>" ["<label>" ...]   # sequential
#   bash paper_experiments/2026/qwen_lora_full_run.sh gen-t2i  <gpu>      # all nine classes, LoRA checkpoint 1500
#   bash paper_experiments/2026/qwen_lora_full_run.sh gen-edit <gpu>      # all nine classes, untuned, varied prompts
#
# Generation covers every training-pool image of the nine labels (pool9.csv, written here from the training CSV),
# 10 images each, into /workspace/synderm/qwen_lora/generations/<method>/<00..09>/<md5hash>.png with JSON sidecars.
# qwen_lora_generate.py skips images that exist, so a re-run resumes.
set -euo pipefail
export HF_HOME=/dev/shm/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO=$(cd "$(dirname "$0")/../.." && pwd)
TRAIN="$REPO/paper_experiments/2026/qwen_lora_train.py"
GEN="$REPO/paper_experiments/2026/qwen_lora_generate.py"
RUN=r16-lr0.0001-512px-2000steps
OUT=/workspace/synderm/qwen_lora/generations
POOL9=$OUT/pool9.csv
NINE=("psoriasis" "lichen planus" "basal cell carcinoma" "allergic contact dermatitis" "lupus erythematosus"
      "neutrophilic dermatoses" "photodermatoses" "sarcoidosis" "squamous cell carcinoma")
mkdir -p "$OUT"
[ -f "$POOL9" ] || python - "$REPO/fitz_metadata/fitzpatrick17k_10label_clean_training.csv" "$POOL9" <<'PY'
import sys, pandas as pd
nine = ["psoriasis", "lichen planus", "basal cell carcinoma", "allergic contact dermatitis", "lupus erythematosus",
        "neutrophilic dermatoses", "photodermatoses", "sarcoidosis", "squamous cell carcinoma"]
d = pd.read_csv(sys.argv[1]); d = d[d["label"].isin(nine)][["md5hash", "label", "fitzpatrick_scale"]]
d.to_csv(sys.argv[2], index=False); print(len(d), "pool images over", d["label"].nunique(), "labels ->", sys.argv[2])
PY

case ${1:?train-t2i|gen-t2i|gen-edit} in
train-t2i)
  GPU=${2:?gpu}; shift 2
  for label in "$@"; do
    slug=${label// /-}
    if [ -f /workspace/synderm/qwen_lora/t2i/$slug/$RUN/pytorch_lora_weights.safetensors ]; then
      echo "skip $label: trained"; continue; fi
    echo "train t2i $label on GPU $GPU, $(date -u +%FT%T)"
    python "$TRAIN" --label "$label" --method t2i --gpu "$GPU" --steps 2000 --checkpoint-every 500
  done
  echo "train-t2i done $(date -u +%FT%T)"
  ;;
gen-t2i)
  GPU=${2:?gpu}
  for label in "${NINE[@]}"; do
    slug=${label// /-}
    lora=/workspace/synderm/qwen_lora/t2i/$slug/$RUN/checkpoint-1500/pytorch_lora_weights.safetensors
    [ -f "$lora" ] || { echo "missing LoRA for $label: $lora"; exit 1; }
    echo "gen t2i $label on GPU $GPU, $(date -u +%FT%T)"
    python "$GEN" --method t2i --gpu "$GPU" --reals "$POOL9" --label "$label" --k 10 --prompts single \
        --lora "$lora" --out "$OUT/t2i"
  done
  echo "gen-t2i done $(date -u +%FT%T)"
  ;;
gen-edit)
  GPU=${2:?gpu}
  echo "gen edit, all nine labels, on GPU $GPU, $(date -u +%FT%T)"
  python "$GEN" --method edit --gpu "$GPU" --reals "$POOL9" --k 10 --prompts varied --out "$OUT/edit"
  echo "gen-edit done $(date -u +%FT%T)"
  ;;
*) echo "usage: $0 train-t2i <gpu> <labels...> | gen-t2i <gpu> | gen-edit <gpu>" >&2; exit 1 ;;
esac
