#!/usr/bin/env bash
# Step 1A of QWEN_LORA.md: generate the psoriasis pilot images for every LoRA checkpoint and for untuned Qwen,
# then score every folder next to the GPT Image varied-prompt pilot. Two GPUs: edit runs on GPU 0, t2i on GPU 1.
#
#   bash paper_experiments/2026/qwen_lora_pilot.sh psoriasis r16-lr0.0001-512px-2000steps
#
# Reals: the 16 psoriasis images of the API pilot (reals.csv of that folder). 10 images per real, seed-only variety
# (--prompts single). Output folders: /workspace/synderm/qwen_lora/pilot/<method>-<label>-<ckpt>-single/.
set -euo pipefail
LABEL=${1:?label}; RUN=${2:?run name}; SLUG=${LABEL// /-}
export HF_HOME=/dev/shm/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO=$(cd "$(dirname "$0")/../.." && pwd)
REALS=/workspace/synderm/pilot/gpt-image-2.5-sunburst-medium-varied-seed20261028-n16/reals.csv
PILOT=/workspace/synderm/qwen_lora/pilot
LOG=/workspace/logs/pod-2026-10-28b
mkdir -p "$PILOT" "$LOG"

generate_all() {  # one method on one GPU, sequentially: untuned, then each checkpoint, then the final weights
  local method=$1 gpu=$2
  local lora_dir=/workspace/synderm/qwen_lora/$method/$SLUG/$RUN
  python "$REPO/paper_experiments/2026/qwen_lora_generate.py" --method "$method" --gpu "$gpu" --reals "$REALS" \
      --label "$LABEL" --k 10 --prompts single --out "$PILOT/$method-$SLUG-untuned-single"
  for ckpt in 500 1000 1500; do
    python "$REPO/paper_experiments/2026/qwen_lora_generate.py" --method "$method" --gpu "$gpu" --reals "$REALS" \
        --label "$LABEL" --k 10 --prompts single --lora "$lora_dir/checkpoint-$ckpt/pytorch_lora_weights.safetensors" \
        --out "$PILOT/$method-$SLUG-ckpt$ckpt-single"
  done
  python "$REPO/paper_experiments/2026/qwen_lora_generate.py" --method "$method" --gpu "$gpu" --reals "$REALS" \
      --label "$LABEL" --k 10 --prompts single --lora "$lora_dir/pytorch_lora_weights.safetensors" \
      --out "$PILOT/$method-$SLUG-ckpt2000-single"
}

generate_all edit 0 > "$LOG/pilot-gen-edit-$SLUG.log" 2>&1 &
generate_all t2i 1 > "$LOG/pilot-gen-t2i-$SLUG.log" 2>&1 &
wait

python "$REPO/paper_experiments/2026/qwen_lora_eval.py" --gpu 0 --label "$LABEL" --out "$PILOT/eval-$SLUG" --folders \
    /workspace/synderm/pilot/gpt-image-2.5-sunburst-medium-varied-seed20261028-n16 \
    "$PILOT"/{edit,t2i}-$SLUG-{untuned,ckpt500,ckpt1000,ckpt1500,ckpt2000}-single \
    > "$LOG/pilot-eval-$SLUG.log" 2>&1
echo "pilot done: $PILOT/eval-$SLUG/metrics.csv"
