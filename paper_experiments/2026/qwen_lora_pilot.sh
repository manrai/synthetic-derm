#!/usr/bin/env bash
# Step 1A of QWEN_LORA.md: pilot generation and scoring for one class.
#
#   bash paper_experiments/2026/qwen_lora_pilot.sh generate <method> <gpu> <label> <run name>
#   bash paper_experiments/2026/qwen_lora_pilot.sh eval <gpu> <label>
#
# generate: for untuned Qwen, then LoRA checkpoints 500/1000/1500, then the final 2000-step weights, make 10
#   images per real for the 16 reals of the API pilot (reals.csv of that folder), seed-only variety (--prompts single).
#   Output: /workspace/synderm/qwen_lora/pilot/<method>-<label>-<ckpt>-single/.
# eval: score all ten Qwen folders next to the GPT Image varied-prompt pilot folder with qwen_lora_eval.py.
set -euo pipefail
export HF_HOME=/dev/shm/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO=$(cd "$(dirname "$0")/../.." && pwd)
GEN="$REPO/paper_experiments/2026/qwen_lora_generate.py"
GPT_PILOT=/workspace/synderm/pilot/gpt-image-2.5-sunburst-medium-varied-seed20261028-n16
PILOT=/workspace/synderm/qwen_lora/pilot
mkdir -p "$PILOT"

case ${1:?generate|eval} in
generate)
  METHOD=${2:?method}; GPU=${3:?gpu}; LABEL=${4:?label}; RUN=${5:?run name}; SLUG=${LABEL// /-}
  LORA_DIR=/workspace/synderm/qwen_lora/$METHOD/$SLUG/$RUN
  python "$GEN" --method "$METHOD" --gpu "$GPU" --reals "$GPT_PILOT/reals.csv" --label "$LABEL" --k 10 \
      --prompts single --out "$PILOT/$METHOD-$SLUG-untuned-single"
  for ckpt in 500 1000 1500; do
    python "$GEN" --method "$METHOD" --gpu "$GPU" --reals "$GPT_PILOT/reals.csv" --label "$LABEL" --k 10 \
        --prompts single --lora "$LORA_DIR/checkpoint-$ckpt/pytorch_lora_weights.safetensors" \
        --out "$PILOT/$METHOD-$SLUG-ckpt$ckpt-single"
  done
  python "$GEN" --method "$METHOD" --gpu "$GPU" --reals "$GPT_PILOT/reals.csv" --label "$LABEL" --k 10 \
      --prompts single --lora "$LORA_DIR/pytorch_lora_weights.safetensors" --out "$PILOT/$METHOD-$SLUG-ckpt2000-single"
  echo "generate done: $METHOD $LABEL"
  ;;
eval)
  GPU=${2:?gpu}; LABEL=${3:?label}; SLUG=${LABEL// /-}
  python "$REPO/paper_experiments/2026/qwen_lora_eval.py" --gpu "$GPU" --label "$LABEL" --out "$PILOT/eval-$SLUG" \
      --folders "$GPT_PILOT" "$PILOT"/{edit,t2i}-$SLUG-{untuned,ckpt500,ckpt1000,ckpt1500,ckpt2000}-single
  echo "eval done: $PILOT/eval-$SLUG/metrics.csv"
  ;;
*) echo "usage: $0 generate <method> <gpu> <label> <run> | eval <gpu> <label>" >&2; exit 1 ;;
esac
