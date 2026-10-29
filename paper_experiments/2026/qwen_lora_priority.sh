#!/usr/bin/env bash
# Early single-seed sweep (asked for 2026-10-29): generate the synthetic images of one sweep seed's n=16 real subset
# first, so the n in {1, 8, 16} classifier runs can start before Step 4 finishes. The real subsets are drawn exactly as
# qwen_lora_sweep.py draws them (validation 40 per label, then n per label from the rest, same random_state); n=1 and
# n=8 nest inside n=16 with pandas' seeded sampling (checked by build-csv). Images go to the same RAM folders as the
# full run, which skips them when it resumes, so the final dataset is unchanged; only the order differs.
#
#   bash paper_experiments/2026/qwen_lora_priority.sh build-csv <seed>      # writes generations/priority-<seed>.csv
#   bash paper_experiments/2026/qwen_lora_priority.sh gen-t2i  <gpu> <seed>  # 9 classes, LoRA ckpt1500, single prompt
#   bash paper_experiments/2026/qwen_lora_priority.sh gen-edit <gpu> <seed>  # untuned, varied prompts
set -euo pipefail
export HF_HOME=/dev/shm/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO=$(cd "$(dirname "$0")/../.." && pwd)
GEN="$REPO/paper_experiments/2026/qwen_lora_generate.py"
RUN=r16-lr0.0001-512px-2000steps
OUT=/dev/shm/qwen_lora_generations
BACKUP=/workspace/synderm/qwen_lora/generations
NINE=("psoriasis" "lichen planus" "basal cell carcinoma" "allergic contact dermatitis" "lupus erythematosus"
      "neutrophilic dermatoses" "photodermatoses" "sarcoidosis" "squamous cell carcinoma")

case ${1:?build-csv|gen-t2i|gen-edit} in
build-csv)
  SEED=${2:?seed}
  python - "$REPO" "$SEED" "$BACKUP/priority-$SEED.csv" <<'PY'
import sys, pandas as pd
repo, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
meta = pd.read_csv(f"{repo}/fitz_metadata/fitzpatrick17k_10label_clean_training.csv")
nine = meta["label"].value_counts().index[:9]; meta = meta[meta["label"].isin(nine)].reset_index(drop=True)
test = pd.read_csv(f"{repo}/fitz_metadata/fitzpatrick17k_10label_clean_held_out_set.csv")
pool = meta[~meta["md5hash"].isin(test["md5hash"])].reset_index(drop=True)
def sample_per_label(df, n):  # as in qwen_lora_sweep.py
    return pd.concat([df[df["label"] == lab].sample(n=n, random_state=seed) for lab in sorted(df["label"].unique())]).reset_index(drop=True)
val = sample_per_label(pool, 40); train_pool = pool[~pool["md5hash"].isin(val["md5hash"])].reset_index(drop=True)
subsets = {n: set(sample_per_label(train_pool, n)["md5hash"]) for n in (1, 8, 16)}
assert subsets[1] <= subsets[8] <= subsets[16], "n subsets do not nest; include their union instead"
sel = train_pool[train_pool["md5hash"].isin(subsets[16])][["md5hash", "label", "fitzpatrick_scale"]]
sel.to_csv(out, index=False); print(len(sel), "reals for seed", seed, "->", out)
PY
  ;;
gen-t2i)
  GPU=${2:?gpu}; SEED=${3:?seed}; CSV=$BACKUP/priority-$SEED.csv
  for label in "${NINE[@]}"; do
    slug=${label// /-}
    lora=/workspace/synderm/qwen_lora/t2i/$slug/$RUN/checkpoint-1500/pytorch_lora_weights.safetensors
    echo "priority t2i $label on GPU $GPU, $(date -u +%FT%T)"
    python "$GEN" --method t2i --gpu "$GPU" --reals "$CSV" --label "$label" --k 10 --prompts single --lora "$lora" --out "$OUT/t2i"
  done
  touch "$OUT/t2i/priority-$SEED.done"; echo "priority gen-t2i done $(date -u +%FT%T)"
  ;;
gen-edit)
  GPU=${2:?gpu}; SEED=${3:?seed}; CSV=$BACKUP/priority-$SEED.csv
  echo "priority edit on GPU $GPU, $(date -u +%FT%T)"
  python "$GEN" --method edit --gpu "$GPU" --reals "$CSV" --k 10 --prompts varied --out "$OUT/edit"
  touch "$OUT/edit/priority-$SEED.done"; echo "priority gen-edit done $(date -u +%FT%T)"
  ;;
*) echo "usage: $0 build-csv <seed> | gen-t2i <gpu> <seed> | gen-edit <gpu> <seed>" >&2; exit 1 ;;
esac
