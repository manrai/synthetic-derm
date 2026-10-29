#!/usr/bin/env bash
# Mirror the RAM generation folders to the /workspace volume, 20 streams at a time (one per method/slot folder).
# Replaces `qwen_lora_full_run.sh sync` for the Step 4 run: a single rsync stream wrote ~1 file per 0.8 s on the
# geesefs volume, about the rate the two generation arms produce files, so one pass took an hour and the mirror
# never caught up. Parallel streams scale on this mount (the 24-way image copy on 2026-10-28 ran 24x faster).
#
#   bash paper_experiments/2026/qwen_lora_sync.sh [minutes between passes, default 20]
set -uo pipefail
OUT=/dev/shm/qwen_lora_generations
BACKUP=/workspace/synderm/qwen_lora/generations
every=${1:-20}
while true; do
  t0=$(date +%s)
  for method in edit t2i; do
    for slot in 00 01 02 03 04 05 06 07 08 09; do
      [ -d "$OUT/$method/$slot" ] || continue
      mkdir -p "$BACKUP/$method/$slot"
      rsync -r --size-only --no-owner --no-group "$OUT/$method/$slot/" "$BACKUP/$method/$slot/" &
    done
  done
  wait
  echo "synced $(date -u +%FT%T) in $(( $(date +%s) - t0 )) s: $(find "$OUT" -name '*.png' | wc -l) png in RAM"
  sleep $((every * 60))
done
