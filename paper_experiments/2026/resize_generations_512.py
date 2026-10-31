#!/usr/bin/env python3
# 512 px lossless PNG copies of the Nano Banana 1K generations for the classifier sweep (originals untouched)
import os, time, pandas as pd
from multiprocessing import Pool
from PIL import Image
SRC = "/workspace/synderm/api_generations/nano-banana-2.1"; DST = "/root/data/api512/nano-banana-2.1"
rows = pd.read_csv(f"{SRC}/synthetic.csv")
jobs = [(f"{SRC}/{k:02d}/{m}.png", f"{DST}/{k:02d}/{m}.png") for m, k in zip(rows.md5hash, rows.slot)]
for k in range(10): os.makedirs(f"{DST}/{k:02d}", exist_ok=True)
def one(j):
    src, dst = j
    if os.path.exists(dst): return 0
    for attempt in range(3):  # the volume sometimes returns a short read; retry, then report
        try:
            Image.open(src).convert("RGB").resize((512, 512), Image.LANCZOS).save(dst + ".tmp.png"); os.rename(dst + ".tmp.png", dst); return 1
        except OSError as e:
            err = e; time.sleep(2)
    print("FAILED", src, err, flush=True); return 0
with Pool(48) as p:
    n = 0
    for i, r in enumerate(p.imap_unordered(one, jobs, chunksize=16)):
        n += r
        if i % 2000 == 0: print(i, "of", len(jobs), flush=True)
missing = [d for _, d in jobs if not os.path.exists(d)]
print("DONE", len(jobs) - len(missing), "of", len(jobs), "present;", len(missing), "missing", flush=True)
