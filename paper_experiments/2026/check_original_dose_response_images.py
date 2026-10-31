#!/usr/bin/env python3
"""Check the 2023 dose-response (Figure 5) text-to-image generations on the original machine: how many distinct
synthetic images each real image has across the two generation folders.

Run on the machine that holds the 2023 data folder (standard library only, no images decoded):

    python3 check_original_dose_response_images.py --root /n/data1/hms/dbmi/manrai/derm
    python3 check_original_dose_response_images.py --root ... --skip-size-check   # faster: no per-file stat

INPUT   <root>/generations-more/text-to-image/<label>/text-to-image/<slot>/<md5hash>.png       (slots 10-99)
        <root>/generations-lots-more/text-to-image/<label>/text-to-image/<slot>/<md5hash>.png  (slots 0-109)
        The `all` folder and non-folder entries (args.json, grid) are skipped.
OVERLAP Slot numbers 10-99 exist in both folders. A random sample of 300 (real, slot) pairs present in both is
        compared byte for byte: if most are identical the second folder is a copy and each slot number counts once;
        otherwise every (folder, slot) file counts as its own image.
OUTPUT  a per-label table (reals, min / median / max images per real, reals below 150) and
        --out (default fig5_per_real_counts.csv): md5hash, label, n_images, n_generations_more, n_generations_lots_more.
"""
import argparse, collections, hashlib, os, random, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/n/data1/hms/dbmi/manrai/derm")
ap.add_argument("--out", default="fig5_per_real_counts.csv")
ap.add_argument("--target", type=int, default=150, help="images per real the paper's largest dose needs")
ap.add_argument("--skip-size-check", action="store_true", help="do not stat every file for zero size")
a = ap.parse_args()

FOLDERS = ["generations-more", "generations-lots-more"]

# ---------------------------------------------------------------- list every generated file
files = collections.defaultdict(dict)  # md5 -> {(folder, slot): path}
label_of, empty = {}, 0
slots_seen = collections.defaultdict(set)
for folder in FOLDERS:
    base = os.path.join(a.root, folder, "text-to-image")
    for label in sorted(os.listdir(base)):
        d = os.path.join(base, label, "text-to-image")
        if label == "all" or not os.path.isdir(d):
            continue
        for slot in os.listdir(d):
            for fn in os.listdir(os.path.join(d, slot)):
                if not fn.endswith(".png"):
                    continue
                path = os.path.join(d, slot, fn)
                if not a.skip_size_check and os.path.getsize(path) == 0:
                    empty += 1
                    continue
                md5 = fn[:-4]
                files[md5][(folder, int(slot))] = path
                label_of[md5] = label
                slots_seen[folder].add(int(slot))
for folder in FOLDERS:
    s = slots_seen[folder]
    print(f"{folder}: slots {min(s)}-{max(s)} ({len(s)} folders)")
print("empty files skipped:", "not checked" if a.skip_size_check else empty)

# ---------------------------------------------------------------- overlapping slot numbers: copies or separate images?
pairs = [(v[(FOLDERS[0], s)], v[(FOLDERS[1], s)]) for v in files.values() for (f, s) in v
         if f == FOLDERS[0] and (FOLDERS[1], s) in v]
random.seed(0)
sample = random.sample(pairs, min(300, len(pairs)))
digest = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
same = sum(digest(x) == digest(y) for x, y in sample)
copies = same > len(sample) / 2
print(f"overlapping (real, slot) pairs: {len(pairs)}; identical files in a sample of {len(sample)}: {same}")
print("counting " + ("distinct slot numbers (the overlap is copies)" if copies else "every (folder, slot) file (the overlap is separate images)"))

# ---------------------------------------------------------------- images per real and per label
per_real = {md5: (len({s for (_, s) in v}) if copies else len(v)) for md5, v in files.items()}
by_label = collections.defaultdict(list)
for md5, n in per_real.items():
    by_label[label_of[md5]].append(n)
print(f"\n{'label':30s} {'reals':>5s} {'min':>5s} {'median':>7s} {'max':>5s} {'reals<' + str(a.target):>11s}")
for label, v in sorted(by_label.items()):
    print(f"{label:30s} {len(v):5d} {min(v):5d} {statistics.median(v):7.0f} {max(v):5d} {sum(x < a.target for x in v):11d}")
total_short = sum(n < a.target for n in per_real.values())
print(f"\n{len(per_real)} reals; {total_short} have fewer than {a.target} images")

with open(a.out, "w") as out:
    out.write("md5hash,label,n_images,n_generations_more,n_generations_lots_more\n")
    for md5 in sorted(per_real):
        n_f = [sum(f == folder for (f, _) in files[md5]) for folder in FOLDERS]
        out.write(f"{md5},{label_of[md5]},{per_real[md5]},{n_f[0]},{n_f[1]}\n")
print("wrote", a.out)
