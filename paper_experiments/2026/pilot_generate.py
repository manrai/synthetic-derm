#!/usr/bin/env python3
"""Pilot generation (plan Step 1): reference-image edits for the paper's n-per-label sweep.

Self-contained. Read top to bottom: configuration, data split, job list, API calls, main pass,
top-up for refusals, synthetic.csv. Only three small functions (env loading, one API call per
provider); everything else is linear code in main().

    python pilot_generate.py --model gpt-image-2.5-sunburst --prompts varied --threads 20 --budget 60
    python pilot_generate.py --model nano-banana-2.1 --prompts varied --threads 20 --budget 80
    python pilot_generate.py --prompts varied --dry-run   # write the split CSVs only, no API calls

PROMPT MODES (--prompts)
  single  every slot of a reference uses the one EDIT_PROMPT; diversity comes from sampling alone
          (the models expose no seed, and the screen showed the 10 repeats look alike)
  varied  slot k uses EDIT_FRAME with EDIT_VARIATIONS[k % 10]: ten body-site / framing / lighting
          variations, the same list for every class and reference, so the variation is orthogonal
          to the label (screen notebook section 6b; chosen for the pilot on 2026-10-28)
  The two modes write to different output folders (varied adds "-varied" to the name).

DATA SPLIT (reproduces paper_experiments/Experiments/n_per_label.py for one seed)
  training pool  fitz_metadata/fitzpatrick17k_10label_clean_training.csv, nine most common labels
  held-out test  fitz_metadata/fitzpatrick17k_10label_clean_held_out_set.csv, same labels  -> test.csv  (360)
  validation     40 per label drawn from the pool with the seed                            -> val.csv   (360)
  reals          n per label drawn from pool minus validation with the seed               -> reals.csv (9n)
                 columns in_n1, in_n8, in_n16 mark the nested subsets the sweep uses
  Real images are read from --images-dir/<md5hash>.jpg.

OUTPUT (--out, default /workspace/synderm/pilot/<model>-<quality>[-varied]-seed<seed>-n<n>/)
  <label>/edit/<kk>/<md5hash>.png + .json   one edit per (real, slot kk); JSON holds status, cost, prompt
  synthetic.csv                             every (real, slot) -> image path, including top-ups
  progress.json, generate.log               running totals and the log

REFUSALS  a refused request is retried once; a slot refused twice is a deficit. Deficits are filled
with extra edits (slots 10, 11, ...) of fully accepted references in the same class, so every
class ends with n x K synthetic images and the real subset is unchanged. Re-running resumes.
"""
import argparse, base64, io, json, os, random, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[2]

# Edit instruction (same text as synderm/backends/prompts.py EDIT_TEMPLATE, copied so the script stands alone).
EDIT_PROMPT = ("A clinical photograph of the same skin disease, {condition}, on a different patient who has the "
               "same skin tone, at a different body site, keeping the characteristic appearance of the lesions. "
               "No text, watermarks or annotations.")

# Varied mode (same text as screen notebook section 6b): a shared frame plus one variation per slot.
EDIT_FRAME = ("A clinical photograph of the same skin disease, {condition}, keeping the characteristic appearance "
              "of the lesions and the same skin tone as the reference, on a different patient. "
              "No text, watermarks or annotations. {variation}")
EDIT_VARIATIONS = [
    "Close-up of a single lesion on the forearm, exam-room lighting.",
    "Several lesions across the upper back, photographed from about one metre.",
    "Lesions on the cheek and jawline of an older adult, soft daylight from a window.",
    "Lesions on the lower leg and shin, straight-on view, camera flash.",
    "Lesions on the back of the hand and fingers, close-up, slightly oblique angle.",
    "Widespread involvement of the chest and abdomen of a young adult, mid-range view.",
    "Lesions on the neck and behind the ear, close-up, overhead clinic lighting.",
    "Lesions on the thigh, photographed on an exam table with a paper sheet in the background.",
    "Lesions on the scalp margin and forehead, close-up, hair partly visible.",
    "Lesions on the foot and ankle, mid-range view, natural outdoor light.",
]

# Model table. OpenAI cost is computed from returned tokens; Google from image output tokens x rate.
MODELS = {
    "gpt-image-2.5-sunburst": dict(provider="openai", model_id="gpt-image-2.5-sunburst", default_quality="medium"),
    "gpt-image-2": dict(provider="openai", model_id="gpt-image-2", default_quality="medium"),
    "nano-banana-2.1": dict(provider="google", model_id="gemini-nano-banana-2.1", default_quality="1K", usd_per_image_token=30 / 1e6),
    "nano-banana-pro": dict(provider="google", model_id="gemini-3-pro-image", default_quality="1K", usd_per_image_token=120 / 1e6),
}


def load_env(path):
    """Export KEY=value lines from the gitignored .env (shell variables take precedence)."""
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Refused(Exception):
    """The provider declined the request on safety grounds."""


def openai_edit(model_id, quality, prompt, ref_path):
    """One reference-image edit. Returns (PIL image, info dict with token counts and cost_usd)."""
    import openai
    client = openai.OpenAI()
    try:
        with open(ref_path, "rb") as f:
            r = client.images.edit(model=model_id, image=[f], prompt=prompt, n=1, size="1024x1024", quality=quality, output_format="png")
    except openai.BadRequestError as e:
        if "safety" in str(e).lower() or "moderation" in str(e).lower():
            raise Refused(str(e)[:300]) from e
        raise
    image = Image.open(io.BytesIO(base64.b64decode(r.data[0].b64_json)))
    u = r.usage  # token counts only; priced per the OpenAI pricing page (2026-10-28)
    in_img, in_txt = u.input_tokens_details.image_tokens or 0, u.input_tokens_details.text_tokens or 0
    out_img = (u.output_tokens_details.image_tokens if u.output_tokens_details else None) or u.output_tokens
    out_txt = (u.output_tokens_details.text_tokens if u.output_tokens_details else 0) or 0
    cost = in_img * 8 / 1e6 + in_txt * 5 / 1e6 + out_img * 30 / 1e6  # text output not priced for the gpt-image-2 family
    return image, dict(input_image_tokens=in_img, input_text_tokens=in_txt, output_image_tokens=out_img,
                       output_text_tokens=out_txt, cost_usd=round(cost, 5))


def google_edit(model_id, image_size, prompt, ref_path, usd_per_image_token):
    """One reference-image edit through the Gemini Interactions API (JPEG output only)."""
    from google import genai
    client = genai.Client()  # reads GEMINI_API_KEY
    buf = io.BytesIO(); Image.open(ref_path).convert("RGB").save(buf, format="JPEG")
    try:
        r = client.interactions.create(
            model=model_id,
            input=[{"type": "text", "text": prompt},
                   {"type": "image", "data": base64.b64encode(buf.getvalue()).decode(), "mime_type": "image/jpeg"}],
            response_format={"type": "image", "mime_type": "image/jpeg", "aspect_ratio": "1:1", "image_size": image_size})
    except Exception as e:
        if "blocked" in str(e).lower() or "violates" in str(e).lower():
            raise Refused(str(e)[:300]) from e
        raise
    if r.output_image is None or not r.output_image.data:
        raise Refused(f"{r.status}: {(r.output_text or 'no image returned')[:300]}")
    image = Image.open(io.BytesIO(base64.b64decode(r.output_image.data)))
    u = r.usage
    img_tokens = sum(t.tokens for t in (u.output_tokens_by_modality or []) if t.modality == "image") if u else 0
    return image, dict(input_tokens=getattr(u, "total_input_tokens", None), output_image_tokens=img_tokens,
                       thought_tokens=getattr(u, "total_thought_tokens", None), cost_usd=round(img_tokens * usd_per_image_token, 5))


def main():
    # ---------------------------------------------------------------- configuration
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-image-2.5-sunburst", choices=sorted(MODELS))
    ap.add_argument("--quality", default=None, help="OpenAI quality (low/medium/high) or Google image size (1K/2K); default per model")
    ap.add_argument("--prompts", default="single", choices=["single", "varied"], help="one prompt per reference, or ten variations (slot k -> variation k)")
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--n", type=int, default=16, help="real images per label in the largest subset")
    ap.add_argument("--k", type=int, default=10, help="synthetic images per real")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--budget", type=float, default=60.0, help="USD; stop submitting new requests above this")
    ap.add_argument("--images-dir", default="/root/data/fitzpatrick17k/finalfitz17k")
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels", nargs="*", default=None, help="restrict to these labels (small tests)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    load_env(REPO / ".env")
    m = MODELS[a.model]
    quality = a.quality or m["default_quality"]
    out = Path(a.out or f"/workspace/synderm/pilot/{a.model}-{quality}{'-varied' if a.prompts == 'varied' else ''}-seed{a.seed}-n{a.n}")
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "generate.log", "a")

    def say(*s):
        msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in s)
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    # ---------------------------------------------------------------- data split (the paper's sampling)
    meta = pd.read_csv(REPO / "fitz_metadata/fitzpatrick17k_10label_clean_training.csv")
    top9 = meta["label"].value_counts().index[:9]                      # drops folliculitis
    meta = meta[meta["label"].isin(top9)].reset_index(drop=True)
    test = pd.read_csv(REPO / "fitz_metadata/fitzpatrick17k_10label_clean_held_out_set.csv")
    test = test[test["label"].isin(top9)].reset_index(drop=True)
    pool = meta[~meta["md5hash"].isin(test["md5hash"])]                 # training pool, 3,339 images
    # per-label sample: same rows as the paper's groupby().apply(lambda x: x.sample(...)), written as a
    # concat so pandas 3 keeps the label column
    val = pd.concat([g.sample(n=40, random_state=a.seed) for _, g in pool.groupby("label")]).reset_index(drop=True)
    train_pool = pool[~pool["md5hash"].isin(val["md5hash"])]
    subset_sizes = sorted({1, 8, 16, 32, 64, 128, 228} & set(range(1, a.n + 1)) | {a.n})
    subsets = {s: pd.concat([g.sample(n=s, random_state=a.seed) for _, g in train_pool.groupby("label")]).reset_index(drop=True)
               for s in subset_sizes}
    reals = subsets[a.n][["md5hash", "label", "fitzpatrick_scale", "url"]].copy()
    for s, df in subsets.items():
        reals[f"in_n{s}"] = reals["md5hash"].isin(df["md5hash"])
        assert set(df["md5hash"]) <= set(reals["md5hash"]), f"subset {s} is not nested in {a.n}"
    if a.labels:
        reals = reals[reals["label"].isin(a.labels)]
    reals.to_csv(out / "reals.csv", index=False); val.to_csv(out / "val.csv", index=False); test.to_csv(out / "test.csv", index=False)
    say(f"model {a.model} ({m['model_id']}, {quality}) seed {a.seed} prompts={a.prompts}")
    say(f"split: pool {len(pool)} -> val {len(val)} ({out / 'val.csv'}) | reals {len(reals)} = {a.n} per label "
        f"({out / 'reals.csv'}, nested subsets {subset_sizes}) | test {len(test)} ({out / 'test.csv'})")
    say(f"real images from {a.images_dir}; outputs under {out}")

    # ---------------------------------------------------------------- job list: one request per (real, slot)
    def png_path(label, md5, slot):
        return out / label.replace(" ", "-") / "edit" / f"{slot:02d}" / f"{md5}.png"

    def record(label, md5, slot):
        p = png_path(label, md5, slot).with_suffix(".json")
        return json.loads(p.read_text()) if p.exists() else None

    def is_done(rec):
        return rec is not None and (rec["status"] == "ok" or (rec["status"] == "refused" and rec.get("refusals", 0) >= 2))

    jobs = [(r.md5hash, r.label, k) for r in reals.itertuples() for k in range(a.k)]
    todo = [j for j in jobs if not is_done(record(j[1], j[0], j[2]))]
    say(f"main pass: {len(jobs)} slots, {len(todo)} to run, {len(jobs) - len(todo)} already done")
    if a.dry_run:
        say("dry run: split CSVs written, no requests made"); return

    # ---------------------------------------------------------------- shared state for the workers
    lock = threading.Lock()
    totals = dict(ok=0, refused=0, error=0, cost_usd=0.0, rate_limits=0)
    throttle = dict(delay=0.0, ok_streak=0)   # 429 -> double the pre-call delay; 50 successes -> halve it
    stop = threading.Event()                   # set when the budget cap is hit

    def generate(md5, label, slot):
        """Worker: one slot. Retries refusals once, backs off on 429, records a JSON sidecar."""
        if a.prompts == "varied":
            variation = slot % len(EDIT_VARIATIONS)        # top-up slots 10, 11, ... reuse the list
            prompt = EDIT_FRAME.format(condition=label, variation=EDIT_VARIATIONS[variation])
        else:
            variation = None
            prompt = EDIT_PROMPT.format(condition=label)
        ref = f"{a.images_dir}/{md5}.jpg"
        png = png_path(label, md5, slot)
        prev = record(label, md5, slot) or {}
        attempts, refusals = int(prev.get("attempts", 0)), int(prev.get("refusals", 0))
        rec = dict(md5hash=md5, label=label, slot=slot, model=m["model_id"], quality=quality, prompt_mode=a.prompts,
                   variation=variation, prompt=prompt, ref=ref)
        for _ in range(8):
            if stop.is_set():
                rec["status"] = "skipped_budget"; break
            with lock:
                delay = throttle["delay"]
            if delay:
                time.sleep(delay * random.uniform(0.5, 1.5))
            t = time.time(); attempts += 1
            try:
                if m["provider"] == "openai":
                    image, info = openai_edit(m["model_id"], quality, prompt, ref)
                else:
                    image, info = google_edit(m["model_id"], quality, prompt, ref, m["usd_per_image_token"])
                png.parent.mkdir(parents=True, exist_ok=True); image.save(png)
                rec.update(status="ok", latency_s=round(time.time() - t, 1), **info)
                with lock:
                    throttle["ok_streak"] += 1
                    if throttle["ok_streak"] >= 50 and throttle["delay"]:
                        throttle["delay"] /= 2; throttle["ok_streak"] = 0
                break
            except Refused as e:
                refusals += 1; rec["error"] = str(e)
                if refusals >= 2:
                    rec["status"] = "refused"; break      # second refusal: give up on this slot
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:300]}"
                if "429" in msg or "RateLimit" in type(e).__name__ or "RESOURCE_EXHAUSTED" in msg:
                    with lock:
                        throttle["delay"] = min(max(1.0, throttle["delay"] * 2), 30.0); throttle["ok_streak"] = 0
                        totals["rate_limits"] += 1; d = throttle["delay"]
                    say(f"429 on {label}/{md5[:8]}/{slot}: delay now {d:.0f}s"); time.sleep(d)
                else:
                    rec.update(status="error", error=msg)
                    if attempts >= 3: break
                    time.sleep(2 ** attempts)
        rec.setdefault("status", "error")
        rec.update(attempts=attempts, refusals=refusals, image=str(png) if rec["status"] == "ok" else None)
        png.parent.mkdir(parents=True, exist_ok=True)
        png.with_suffix(".json").write_text(json.dumps(rec, indent=1))
        with lock:
            if rec["status"] in totals: totals[rec["status"]] += 1
            totals["cost_usd"] += float(rec.get("cost_usd") or 0)
            done = totals["ok"] + totals["refused"] + totals["error"]
            if totals["cost_usd"] >= a.budget and not stop.is_set():
                stop.set(); say(f"BUDGET CAP ${a.budget:.0f} reached at ${totals['cost_usd']:.2f}: no new requests")
            if done % 25 == 0:
                say(f"progress ok={totals['ok']} refused={totals['refused']} error={totals['error']} "
                    f"429s={totals['rate_limits']} cost=${totals['cost_usd']:.2f}")
            (out / "progress.json").write_text(json.dumps({**totals, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))

    # ---------------------------------------------------------------- main pass
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for f in as_completed([ex.submit(generate, *j) for j in todo]):
            f.result()

    # ---------------------------------------------------------------- top-up refused slots, write synthetic.csv
    recs = pd.DataFrame([record(label, md5, k) for md5, label, k in jobs])
    rows = []
    for label, g in recs.groupby("label"):
        ok_per_ref = g[g.status == "ok"].groupby("md5hash").size()
        accepted = [r for r in ok_per_ref.index if ok_per_ref[r] == a.k]   # refs with all K slots ok
        deficit = g[g.status != "ok"]
        say(f"{label}: {int((g.status == 'ok').sum())}/{len(g)} ok, {len(deficit)} deficit slots, {len(accepted)} fully accepted refs")
        for r in g[g.status == "ok"].itertuples():
            rows.append(dict(md5hash=r.md5hash, label=label, slot=r.slot, location=r.image, source_md5=r.md5hash, topup=False))
        if len(deficit) == 0:
            continue
        if not accepted:
            say(f"!! {label}: no fully accepted reference to top up from"); continue
        # spread the deficit round-robin over accepted refs, using slots K, K+1, ...
        extra = [(accepted[i % len(accepted)], label, a.k + i // len(accepted)) for i in range(len(deficit))]
        extra_todo = [j for j in extra if not is_done(record(j[1], j[0], j[2]))]
        say(f"top-up {label}: {len(extra_todo)} requests")
        with ThreadPoolExecutor(max_workers=a.threads) as ex:
            for f in as_completed([ex.submit(generate, *j) for j in extra_todo]):
                f.result()
        for (src, _, k), d in zip(extra, deficit.itertuples()):
            st = record(label, src, k) or {}
            if st.get("status") == "ok":
                rows.append(dict(md5hash=d.md5hash, label=label, slot=d.slot, location=st["image"], source_md5=src, topup=True))
            else:
                say(f"!! top-up failed for {label}/{d.md5hash[:8]} slot {d.slot}: {st.get('status')}")
    syn = pd.DataFrame(rows)
    syn.to_csv(out / "synthetic.csv", index=False)
    say(f"DONE: {len(syn)} synthetic rows ({int(syn.topup.sum()) if len(syn) else 0} top-ups) for {len(reals)} reals; "
        f"ok={totals['ok']} refused={totals['refused']} error={totals['error']} cost this run=${totals['cost_usd']:.2f}")


if __name__ == "__main__":
    main()
