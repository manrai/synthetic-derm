#!/usr/bin/env python3
"""Figure 3 proprietary arms: 10 reference-image edits for every training-pool image, written in the sweep's layout.

Self-contained; read top to bottom: configuration, reals, import of the pilot images, job list, API call,
main pass, top-up of refused slots, synthetic.csv. Derived from pilot_generate.py (same API calls, prompts,
refusal and back-off handling); differences are the input (a reals CSV covering the whole pool) and the
output layout, which is the one qwen_lora_sweep.py reads:  <out>/<kk>/<md5hash>.png + .json, kk = 00..09.

    python paper_experiments/2026/api_generate_pool.py --model nano-banana-2.1 --budget 1200 --threads 20 --dry-run
    python paper_experiments/2026/api_generate_pool.py --model nano-banana-2.1 --budget 1200 --threads 20

PROMPTS  slot k uses EDIT_FRAME with EDIT_VARIATIONS[k]: the ten body-site / framing / lighting variations of the
         pilot (the same list for every class and reference). The models expose no seed.
REALS    --reals CSV with md5hash and label (default: generations/pool9.csv, the 3,339 training-pool images of the
         nine Figure 3 classes); images from --images-dir/<md5hash>.jpg.
PILOT    before any API call, the pilot folder's images (--pilot-dir, 10 edits for 144 reals of seed 20261028) are
         copied into the layout, so they are not paid for twice. Their JSON sidecars are copied too.
REFUSALS a refused request is retried once. A slot refused twice is filled by an edit of a *different* reference of
         the same class (one that had all its own slots accepted), with the refused slot's prompt, and saved under the
         refused real's path; its JSON has topup=true and ref_md5 = the reference actually used. Same convention as
         the pilot's synthetic.csv, so every (real, slot) file the sweep asks for exists.
BUDGET   --budget is a hard cap in USD for this process: no new request is submitted once the recorded cost reaches
         it (cost from the provider's returned token counts). Nothing in the script raises it.
STOP     20 consecutive failed requests (quota, auth, outage) stop the run with a message instead of spinning.
OUTPUT   <out>/<kk>/<md5hash>.png + .json, <out>/synthetic.csv (md5hash, label, slot, location, ref_md5, topup),
         <out>/progress.json, <out>/generate.log. Re-running resumes: finished slots are skipped.
"""
import argparse, base64, io, json, os, random, shutil, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[2]

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

MODELS = {
    "gpt-image-2.5-sunburst": dict(provider="openai", model_id="gpt-image-2.5-sunburst", default_quality="medium",
                                   pilot_dir="/workspace/synderm/pilot/gpt-image-2.5-sunburst-medium-varied-seed20261028-n16"),
    "nano-banana-2.1": dict(provider="google", model_id="gemini-nano-banana-2.1", default_quality="1K", usd_per_image_token=30 / 1e6,
                            pilot_dir="/workspace/synderm/pilot/nano-banana-2.1-1K-varied-seed20261028-n16"),
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
    u = r.usage
    in_img, in_txt = u.input_tokens_details.image_tokens or 0, u.input_tokens_details.text_tokens or 0
    out_img = (u.output_tokens_details.image_tokens if u.output_tokens_details else None) or u.output_tokens
    cost = in_img * 8 / 1e6 + in_txt * 5 / 1e6 + out_img * 30 / 1e6
    return image, dict(input_image_tokens=in_img, input_text_tokens=in_txt, output_image_tokens=out_img, cost_usd=round(cost, 5))


def google_edit(model_id, image_size, prompt, ref_path, usd_per_image_token):
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
                       cost_usd=round(img_tokens * usd_per_image_token, 5))


def main():
    # ---------------------------------------------------------------- configuration
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--quality", default=None)
    ap.add_argument("--budget", type=float, required=True, help="USD hard cap for this process")
    ap.add_argument("--reals", default="/workspace/synderm/qwen_lora/generations/pool9.csv")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--images-dir", default="/root/data/fitzpatrick17k/finalfitz17k")
    ap.add_argument("--out", default=None, help="default /workspace/synderm/api_generations/<model>")
    ap.add_argument("--pilot-dir", default=None, help="pilot folder whose images are imported first; '' to skip")
    ap.add_argument("--labels", nargs="*", default=None, help="restrict to these labels (tests)")
    ap.add_argument("--max-reals", type=int, default=None, help="first N reals only (tests)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    load_env(REPO / ".env")
    m = MODELS[a.model]
    quality = a.quality or m["default_quality"]
    out = Path(a.out or f"/workspace/synderm/api_generations/{a.model}")
    out.mkdir(parents=True, exist_ok=True)
    for k in range(a.k):
        (out / f"{k:02d}").mkdir(exist_ok=True)
    log = open(out / "generate.log", "a")

    def say(*s):
        msg = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in s)
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    reals = pd.read_csv(a.reals)[["md5hash", "label"]]
    if a.labels:
        reals = reals[reals["label"].isin(a.labels)]
    if a.max_reals:
        reals = reals.head(a.max_reals)
    reals = reals.reset_index(drop=True)
    label_of = dict(zip(reals["md5hash"], reals["label"]))
    say(f"model {a.model} ({m['model_id']}, {quality}), {len(reals)} reals x {a.k} slots from {a.reals}, budget ${a.budget:.0f}, out {out}")

    def png_path(md5, slot):
        return out / f"{slot:02d}" / f"{md5}.png"

    # ---------------------------------------------------------------- import the pilot's images (no API calls)
    pilot_dir = m["pilot_dir"] if a.pilot_dir is None else a.pilot_dir
    if pilot_dir and Path(pilot_dir, "synthetic.csv").exists():
        syn = pd.read_csv(Path(pilot_dir) / "synthetic.csv")
        syn = syn[(syn["slot"] < a.k) & syn["md5hash"].isin(label_of)]
        n_copied = 0
        for r in syn.itertuples():
            dst = png_path(r.md5hash, int(r.slot))
            if dst.exists():
                continue
            src = Path(r.location)
            rec = json.loads(src.with_suffix(".json").read_text())
            rec.update(md5hash=r.md5hash, slot=int(r.slot), topup=bool(r.topup), ref_md5=r.source_md5,
                       imported_from=str(src), image=str(dst))
            shutil.copyfile(src, dst); dst.with_suffix(".json").write_text(json.dumps(rec, indent=1)); n_copied += 1
        say(f"pilot import from {pilot_dir}: {n_copied} images copied, {len(syn) - n_copied} were already present")

    # ---------------------------------------------------------------- job list (one directory listing per slot)
    have_png = {k: {p[:-4] for p in os.listdir(out / f"{k:02d}") if p.endswith(".png")} for k in range(a.k)}
    have_json = {k: {p[:-5] for p in os.listdir(out / f"{k:02d}") if p.endswith(".json")} for k in range(a.k)}

    def record(md5, slot):
        p = png_path(md5, slot).with_suffix(".json")
        return json.loads(p.read_text()) if p.exists() else None

    def is_done(md5, slot):
        if md5 in have_png[slot]:
            return True
        if md5 in have_json[slot]:
            rec = record(md5, slot)
            return rec is not None and rec.get("status") == "refused" and rec.get("refusals", 0) >= 2
        return False

    jobs = [(r.md5hash, r.label, k) for r in reals.itertuples() for k in range(a.k)]
    todo = [j for j in jobs if not is_done(j[0], j[2])]
    est = len(todo) * (0.0334 if m["provider"] == "google" else 0.0178)
    say(f"main pass: {len(jobs)} slots, {len(jobs) - len(todo)} done, {len(todo)} to run (about ${est:,.0f} at the pilot's rate)")
    if a.dry_run:
        say("dry run: no requests made"); return

    # ---------------------------------------------------------------- shared state
    lock = threading.Lock()
    totals = dict(ok=0, refused=0, error=0, cost_usd=0.0, rate_limits=0, consecutive_failures=0)
    throttle = dict(delay=0.0, ok_streak=0)
    stop = threading.Event()
    stop_reason = []

    def generate(md5, label, slot, ref_md5=None):
        """One slot. ref_md5: reference image to edit (another real of the class for top-ups); default the real itself."""
        ref_md5 = ref_md5 or md5
        prompt = EDIT_FRAME.format(condition=label, variation=EDIT_VARIATIONS[slot % len(EDIT_VARIATIONS)])
        ref = f"{a.images_dir}/{ref_md5}.jpg"
        png = png_path(md5, slot)
        prev = record(md5, slot) or {}
        attempts, refusals = int(prev.get("attempts", 0)), int(prev.get("refusals", 0))
        rec = dict(md5hash=md5, label=label, slot=slot, model=m["model_id"], quality=quality, prompt_mode="varied",
                   variation=slot % len(EDIT_VARIATIONS), prompt=prompt, ref_md5=ref_md5, topup=ref_md5 != md5)
        if ref_md5 != md5:
            rec["refused_error"] = prev.get("error")   # why the real's own edit was refused
        for _ in range(8):
            if stop.is_set():
                rec["status"] = "skipped"; break
            if png.exists():
                rec["status"] = "exists"; break
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
                image.save(png)
                rec.update(status="ok", latency_s=round(time.time() - t, 1), **info)
                with lock:
                    totals["consecutive_failures"] = 0
                    throttle["ok_streak"] += 1
                    if throttle["ok_streak"] >= 50 and throttle["delay"]:
                        throttle["delay"] /= 2; throttle["ok_streak"] = 0
                break
            except Refused as e:
                refusals += 1; rec["error"] = str(e)
                with lock:
                    totals["consecutive_failures"] = 0
                if refusals >= 2:
                    rec["status"] = "refused"; break
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
        if rec.get("status") == "exists":
            return
        rec.setdefault("status", "error")
        rec.update(attempts=attempts, refusals=refusals, image=str(png) if rec["status"] == "ok" else None)
        png.with_suffix(".json").write_text(json.dumps(rec, indent=1))
        with lock:
            if rec["status"] in totals: totals[rec["status"]] += 1
            totals["cost_usd"] += float(rec.get("cost_usd") or 0)
            if rec["status"] == "error":
                totals["consecutive_failures"] += 1
                if totals["consecutive_failures"] >= 20 and not stop.is_set():
                    stop.set(); stop_reason.append(f"20 consecutive failed requests, last: {rec['error']}")
                    say("STOPPED:", stop_reason[-1])
            if totals["cost_usd"] >= a.budget and not stop.is_set():
                stop.set(); stop_reason.append(f"budget cap ${a.budget:.0f} reached at ${totals['cost_usd']:.2f}")
                say("STOPPED:", stop_reason[-1])
            done = totals["ok"] + totals["refused"] + totals["error"]
            if done % 100 == 0:
                say(f"progress ok={totals['ok']} refused={totals['refused']} error={totals['error']} "
                    f"429s={totals['rate_limits']} cost=${totals['cost_usd']:.2f}")
            (out / "progress.json").write_text(json.dumps({**totals, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))

    # ---------------------------------------------------------------- main pass
    say(f"main pass start: {len(todo)} requests, {a.threads} threads")
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for f in as_completed([ex.submit(generate, *j) for j in todo]):
            f.result()
    say(f"main pass end: ok={totals['ok']} refused={totals['refused']} error={totals['error']} cost=${totals['cost_usd']:.2f}")

    # ---------------------------------------------------------------- top-up refused slots with another reference of the class
    if not stop.is_set():
        recs = [(md5, label, k, record(md5, k)) for md5, label, k in jobs if md5 not in have_png[k] and not png_path(md5, k).exists()]
        deficits = [(md5, label, k) for md5, label, k, rec in recs if rec and rec.get("status") == "refused"]
        say(f"top-up: {len(deficits)} refused slots")
        if deficits:
            accepted = {}
            for label in reals["label"].unique():
                refs = [md5 for md5, lab in label_of.items() if lab == label]
                accepted[label] = [r for r in refs if all(png_path(r, k).exists() and not (record(r, k) or {}).get("topup") for k in range(a.k))]
                say(f"  {label}: {len(accepted[label])} fully accepted references")
            rng = random.Random(0)
            extra = [(md5, label, k, rng.choice([r for r in accepted[label] if r != md5])) for md5, label, k in deficits if accepted[label]]
            with ThreadPoolExecutor(max_workers=a.threads) as ex:
                for f in as_completed([ex.submit(generate, *j) for j in extra]):
                    f.result()
            still = [j for j in deficits if not png_path(j[0], j[2]).exists()]
            say(f"top-up end: {len(deficits) - len(still)} filled, {len(still)} still missing, cost so far ${totals['cost_usd']:.2f}")

    # ---------------------------------------------------------------- synthetic.csv
    rows = []
    for md5, label, k in jobs:
        p = png_path(md5, k)
        if p.exists():
            rec = record(md5, k) or {}
            rows.append(dict(md5hash=md5, label=label, slot=k, location=str(p), ref_md5=rec.get("ref_md5", md5), topup=bool(rec.get("topup", False))))
    pd.DataFrame(rows).to_csv(out / "synthetic.csv", index=False)
    missing = len(jobs) - len(rows)
    say(f"DONE: {len(rows)} of {len(jobs)} slots have images ({missing} missing); ok={totals['ok']} refused={totals['refused']} "
        f"error={totals['error']} cost this run=${totals['cost_usd']:.2f}" + (f"; stopped: {stop_reason[-1]}" if stop_reason else ""))


if __name__ == "__main__":
    main()
