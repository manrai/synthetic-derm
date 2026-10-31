#!/usr/bin/env python3
"""Nano Banana 2.1 reference-image edits through the Gemini Batch API (half price), in the same layout and JSON
sidecar format as api_generate_pool.py. Written for Figure 5 (Step 6 of QWEN_LORA.md): slots 10-149 for the reals of
seed 119108; slots 0-9 already exist from Figure 3.

    python paper_experiments/2026/api_generate_batch.py --reals /workspace/synderm/fig5/reals-119108-n16.csv \\
        --slots 10 149 --budget 380 --dry-run
    python paper_experiments/2026/api_generate_batch.py --reals ... --slots 10 149 --budget 380

PROMPTS  EDIT_FRAME + EDIT_VARIATIONS[slot] from api_generate_pool.py (the one shared definition of the 150 prompts).
REQUESTS one JSONL line per (real, slot): the prompt and the reference JPEG inline, 1K, 1:1. Lines are grouped into
         batch jobs of --chunk requests (a JSONL file under 2 GB; results ~1.4 MB per image).
STATE    <out>/batch_state.json lists every submitted job, its request keys and whether its results were written, so a
         restart polls jobs already submitted instead of paying for them again. A (real, slot) is done when its PNG
         exists.
ROUNDS   1: every missing slot, the real as reference. 2: slots refused once, again with the real. 3-5: slots refused
         twice, each round with another real of the same class whose own edit for that slot was accepted (up to 3
         references, a fixed random choice seeded by the (real, slot) key); topup=true and ref_md5 in the sidecar. Request errors (not refusals)
         are resubmitted in the next round, at most 3 times.
BUDGET   --budget: hard cap in USD for everything recorded in batch_state.json (all rounds, all restarts). A job is only
         submitted if recorded spend + the worst case of the jobs still running + its own worst case (requests x
         MAX_USD_PER_REQUEST) stays under the cap. Cost per image from the returned usage at batch rates: input
         $0.75/M tokens, output image $15/M tokens.
OUTPUT   <out>/<kk>/<md5hash>.png + .json (status, prompt, ref_md5, topup, tokens, cost_usd, batch_job),
         <out>/synthetic_<reals-file-stem>.csv for these reals and slots 0..max, <out>/batch.log.
"""
import argparse, base64, io, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_generate_pool import EDIT_FRAME, EDIT_VARIATIONS, load_env, REPO  # the shared prompt list

MODEL_ID = "gemini-nano-banana-2.1"
USD_IN, USD_OUT = 0.75 / 1e6, 15 / 1e6           # batch rates (half the standard $1.50 / $30 per 1M tokens)
MAX_USD_PER_REQUEST = 1400 * USD_IN + 1120 * USD_OUT  # 1,180 input and 1,120 output tokens measured; margin on input
DONE_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

ap = argparse.ArgumentParser()
ap.add_argument("--reals", required=True, help="CSV with md5hash, label")
ap.add_argument("--slots", type=int, nargs=2, default=[10, 149], help="first and last slot, inclusive")
ap.add_argument("--budget", type=float, required=True, help="USD hard cap over all batch jobs in batch_state.json")
ap.add_argument("--chunk", type=int, default=4500, help="requests per batch job")
ap.add_argument("--images-dir", default="/root/data/fitzpatrick17k/finalfitz17k")
ap.add_argument("--out", default="/workspace/synderm/api_generations/nano-banana-2.1")
ap.add_argument("--stage", default="/root/stage/batch", help="local folder for request files")
ap.add_argument("--poll", type=int, default=60)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
assert a.slots[1] < len(EDIT_VARIATIONS), "not enough prompts"

load_env(REPO / ".env")
from google import genai
client = genai.Client()
out, stage = Path(a.out), Path(a.stage)
stage.mkdir(parents=True, exist_ok=True)
for k in range(a.slots[0], a.slots[1] + 1):
    (out / f"{k:02d}").mkdir(parents=True, exist_ok=True)
log = open(out / "batch.log", "a")
state_path = out / "batch_state.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": []}

def say(*s):
    msg = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in s)
    print(msg, flush=True); log.write(msg + "\n"); log.flush()

def save_state():
    tmp = state_path.with_suffix(".tmp"); tmp.write_text(json.dumps(state, indent=1)); os.replace(tmp, state_path)

png_path = lambda md5, slot: out / f"{slot:02d}" / f"{md5}.png"
def record(md5, slot):
    p = png_path(md5, slot).with_suffix(".json")
    return json.loads(p.read_text()) if p.exists() else {}

def parallel(fn, items):
    # the volume answers one stat in ~30 ms; 64 at a time turns 80k checks from ~50 min into ~1 min
    with ThreadPoolExecutor(64) as ex:
        return list(ex.map(fn, items))

def still_missing(pairs):
    return [ms for ms, e in zip(pairs, parallel(lambda ms: png_path(*ms).exists(), pairs)) if not e]

reals = pd.read_csv(a.reals)[["md5hash", "label"]]
label_of = dict(zip(reals["md5hash"], reals["label"]))
slots = list(range(a.slots[0], a.slots[1] + 1))
spent = lambda: sum(j.get("cost_usd", 0.0) for j in state["jobs"])
say(f"{len(reals)} reals x slots {a.slots[0]}-{a.slots[1]} from {a.reals}; budget ${a.budget:.0f}; recorded so far ${spent():.2f}")

# ---------------------------------------------------------------- one request line
_jpeg_b64 = {}
def request_line(md5, slot, ref, rnd):
    if ref not in _jpeg_b64:  # each reference is encoded once (140 requests share it)
        buf = io.BytesIO(); Image.open(f"{a.images_dir}/{ref}.jpg").convert("RGB").save(buf, format="JPEG")
        _jpeg_b64[ref] = base64.b64encode(buf.getvalue()).decode()
    prompt = EDIT_FRAME.format(condition=label_of[md5], variation=EDIT_VARIATIONS[slot])
    req = {"contents": [{"role": "user", "parts": [{"text": prompt},
           {"inline_data": {"mime_type": "image/jpeg", "data": _jpeg_b64[ref]}}]}],
           "generation_config": {"response_modalities": ["IMAGE"], "image_config": {"aspect_ratio": "1:1", "image_size": "1K"}}}
    return json.dumps({"key": f"{md5}|{slot}|{ref}|{rnd}", "request": req})

# ---------------------------------------------------------------- write the results of one finished job
def collect(job_entry):
    job = client.batches.get(name=job_entry["name"])
    job_entry["state"] = job.state.name
    if job.state.name != "JOB_STATE_SUCCEEDED":
        say(f"job {job_entry['name']} ended {job.state.name}: {job.error}; its requests go to the next round")
        job_entry["collected"] = True; save_state(); return
    url = f"https://generativelanguage.googleapis.com/download/v1beta/{job.dest.file_name}:download?alt=media"
    n_ok = n_ref = n_err = 0; cost = 0.0
    with requests.get(url, headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, stream=True, timeout=3600) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            e = json.loads(line)
            md5, slot, ref, rnd = e["key"].split("|"); slot, rnd = int(slot), int(rnd)
            prev = record(md5, slot)
            rec = dict(md5hash=md5, label=label_of[md5], slot=slot, model=MODEL_ID, quality="1K", prompt_mode="varied",
                       variation=slot, prompt=EDIT_FRAME.format(condition=label_of[md5], variation=EDIT_VARIATIONS[slot]),
                       ref_md5=ref, topup=ref != md5, mode="batch", batch_job=job_entry["name"], round=rnd,
                       attempts=int(prev.get("attempts", 0)) + 1, refusals=int(prev.get("refusals", 0)),
                       errors=int(prev.get("errors", 0)))
            if ref != md5:
                rec["refused_error"] = prev.get("refused_error") or prev.get("error")
            res = e.get("response")
            if res is None:                                   # request-level error (status object)
                rec.update(status="error", error=json.dumps(e.get("error") or e.get("status") or e)[:300])
                rec["errors"] += 1; n_err += 1
            else:
                u = res.get("usageMetadata", {})
                in_tok = u.get("promptTokenCount", 0) or 0
                out_tok = sum(d.get("tokenCount", 0) for d in u.get("candidatesTokensDetails", []) if d.get("modality") == "IMAGE")
                c = round(in_tok * USD_IN + out_tok * USD_OUT, 6); cost += c
                parts = [p for cand in res.get("candidates", []) for p in (cand.get("content") or {}).get("parts", [])]
                img = next((p["inlineData"] for p in parts if "inlineData" in p), None)
                rec.update(input_tokens=in_tok, output_image_tokens=out_tok, cost_usd=c)
                if img:
                    png = png_path(md5, slot); tmp = png.with_suffix(".tmp.png")
                    Image.open(io.BytesIO(base64.b64decode(img["data"]))).save(tmp); os.replace(tmp, png)
                    rec.update(status="ok", image=str(png)); n_ok += 1
                else:                                         # blocked input or output: a refusal
                    why = res.get("promptFeedback") or [{k: v for k, v in cd.items() if k != "content"} for cd in res.get("candidates", [])]
                    rec.update(status="refused", error=json.dumps(why)[:300], image=None)
                    rec["refusals"] += 1; n_ref += 1
            png_path(md5, slot).with_suffix(".json").write_text(json.dumps(rec, indent=1))
    job_entry.update(collected=True, cost_usd=round(cost, 4), ok=n_ok, refused=n_ref, error=n_err)
    save_state()
    say(f"job {job_entry['name']} written: ok={n_ok} refused={n_ref} error={n_err} cost=${cost:.2f}; total recorded ${spent():.2f}")

# ---------------------------------------------------------------- submit one round, wait, collect
def run_round(rnd, todo):
    """todo: list of (md5, slot, ref). Skips keys already in a submitted, uncollected job."""
    pending = {k for j in state["jobs"] if not j.get("collected") for k in j["keys"]}
    todo = [t for t in todo if f"{t[0]}|{t[1]}|{t[2]}|{rnd}" not in pending]
    for i in range(0, len(todo), a.chunk):
        part = todo[i:i + a.chunk]
        outstanding = sum(len(j["keys"]) for j in state["jobs"] if not j.get("collected")) * MAX_USD_PER_REQUEST
        worst = len(part) * MAX_USD_PER_REQUEST
        if spent() + outstanding + worst > a.budget:
            say(f"STOPPED: budget ${a.budget:.0f}: recorded ${spent():.2f} + running ${outstanding:.2f} + this job ${worst:.2f} would exceed it")
            return False
        f = stage / f"round{rnd}_{i // a.chunk:03d}.jsonl"
        keys = []
        with open(f, "w") as fh:
            for md5, slot, ref in part:
                fh.write(request_line(md5, slot, ref, rnd) + "\n"); keys.append(f"{md5}|{slot}|{ref}|{rnd}")
        up = client.files.upload(file=str(f), config={"display_name": f.stem, "mime_type": "jsonl"})
        for attempt in range(30):  # a full queue (enqueued-token cap) is retried for up to ~1 h
            try:
                job = client.batches.create(model=MODEL_ID, src=up.name, config={"display_name": f"fig5-{f.stem}"}); break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                    say(f"queue full ({str(e)[:120]}); retry in 2 min"); wait_and_collect(once=True); time.sleep(120)
                else:
                    raise
        else:
            say("STOPPED: could not submit after 30 tries"); return False
        state["jobs"].append(dict(name=job.name, round=rnd, file=up.name, keys=keys, submitted=time.strftime("%F %T"),
                                  worst_usd=round(worst, 2)))
        save_state(); f.unlink()
        say(f"round {rnd}: submitted {job.name} with {len(part)} requests (worst case ${worst:.2f})")
    wait_and_collect()
    return True

def wait_and_collect(once=False):
    while True:
        open_jobs = [j for j in state["jobs"] if not j.get("collected")]
        if not open_jobs:
            return
        for j in open_jobs:
            st = client.batches.get(name=j["name"]).state.name
            if st in DONE_STATES:
                collect(j)
                try:
                    client.files.delete(name=j["file"])
                except Exception:
                    pass
            elif st != j.get("state"):
                j["state"] = st; save_state(); say(f"job {j['name']}: {st}")
        if once:
            return
        if any(not j.get("collected") for j in state["jobs"]):
            time.sleep(a.poll)

# ---------------------------------------------------------------- dry run and resume
wait_and_collect(once=True)  # collect jobs submitted before a restart that have finished; the rest are awaited later
missing = still_missing([(m, s) for m in reals["md5hash"] for s in slots])
say(f"{len(reals) * len(slots)} slots, {len(reals) * len(slots) - len(missing)} done, {len(missing)} missing; "
    f"worst case ${len(missing) * MAX_USD_PER_REQUEST:.2f}, expected ~${len(missing) * (1180 * USD_IN + 1120 * USD_OUT):.2f}")
if a.dry_run:
    say("dry run: nothing submitted"); sys.exit(0)

# ---------------------------------------------------------------- rounds
ok = True
for rnd in (1, 2):  # own reference: first try, then one retry for refusals and errors
    left = still_missing(missing)
    recs = parallel(lambda ms: record(*ms), left)
    todo = [(m, s, m) for (m, s), r in zip(left, recs) if r.get("refusals", 0) < 2 and r.get("errors", 0) < 3]
    if not todo or not ok:
        break
    say(f"round {rnd}: {len(todo)} requests with the real as reference")
    ok = run_round(rnd, todo)
for rnd, attempt in ((3, 0), (4, 1), (5, 2)):  # same-slot top-ups from other reals of the class
    deficits = still_missing(missing)
    if not deficits or not ok:
        break
    # a reference qualifies for slot s if its own edit for s exists and is not itself a top-up; top-ups are the
    # submitted keys whose reference differs from the real (slots 10-149 have no top-ups from any other script)
    have = set((m, s) for m, s in missing) - set(deficits)
    have |= {(m, s) for m in reals["md5hash"] for s in slots} - set(missing)
    topups = {(k.split("|")[0], int(k.split("|")[1])) for j in state["jobs"] for k in j["keys"] if k.split("|")[0] != k.split("|")[2]}
    todo = []
    for m, s in deficits:
        cands = sorted(r for r, lab in label_of.items() if lab == label_of[m] and r != m and (r, s) in have
                       and (r, s) not in topups)
        picks = random.Random(f"{m}|{s}").sample(cands, min(3, len(cands)))
        if len(picks) > attempt:
            todo.append((m, s, picks[attempt]))
    say(f"round {rnd}: {len(todo)} top-ups (reference {attempt + 1} of up to 3)")
    if todo:
        ok = run_round(rnd, todo)

# ---------------------------------------------------------------- synthetic CSV for these reals, slots 0..last
rows, still = [], []
pairs = [(m, s) for m in reals["md5hash"] for s in range(0, a.slots[1] + 1)]
for (m, s), rec in zip(pairs, parallel(lambda ms: record(*ms) if png_path(*ms).exists() else None, pairs)):
    if rec is None:
        still.append((m, s))
    else:
        rows.append(dict(md5hash=m, label=label_of[m], slot=s, location=str(png_path(m, s)), ref_md5=rec.get("ref_md5", m),
                         topup=bool(rec.get("topup", False))))
csv = out / f"synthetic_{Path(a.reals).stem}.csv"
pd.DataFrame(rows).to_csv(csv, index=False)
say(f"DONE: {len(rows)} of {len(reals) * (a.slots[1] + 1)} (real, slot) files exist, {len(still)} missing {still[:5]}; "
    f"batch spend recorded ${spent():.2f} of ${a.budget:.0f}; wrote {csv}")
