"""Zero-shot screen: generate a few images per disease class from proprietary API models and
local open models, cache everything to disk, and build per-class comparison grids.

Used by ``00_zero_shot_screen.ipynb``. Every generation is written as
``<out_dir>/<model>/<label>/<mode>-<prompt_key>-<i>.png`` with a sidecar JSON recording the
prompt, status (ok / refused / error / no_key), latency, token usage and estimated cost.
Re-running skips finished and refused jobs and retries errors.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image, ImageDraw

from synderm.backends import prompts as P

# ----------------------------------------------------------------------------------------
# Classes and prompts
# ----------------------------------------------------------------------------------------

#: The nine Figure 3 classes (F17k-9), hyphenated as in the repo.
CLASSES = [
    "allergic-contact-dermatitis",
    "basal-cell-carcinoma",
    "lichen-planus",
    "lupus-erythematosus",
    "neutrophilic-dermatoses",
    "photodermatoses",
    "psoriasis",
    "sarcoidosis",
    "squamous-cell-carcinoma",
]

PROMPT_TEMPLATES = {
    "paper": P.PAPER_TEMPLATE,        # "An image of {condition}, a skin disease"
    "clinical": P.CLINICAL_TEMPLATE,  # fuller scene description
}
EDIT_TEMPLATE = P.EDIT_TEMPLATE       # same condition, different patient, same skin tone


# ----------------------------------------------------------------------------------------
# Model table
# ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    name: str              # row label and output folder
    provider: str          # "openai" | "google" | "local"
    model_id: str
    price_per_image: float # USD, list price at ~1K output; used only for the estimate column
    quality: Optional[str] = None  # OpenAI quality parameter
    notes: str = ""


MODELS: dict[str, ModelSpec] = {m.name: m for m in [
    # OpenAI. Output is priced per token ($30/M for gpt-image-2 and 2.5, $32/M for 1.5); the
    # per-image figure below is a planning estimate at medium quality, 1024x1024.
    ModelSpec("gpt-image-2.5-sunburst", "openai", "gpt-image-2.5-sunburst", 0.07, "medium"),
    ModelSpec("gpt-image-2.5-flare", "openai", "gpt-image-2.5-flare", 0.07, "medium"),
    ModelSpec("gpt-image-2", "openai", "gpt-image-2", 0.05, "medium"),
    ModelSpec("gpt-image-1.5", "openai", "gpt-image-1.5", 0.05, "medium", notes="deprecated by OpenAI"),
    # Google. Per-image list prices at 1K from ai.google.dev/gemini-api/docs/pricing (2026-10-28).
    ModelSpec("nano-banana-2.1", "google", "gemini-nano-banana-2.1", 0.0336),
    ModelSpec("nano-banana-pro", "google", "gemini-3-pro-image", 0.134),
    ModelSpec("nano-banana-2", "google", "gemini-3.1-flash-image", 0.067),
    # Local open-weight models through synderm backends, no LoRA.
    ModelSpec("qwen-image-2.1", "local", "qwen-image-2.1", 0.0, notes="768 px, 40 steps, 5090"),
    ModelSpec("flux-3-klein", "local", "flux-3-klein", 0.0, notes="optional; cached on /workspace"),
    ModelSpec("sdxl", "local", "sdxl", 0.0, notes="optional; cached on /workspace"),
]}


# ----------------------------------------------------------------------------------------
# Real images
# ----------------------------------------------------------------------------------------

def real_examples(images_dir: Path, metadata_csv: Path, labels=CLASSES, n_show=4, n_ref=2, seed=0):
    """Fixed real examples per class: ``show`` images for the grid and ``ref`` images for edits."""
    from synderm.data import normalize_label
    df = pd.read_csv(metadata_csv)
    df["label"] = df["label"].map(normalize_label)
    out = {}
    for lab in labels:
        rows = df[df["label"] == lab].sample(n=n_show + n_ref, random_state=seed)
        paths = [images_dir / f"{h}.jpg" for h in rows["md5hash"]]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} real images missing for {lab}, e.g. {missing[0]}")
        out[lab] = {"show": paths[:n_show], "ref": paths[n_show:]}
    return out


# ----------------------------------------------------------------------------------------
# Jobs and cache
# ----------------------------------------------------------------------------------------

@dataclass
class Job:
    model: str
    label: str
    mode: str          # "t2i" | "edit"
    prompt_key: str    # "paper" | "clinical" | "edit"
    i: int             # repeat index; also the seed for local models
    prompt: str
    ref_path: Optional[str] = None

    @property
    def stem(self) -> str:
        return f"{self.mode}-{self.prompt_key}-{self.i}"


def make_jobs(model: str, real: dict, n_t2i_per_prompt=2, n_edit=2, labels=CLASSES) -> list[Job]:
    jobs = []
    for lab in labels:
        for key, tmpl in PROMPT_TEMPLATES.items():
            for i in range(n_t2i_per_prompt):
                jobs.append(Job(model, lab, "t2i", key, i, P.caption(lab, tmpl)))
        for i in range(n_edit):
            jobs.append(Job(model, lab, "edit", "edit", i, P.caption(lab, EDIT_TEMPLATE), str(real[lab]["ref"][i])))
    return jobs


class Cache:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)

    def png(self, job: Job) -> Path:
        return self.out_dir / job.model / job.label / f"{job.stem}.png"

    def meta(self, job: Job) -> Path:
        return self.png(job).with_suffix(".json")

    def status(self, job: Job) -> Optional[str]:
        m = self.meta(job)
        if not m.exists():
            return None
        return json.loads(m.read_text()).get("status")

    def write(self, job: Job, image: Optional[Image.Image], record: dict) -> None:
        p = self.png(job)
        p.parent.mkdir(parents=True, exist_ok=True)
        if image is not None:
            image.save(p)
        self.meta(job).write_text(json.dumps({**asdict(job), **record}, indent=1))

    def records(self) -> pd.DataFrame:
        rows = [json.loads(p.read_text()) for p in self.out_dir.rglob("*.json")]
        return pd.DataFrame(rows) if rows else pd.DataFrame()


# ----------------------------------------------------------------------------------------
# Providers
# ----------------------------------------------------------------------------------------

def _png_bytes(path: str) -> bytes:
    with Image.open(path) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()


class Refused(Exception):
    """The provider declined to generate (content moderation)."""


def openai_generate(job: Job, spec: ModelSpec, size="1024x1024") -> tuple[Image.Image, dict]:
    import openai
    client = openai.OpenAI()
    kwargs = dict(model=spec.model_id, prompt=job.prompt, n=1, size=size, output_format="png")
    if spec.quality:
        kwargs["quality"] = spec.quality
    try:
        if job.mode == "t2i":
            # moderation="low" is the least restrictive filter; clinical images trip "auto".
            result = client.images.generate(moderation="low", **kwargs)
        else:
            with open(job.ref_path, "rb") as f:
                result = client.images.edit(image=[f], **kwargs)
    except openai.BadRequestError as e:
        body = getattr(e, "body", None) or {}
        err = body.get("error", body) if isinstance(body, dict) else {}
        if "moderation" in str(err.get("code", "")) or "safety" in str(e).lower() or "moderation" in str(e).lower():
            raise Refused(str(e)[:500]) from e
        raise
    image = Image.open(io.BytesIO(base64.b64decode(result.data[0].b64_json)))
    usage = getattr(result, "usage", None)
    info = {}
    if usage is not None:
        # The endpoint returns token counts only, no price and no separate reasoning-token field.
        # Priced per the OpenAI pricing page (2026-10-28): image input $8/M, text input $5/M,
        # image output $30/M ($32/M for gpt-image-1.5); text output is not listed for the
        # gpt-image-2 family so it is recorded but priced at zero. Verify against the OpenAI
        # usage dashboard after the first few hundred images.
        inp = getattr(usage, "input_tokens_details", None); outp = getattr(usage, "output_tokens_details", None)
        in_img = getattr(inp, "image_tokens", 0) or 0; in_txt = getattr(inp, "text_tokens", 0) or 0
        out_img = getattr(outp, "image_tokens", None); out_txt = getattr(outp, "text_tokens", 0) or 0
        if out_img is None:  # older response shape: no breakdown, treat all output as image tokens
            out_img = getattr(usage, "output_tokens", 0) or 0
        rate_out = 32.0 if spec.model_id == "gpt-image-1.5" else 30.0
        info.update(input_tokens=getattr(usage, "input_tokens", None), output_tokens=getattr(usage, "output_tokens", None),
                    input_image_tokens=in_img, input_text_tokens=in_txt, output_image_tokens=out_img, output_text_tokens=out_txt)
        info["cost_usd"] = round(in_img * 8.0 / 1e6 + in_txt * 5.0 / 1e6 + out_img * rate_out / 1e6, 4)
    return image, info


def google_generate(job: Job, spec: ModelSpec, image_size="1K") -> tuple[Image.Image, dict]:
    from google import genai
    client = genai.Client()  # reads GEMINI_API_KEY or GOOGLE_API_KEY
    if job.mode == "t2i":
        inputs = job.prompt
    else:
        inputs = [
            {"type": "text", "text": job.prompt},
            {"type": "image", "data": base64.b64encode(_png_bytes(job.ref_path)).decode(), "mime_type": "image/png"},
        ]
    try:
        interaction = client.interactions.create(
            model=spec.model_id,
            input=inputs,
            response_format={"type": "image", "mime_type": "image/jpeg", "aspect_ratio": "1:1", "image_size": image_size},  # API serves JPEG only (checked 2026-10-28)
        )
    except Exception as e:
        # Google reports a blocked reference image as HTTP 400 "Input blocked: ... violates ... policy".
        if "blocked" in str(e).lower() or "violates" in str(e).lower():
            raise Refused(str(e)[:500]) from e
        raise
    out = getattr(interaction, "output_image", None)  # ImageContent(data=<base64>, mime_type=...)
    if out is None or not getattr(out, "data", None):
        # No image in the response: treat as a refusal and keep whatever text came back.
        text = str(getattr(interaction, "output_text", "") or "")
        raise Refused(f"{getattr(interaction, 'status', '')}: {text[:500] or 'no image returned'}")
    image = Image.open(io.BytesIO(base64.b64decode(out.data)))
    info = {"cost_usd": spec.price_per_image}
    usage = getattr(interaction, "usage", None)
    if usage is not None:
        # Image output is billed per token: $30/M (nano-banana-2.1, lite), $60/M (nano-banana-2),
        # $120/M (nano-banana-pro). A 1K image is 1120 tokens. Thought tokens are text output.
        rate = {"gemini-nano-banana-2.1": 30.0, "gemini-3.1-flash-lite-image": 30.0,
                "gemini-3.1-flash-image": 60.0, "gemini-3-pro-image": 120.0}.get(spec.model_id)
        img_tokens = sum(m.tokens for m in (getattr(usage, "output_tokens_by_modality", None) or []) if m.modality == "image")
        info.update(input_tokens=getattr(usage, "total_input_tokens", None), output_tokens=getattr(usage, "total_output_tokens", None),
                    image_output_tokens=img_tokens, thought_tokens=getattr(usage, "total_thought_tokens", None))
        if rate and img_tokens:
            info["cost_usd"] = round(img_tokens * rate / 1e6, 4)
    return image, info


def has_key(provider: str) -> bool:
    if provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if provider == "google":
        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    return True


def _run_one(job: Job, spec: ModelSpec, fn: Callable, cache: Cache) -> dict:
    t = time.time()
    try:
        image, info = fn(job, spec)
        rec = {"status": "ok", "latency_s": round(time.time() - t, 1), **info}
        cache.write(job, image, rec)
    except Refused as e:
        rec = {"status": "refused", "latency_s": round(time.time() - t, 1), "error": str(e)}
        cache.write(job, None, rec)
    except Exception as e:  # recorded, retried on the next run
        rec = {"status": "error", "latency_s": round(time.time() - t, 1),
               "error": f"{type(e).__name__}: {str(e)[:500]}", "trace": traceback.format_exc()[-1500:]}
        cache.write(job, None, rec)
    return {**asdict(job), **rec}


def run_api_model(spec: ModelSpec, jobs: list[Job], cache: Cache, max_workers=4, verbose=True) -> list[dict]:
    """Run all jobs for one API model with a small thread pool. Skips ok/refused jobs."""
    fn = {"openai": openai_generate, "google": google_generate}[spec.provider]
    if not has_key(spec.provider):
        print(f"!! {spec.name}: no API key for {spec.provider} in the environment; skipping {len(jobs)} jobs")
        for j in jobs:
            if cache.status(j) is None:
                cache.write(j, None, {"status": "no_key"})
        return []
    todo = [j for j in jobs if cache.status(j) not in ("ok", "refused")]
    if verbose:
        print(f"{spec.name}: {len(todo)} to run, {len(jobs) - len(todo)} cached")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one, j, spec, fn, cache): j for j in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            r = fut.result(); results.append(r)
            if verbose and (r["status"] != "ok" or k % 10 == 0):
                print(f"  [{k}/{len(todo)}] {r['label']} {r['mode']} {r['status']} {r.get('error', '')[:120]}")
    return results


def run_local_model(spec: ModelSpec, jobs: list[Job], cache: Cache, resolution=768, verbose=True) -> list[dict]:
    """Run all jobs for one synderm backend (no LoRA), sequentially on the GPU."""
    from synderm.backends import get_backend
    todo = [j for j in jobs if cache.status(j) != "ok"]
    if verbose:
        print(f"{spec.name}: {len(todo)} to run, {len(jobs) - len(todo)} cached")
    if not todo:
        return []
    gen = get_backend(spec.model_id)
    t = time.time(); gen.load(); load_s = time.time() - t
    results = []
    try:
        for k, job in enumerate(todo, 1):
            t = time.time()
            try:
                if job.mode == "t2i":
                    im = gen.text_to_image(job.prompt, seed=job.i, height=resolution, width=resolution)[0]
                else:
                    ref = Image.open(job.ref_path).convert("RGB")
                    im = gen.edit(job.prompt, ref, seed=job.i, height=resolution, width=resolution)[0]
                rec = {"status": "ok", "latency_s": round(time.time() - t, 1), "load_s": round(load_s), "cost_usd": 0.0}
                cache.write(job, im, rec)
            except Exception as e:
                rec = {"status": "error", "latency_s": round(time.time() - t, 1),
                       "error": f"{type(e).__name__}: {str(e)[:500]}", "trace": traceback.format_exc()[-1500:]}
                cache.write(job, None, rec)
            results.append({**asdict(job), **rec})
            if verbose and (rec["status"] != "ok" or k % 6 == 0):
                print(f"  [{k}/{len(todo)}] {job.label} {job.mode} {rec['status']} {rec['latency_s']}s {rec.get('error', '')[:120]}")
    finally:
        gen.unload()
    return results


# ----------------------------------------------------------------------------------------
# Grids and summary
# ----------------------------------------------------------------------------------------

def _placeholder(text: str, size: int) -> Image.Image:
    im = Image.new("RGB", (size, size), (235, 235, 235))
    d = ImageDraw.Draw(im)
    d.text((8, size // 2 - 6), text, fill=(90, 90, 90))
    return im


def _thumb(path: Path, size: int) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size; s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return im.resize((size, size), Image.LANCZOS)


def class_grid(label: str, models: list[str], real: dict, cache: Cache, thumb=256, n_t2i_per_prompt=2, n_edit=2):
    """One grid per class. Row 0: real examples then the edit references. One row per model:
    t2i (paper prompt), t2i (clinical prompt), edits aligned under their reference image."""
    n_cols = len(real[label]["show"]) + n_edit
    rows, row_titles = [], []
    rows.append([_thumb(p, thumb) for p in real[label]["show"]] + [_thumb(p, thumb) for p in real[label]["ref"][:n_edit]])
    row_titles.append("real")
    for m in models:
        row = []
        for key in PROMPT_TEMPLATES:
            for i in range(n_t2i_per_prompt):
                job = Job(m, label, "t2i", key, i, "")
                st = cache.status(job)
                row.append(_thumb(cache.png(job), thumb) if st == "ok" else _placeholder(st or "missing", thumb))
        for i in range(n_edit):
            job = Job(m, label, "edit", "edit", i, "")
            st = cache.status(job)
            row.append(_thumb(cache.png(job), thumb) if st == "ok" else _placeholder(st or "missing", thumb))
        rows.append(row[:n_cols]); row_titles.append(m)
    col_titles = [f"real {i+1}" for i in range(len(real[label]["show"]))] + [f"edit ref {i+1}" for i in range(n_edit)]
    col_titles_models = (["t2i paper"] * n_t2i_per_prompt + ["t2i clinical"] * n_t2i_per_prompt + ["edit"] * n_edit)[:n_cols]
    return rows, row_titles, col_titles, col_titles_models


def show_grid(rows, row_titles, col_titles, col_titles_models=None, size=2.2, suptitle=None, save: Optional[Path] = None):
    import matplotlib.pyplot as plt
    n_r, n_c = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(n_r, n_c, figsize=(size * n_c, size * n_r + 0.6), squeeze=False)
    for r, row in enumerate(rows):
        for c in range(n_c):
            ax = axes[r][c]; ax.axis("off")
            if c < len(row):
                ax.imshow(row[c])
            if r == 0 and c < len(col_titles):
                ax.set_title(col_titles[c], fontsize=9)
            elif r == 1 and col_titles_models and c < len(col_titles_models):
                ax.set_title(col_titles_models[c], fontsize=8, color="gray")
            if c == 0:
                ax.text(-0.04, 0.5, row_titles[r], transform=ax.transAxes, ha="right", va="center", fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=110, bbox_inches="tight")
    return fig


def summarize(cache: Cache) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model summary and a model x class table of refusals/errors."""
    df = cache.records()
    if df.empty:
        return df, df
    df["cost_usd"] = df.get("cost_usd", 0.0)
    summary = df.groupby("model").agg(
        n=("status", "size"),
        ok=("status", lambda s: (s == "ok").sum()),
        refused=("status", lambda s: (s == "refused").sum()),
        error=("status", lambda s: (s == "error").sum()),
        no_key=("status", lambda s: (s == "no_key").sum()),
        mean_latency_s=("latency_s", "mean"),
        est_cost_usd=("cost_usd", "sum"),
    ).round(2)
    problems = df[df["status"].isin(["refused", "error"])]
    by_class = problems.pivot_table(index="model", columns="label", values="status", aggfunc="size", fill_value=0) if not problems.empty else pd.DataFrame()
    return summary, by_class
