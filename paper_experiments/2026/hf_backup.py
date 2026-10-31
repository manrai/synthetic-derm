"""Private Hugging Face backup of the 2026 generated images and LoRA adapters (one-off, resumable).

Image folders go up as one uncompressed .tar per slot folder (layout and file names kept inside); CSVs and logs as
plain files; LoRA folders as plain files. Real Fitzpatrick photos are never uploaded: every `train-data/` folder and
`vignettes/06_reader_study/images` are excluded. Each tar is built on local disk, uploaded, checked against the
remote sha256 (or size when the hub gives none), then deleted. A unit whose remote copy already matches is skipped.
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

for l in open("/root/synthetic-derm/.env").read().splitlines():
    if "=" in l and not l.startswith("#"):
        k, v = l.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

api = HfApi(token=os.environ["HF_TOKEN"])
W = Path("/workspace/synderm")
STAGE = Path("/root/stage"); STAGE.mkdir(exist_ok=True)
LOG = Path("/workspace/logs/pod-2026-10-31/hf-backup-manifest.jsonl")
EXCL = ["--exclude=train-data", "--exclude=06_reader_study/images", "--exclude=*.tmp.png"]
NB = "tbuckley/synderm-2026-nano-banana-2.1"
QW = "tbuckley/synderm-2026-qwen-image-2.1"
PI = "tbuckley/synderm-2026-pilots"
LO = "tbuckley/synderm-2026-qwen-image-2.1-loras"

def say(*s):
    print(time.strftime("%H:%M:%S"), *s, flush=True)

def repo(rid, rtype):
    api.create_repo(rid, repo_type=rtype, private=True, exist_ok=True)
    info = api.repo_info(rid, repo_type=rtype)
    if not info.private:
        sys.exit(f"STOP: {rid} exists and is not private")

def remote(rid, rtype, path):
    try:
        f = api.get_paths_info(rid, [path], repo_type=rtype, expand=True)
    except Exception:
        return None
    return f[0] if f else None

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

def matches(r, size, sha):
    if r is None or getattr(r, "size", None) != size:
        return False
    lfs = getattr(r, "lfs", None)
    rsha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
    return rsha is None or rsha == sha

done = {}
if LOG.exists():
    for line in LOG.read_text().splitlines():
        e = json.loads(line); done[(e["repo"], e["path"])] = e

def tar_unit(rid, rtype, src_parent, name, path_in_repo):
    """tar src_parent/name -> path_in_repo; skip if already uploaded and verified."""
    if (rid, path_in_repo) in done:
        say("skip (verified before)", rid, path_in_repo); return
    out = STAGE / Path(path_in_repo).name
    # expected content: every file under the folder except the excluded ones
    src_files = [p for p in (Path(src_parent) / name).rglob("*") if p.is_file() and "train-data" not in p.parts
                 and not ("06_reader_study" in p.parts and "images" in p.parts) and not p.name.endswith(".tmp.png")]
    want_n, want_bytes = len(src_files), sum(p.stat().st_size for p in src_files)
    for attempt in range(3):
        # exit 1 with "file changed as we read it" is a timestamp quirk of the geesefs volume; accepted only if the
        # archive then holds exactly the expected number of files and bytes. Anything else is retried.
        r = subprocess.run(["tar", "-cf", str(out), "--warning=no-file-changed", *EXCL, "-C", str(src_parent), name],
                           capture_output=True, text=True)
        listing = subprocess.run(["tar", "-tvf", str(out)], capture_output=True, text=True).stdout.splitlines()
        members = [l.split() for l in listing if not l.startswith("d")]
        got_n, got_bytes = len(members), sum(int(m[2]) for m in members)
        if r.returncode in (0, 1) and (got_n, got_bytes) == (want_n, want_bytes):
            break
        say(f"tar retry {attempt + 1} for {name}: rc={r.returncode} files {got_n}/{want_n} bytes {got_bytes}/{want_bytes} {r.stderr.strip()[:200]}")
        time.sleep(5)
    else:
        sys.exit(f"STOP: tar failed 3 times for {src_parent}/{name}")
    n = got_n
    size, sha = out.stat().st_size, sha256(out)
    say(f"upload {rid}/{path_in_repo}: {n} files, {size / 1e9:.2f} GB")
    api.upload_file(path_or_fileobj=str(out), path_in_repo=path_in_repo, repo_id=rid, repo_type=rtype,
                    commit_message=f"backup {path_in_repo}")
    if not matches(remote(rid, rtype, path_in_repo), size, sha):
        sys.exit(f"STOP: remote copy of {path_in_repo} does not match (size {size}, sha {sha})")
    e = dict(repo=rid, path=path_in_repo, src=f"{src_parent}/{name}", files=n, bytes=size, sha256=sha, time=time.strftime("%F %T"))
    with open(LOG, "a") as f:
        f.write(json.dumps(e) + "\n")
    done[(rid, path_in_repo)] = e
    out.unlink()

def plain(rid, rtype, src, path_in_repo):
    api.upload_file(path_or_fileobj=str(src), path_in_repo=path_in_repo, repo_id=rid, repo_type=rtype, commit_message=f"backup {path_in_repo}")

def readme(rid, rtype, text):
    api.upload_file(path_or_fileobj=text.encode(), path_in_repo="README.md", repo_id=rid, repo_type=rtype, commit_message="README")

COMMON = ("Private backup, not a release. Part of the 2026 synthetic-derm experiments (branch `modernize`, "
          "`paper_experiments/2026/QWEN_LORA.md`). Image folders are stored as uncompressed tars, one per slot folder; "
          "extract with `tar -xf <file>.tar`, which recreates the original folder. `MANIFEST.jsonl` lists each tar's "
          "file count and sha256. No real Fitzpatrick 17k photos are included.\n")

# ---------------------------------------------------------------- pilots, screens, smoke tests, vignette outputs
repo(PI, "dataset")
for parent, name, dst in [(W, "zero_shot_screen_smoke", "zero_shot_screen_smoke.tar"), (W / "qwen_lora", "pilot", "qwen_lora-pilot.tar"),
                          (W, "zero_shot_screen", "zero_shot_screen.tar"), (W, "pilot", "pilot.tar"),
                          (W, "generations", "stage-a-generations.tar"), (W, "smoke", "smoke.tar"),
                          (W / "qwen_lora", "smoke", "qwen_lora-smoke.tar"), (W / "qwen_lora", "sweep-smoke", "qwen_lora-sweep-smoke.tar"),
                          (W, "vignettes", "vignettes.tar")]:
    tar_unit(PI, "dataset", parent, name, dst)
readme(PI, "dataset", "# synderm 2026: pilots, screens and smoke tests\n\n" + COMMON +
       "\nOne tar per folder of `/workspace/synderm`: `pilot` (API pilots, GPT Image 2.5 and Nano Banana 2.1, n=16), "
       "`qwen_lora-pilot` (Qwen LoRA pilot folders and eval grids; the grids show real images next to generated ones), "
       "`zero_shot_screen(_smoke)`, `stage-a-generations` (Oct 21-22 package tests), `smoke` and `qwen_lora-smoke`, "
       "`vignettes` (executed vignette outputs; the reader-study images, which include real photos, are excluded).\n")

# ---------------------------------------------------------------- Qwen-Image-2.1 generations (66,780 images)
repo(QW, "dataset")
G = W / "qwen_lora/generations"
for method in ["edit", "t2i"]:
    for k in range(10):
        tar_unit(QW, "dataset", G / method, f"{k:02d}", f"{method}/{k:02d}.tar")
    for f in sorted((G / method).iterdir()):
        if f.is_file():
            plain(QW, "dataset", f, f"{method}/{f.name}")
for f in sorted(G.iterdir()):
    if f.is_file():
        plain(QW, "dataset", f, f.name)
readme(QW, "dataset", "# synderm 2026: Qwen-Image-2.1 generations for Figure 3\n\n" + COMMON +
       "\n`edit/<00..09>.tar`: untuned Qwen-Image-2.1 edits of each training-pool image with the 10 varied prompts. "
       "`t2i/<00..09>.tar`: per-class LoRA (checkpoint 1500) text-to-image, one prompt. 10 images per pool image "
       "(`pool9.csv`, 3,339 reals), files `<slot>/<md5hash>.png` + `.json` sidecar. Source: `/workspace/synderm/qwen_lora/generations`.\n")

# ---------------------------------------------------------------- LoRA adapters (plain files, train-data excluded)
repo(LO, "model")
for src, dst in [(W / "qwen_lora/t2i", "t2i"), (W / "qwen_lora/edit", "edit"), (W / "lora", "stage-a-2026-10-21")]:
    if (LO, dst) in done:
        say("skip (verified before)", LO, dst); continue
    nfiles = sum(1 for p in src.rglob("*") if p.is_file() and "train-data" not in p.parts)
    say(f"upload folder {src} -> {LO}/{dst}: {nfiles} files")
    api.upload_large_folder(repo_id=LO, repo_type="model", folder_path=str(src), ignore_patterns=["**/train-data/**", "train-data/**"],
                            private=True) if False else \
    api.upload_folder(repo_id=LO, repo_type="model", folder_path=str(src), path_in_repo=dst,
                      ignore_patterns=["**/train-data/**", "train-data/**"], commit_message=f"backup {dst}")
    remote_n = sum(1 for f in api.list_repo_tree(LO, repo_type="model", path_in_repo=dst, recursive=True) if hasattr(f, "size") and not hasattr(f, "tree_id"))
    if remote_n != nfiles:
        sys.exit(f"STOP: {dst}: {remote_n} files on the hub, {nfiles} local")
    e = dict(repo=LO, path=dst, src=str(src), files=nfiles, time=time.strftime("%F %T"))
    with open(LOG, "a") as f:
        f.write(json.dumps(e) + "\n")
    done[(LO, dst)] = e
readme(LO, "model", "# synderm 2026: Qwen-Image-2.1 LoRA adapters\n\n" + COMMON.replace("Image folders are stored", "Files are stored as on the volume; image folders (none here) would be") +
       "\n`t2i/<class>/r16-lr0.0001-512px-2000steps/`: per-class text-to-image LoRAs (rank 16, lr 1e-4, 512 px, 2000 steps) "
       "with checkpoints 500/1000/1500/2000; checkpoint 1500 was used for the Figure 3 t2i arm. `edit/psoriasis/`: the edit "
       "LoRA pilot (not used: it learned to ignore the reference). `*/smoke-20steps*`: smoke tests. `stage-a-2026-10-21/`: "
       "the first multi-class t2i LoRA (200 steps). Training photos (`train-data/`) are not included.\n")

# ---------------------------------------------------------------- Nano Banana 2.1 Figure 3 edits (after the 6-image fix)
fix_log = Path("/workspace/logs/pod-2026-10-31/api-nano-banana-fig3-fix6.log")
while "DONE:" not in fix_log.read_text():
    say("waiting for the Nano Banana fix run to write synthetic.csv"); time.sleep(120)
repo(NB, "dataset")
A = W / "api_generations/nano-banana-2.1"
for k in range(10):
    tar_unit(NB, "dataset", A, f"{k:02d}", f"{k:02d}.tar")
for name in ["synthetic.csv", "generate.log", "progress.json"]:
    plain(NB, "dataset", A / name, name)
readme(NB, "dataset", "# synderm 2026: Nano Banana 2.1 edits for Figure 3\n\n" + COMMON +
       "\n`<00..09>.tar`: `gemini-nano-banana-2.1` (1K) reference-image edits, 10 per training-pool image (3,339 reals of "
       "`pool9.csv`), slot k = varied prompt k; `<slot>/<md5hash>.png` + `.json` sidecar (prompt, reference, tokens, cost). "
       "Refused slots were filled from another reference of the same class (`topup` in `synthetic.csv`). Total cost $1,078.24.\n")

# ---------------------------------------------------------------- manifests
for rid in [QW, PI, NB, LO]:
    rows = [json.dumps(e) for (r, _), e in done.items() if r == rid]
    rtype = "model" if rid == LO else "dataset"
    api.upload_file(path_or_fileobj=("\n".join(rows) + "\n").encode(), path_in_repo="MANIFEST.jsonl", repo_id=rid, repo_type=rtype, commit_message="manifest")
say("BACKUP DONE")
