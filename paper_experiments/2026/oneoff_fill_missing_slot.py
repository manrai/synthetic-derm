# Run 2026-10-31 13:12 UTC (REPRODUCE.md section 5); cost then recorded output tokens only.
# One-off: fill lichen planus f03efb15 slot 07 with an edit of another fully accepted lichen planus reference
import sys, json, random, time, os
sys.path.insert(0, "/root/synthetic-derm/paper_experiments/2026")
import pandas as pd
import api_generate_pool as g
g.load_env(g.REPO / ".env")
O = "/workspace/synderm/api_generations/nano-banana-2.1"
md5, slot, label = "f03efb152b525450bdd403f8a51e7405", 7, "lichen planus"
syn = pd.read_csv(f"{O}/synthetic.csv")
lp = syn[syn.label == label]
ok = lp.assign(own=lp.ref_md5 == lp.md5hash, nt=~lp.topup).groupby("md5hash").agg(n=("slot", "size"), own=("own", "all"), nt=("nt", "all")); ok = (ok.n == 10) & ok.own & ok.nt
refs = sorted(set(ok[ok].index) - {md5, "acb77c90f73261c4be5de36e0db7bc52"})
print(len(refs), "accepted lichen planus references")
prompt = g.EDIT_FRAME.format(condition=label, variation=g.EDIT_VARIATIONS[slot])
jp = f"{O}/{slot:02d}/{md5}.json"; prev = json.load(open(jp))
attempts, refusals = prev["attempts"], prev["refusals"]
for ref in random.Random(1).sample(refs, 4):
    t = time.time(); attempts += 1
    try:
        image, info = g.google_edit("gemini-nano-banana-2.1", "1K", prompt, f"/root/data/fitzpatrick17k/finalfitz17k/{ref}.jpg", 30 / 1e6, 1.5 / 1e6)
    except g.Refused as e:
        refusals += 1; print("refused", ref[:8], str(e)[:120]); continue
    png = f"{O}/{slot:02d}/{md5}.png"; image.save(png)
    rec = dict(md5hash=md5, label=label, slot=slot, model="gemini-nano-banana-2.1", quality="1K", prompt_mode="varied",
               variation=slot, prompt=prompt, ref_md5=ref, topup=True, refused_error=prev["refused_error"], status="ok",
               latency_s=round(time.time() - t, 1), **info, attempts=attempts, refusals=refusals, image=png,
               note="filled by a one-off retry 2026-10-31 after the pool script's top-up was refused")
    json.dump(rec, open(jp, "w"), indent=1)
    row = pd.DataFrame([dict(md5hash=md5, label=label, slot=slot, location=png, ref_md5=ref, topup=True)])
    pd.concat([syn, row]).to_csv(f"{O}/synthetic.csv", index=False)
    print("OK with", ref, info); break
else:
    prev.update(attempts=attempts, refusals=refusals); json.dump(prev, open(jp, "w"), indent=1); print("all 4 refused")
