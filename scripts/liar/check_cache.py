"""LT1-score post-hoc agreement check: cached vs full probe features on 20 test-split transcripts (seed 1).
Pass if Pearson r >= 0.99 and median |difference| <= 0.1. If it fails, every test-split transcript's features
(with and without the instruction) are recomputed with the full path and saved as <tag>_feats_full.npy and
<tag>_feats_noinstr_full.npy (NaN rows for train-split transcripts).
Usage: python3 check_cache.py OUTDIR HF_MODEL TAG"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import liar_common as L
out, name, tag = sys.argv[1:4]
rows = [json.loads(l) for l in open(os.path.join(out, f"{tag}_transcripts.jsonl"))]
F = np.load(os.path.join(out, f"{tag}_feats.npy"))
m = L.load(name, out)
test = [i for i, r in enumerate(rows) if r["qi"] not in L.TRAIN]
pick = sorted(np.random.default_rng(1).choice(test, size=min(20, len(test)), replace=False).tolist())
a, f, saved = [], [], []
for i in pick:
    b = L.base_msgs(rows[i]); a += L.probe_feats(m, b); f += L.probe_feats_full(m, b); saved += F[i].tolist()
a, f, saved = map(np.array, (a, f, saved))
d = np.abs(a - f)
res = dict(tag=tag, n_transcripts=len(pick), pearson_r=float(np.corrcoef(a, f)[0, 1]), median_abs_diff=float(np.median(d)),
           max_abs_diff=float(d.max()), mean_abs_full=float(np.abs(f).mean()),
           rerun_vs_saved_max=float(np.abs(a - saved).max()))
res["pass"] = bool(res["pearson_r"] >= 0.99 and res["median_abs_diff"] <= 0.1)
print(res, flush=True)
json.dump(res, open(os.path.join(out, f"{tag}_check_cache.json"), "w"), indent=1)
if not res["pass"]:
    t0 = time.time(); Ff = np.full(F.shape, np.nan, np.float32); Fn = np.full(F.shape, np.nan, np.float32)
    for j, i in enumerate(test):
        Ff[i] = L.probe_feats_full(m, L.base_msgs(rows[i])); Fn[i] = L.probe_feats_full(m, L.base_msgs(rows[i], instr=False))
        if j % 50 == 0:
            print("full recompute", j, "of", len(test), f"{time.time() - t0:.0f}s", flush=True)
            np.save(os.path.join(out, f"{tag}_feats_full.npy"), Ff); np.save(os.path.join(out, f"{tag}_feats_noinstr_full.npy"), Fn)
    np.save(os.path.join(out, f"{tag}_feats_full.npy"), Ff); np.save(os.path.join(out, f"{tag}_feats_noinstr_full.npy"), Fn)
    print("DONE full recompute", len(test), f"{time.time() - t0:.0f}s", flush=True)
