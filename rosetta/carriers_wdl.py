"""Exploratory (RESULTS.md section 9): do LD2's 16 evaluation variables track winning, drawing or losing separately?
Held-out positions: correlation of each variable's board total with LD2's own W, D and L probabilities, and with the
draw probability after removing W - L (linear). Usage: python3 rosetta/carriers_wdl.py"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.run_feats import CV, CI  # noqa: E402
from rosetta.fresh_where import totals  # noqa: E402
att = np.array(json.load(open(f"{L.RES}/glass2_attr.json"))["attribution"]); top = np.argsort(-att)[:16]
rows = np.asarray(L.TE)[:3000]; m = L.get_net(); W = []
with torch.no_grad():
    for j in range(0, len(rows), 1000):
        _, wdl = m(L.planes(rows[j:j + 1000])); W.append(wdl.numpy())
W = np.concatenate(W); v = W[:, 0] - W[:, 2]
T = totals(np.asarray(CV, dtype=np.float32), CI)[rows]
X = np.column_stack([np.ones(len(v)), v]); dres = W[:, 1] - X @ np.linalg.lstsq(X, W[:, 1], rcond=None)[0]
out = []
for f in top:
    t = T[:, int(f)]; tr = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]
    r = {"feature": int(f), "corr_win": round(float(np.corrcoef(t, W[:, 0])[0, 1]), 3), "corr_draw": round(float(np.corrcoef(t, W[:, 1])[0, 1]), 3),
         "corr_loss": round(float(np.corrcoef(t, W[:, 2])[0, 1]), 3), "corr_draw_beyond_value": round(float(np.corrcoef(tr, dres)[0, 1]), 3)}
    out.append(r); print(r, flush=True)
json.dump(out, open(f"{L.RES}/carriers_wdl_s{L.SITE}.json", "w"), indent=1)
