"""Exploratory (RESULTS.md section 9): how much of each of LD2's 16 evaluation variables do the 61 textbook concepts
restate? Board totals on held-out positions: the textbook concepts' R2 (gradient-boosted trees fitted on training
positions) and the three concepts most correlated with each total. Usage: python3 rosetta/carriers_textbook.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.run_feats import CV, CI  # noqa: E402
from rosetta.fresh_where import totals  # noqa: E402
from kepler.core.residual import surrogate  # noqa: E402
cn = [str(x) for x in np.load(os.path.join(L.C, "LD2", "concepts.npz"))["names"]]
att = np.array(json.load(open(f"{L.RES}/glass2_attr.json"))["attribution"]); top = np.argsort(-att)[:16]
T = totals(np.asarray(CV, dtype=np.float32), CI); TR, TE = np.asarray(L.TR), np.asarray(L.TE)
out = []
for f in top:
    t = T[:, int(f)]
    p = surrogate(L.CX[TR], t[TR]).predict(L.CX[TE]); r2 = float(1 - np.mean((p - t[TE]) ** 2) / np.var(t[TE]))
    c = np.nan_to_num([np.corrcoef(L.CX[TE][:, j], t[TE])[0, 1] for j in range(L.CX.shape[1])])
    best = [(cn[j], round(float(c[j]), 3)) for j in np.argsort(-np.abs(c))[:3]]
    r = {"feature": int(f), "textbook_r2": round(r2, 3), "top_concepts": best}; out.append(r); print(r, flush=True)
json.dump(out, open(f"{L.RES}/carriers_textbook_s{L.SITE}.json", "w"), indent=1)
