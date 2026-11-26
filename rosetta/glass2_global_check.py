"""Exploratory analysis for RESULTS.md section 9 (Rosetta). Usage: python3 rosetta/glass2_global_check.py"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.environ.setdefault("SITE", "10")
from rosetta import run_feats as F
from rosetta.run_feats import L
from rosetta.engine import Defs
att = np.array(json.load(open("results/rosetta/glass2_attr.json"))["attribution"]); order = np.argsort(-att)
D = Defs(L.P, L.names, L.TR, L.TE); rows = D.rt[:1000]; m = L.get_net()
Z = F.dense_codes(rows)                                   # [n, 64, 512]
with torch.no_grad():
    _, wdl = m(L.planes(rows)); v0 = (wdl[:, 0] - wdl[:, 2]).numpy()
sur = L.sur(L.CX[rows]); resid = v0 - sur
out = []
for f in order[:16]:
    z = Z[:, :, int(f)]; tot = z.var()
    pm = z.mean(1)                                         # position mean
    between = pm.var() / tot if tot > 0 else 0.0
    c_v = np.corrcoef(pm, v0)[0, 1] if pm.std() > 0 else 0.0
    c_sur = np.corrcoef(pm, sur)[0, 1] if pm.std() > 0 else 0.0
    c_res = np.corrcoef(pm, resid)[0, 1] if pm.std() > 0 else 0.0
    active = float((z > 0).mean())
    r = {"feature": int(f), "attribution_share": round(float(att[f] / att.sum()), 3), "share_between_positions": round(float(between), 3),
         "squares_active": round(active, 3), "corr_position_mean_with_value": round(float(c_v), 3),
         "corr_with_textbook_value": round(float(c_sur), 3), "corr_with_residual": round(float(c_res), 3)}
    out.append(r); print(r, flush=True)
json.dump(out, open("results/rosetta/glass2_global_check_s10.json", "w"), indent=1)
