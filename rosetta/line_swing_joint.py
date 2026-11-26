"""Exploratory analysis for RESULTS.md section 9 (Rosetta). Usage: python3 rosetta/line_swing_joint.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L
from rosetta.run_feats import CV, CI
from rosetta.fresh_where import totals
from rosetta.fresh_knowledge import outcome
from rosetta.line_swing import swings
from kepler.core.residual import surrogate
K = os.path.expanduser("~/kdata")
lines = json.load(open(os.path.join(L.R, "LD2_lines.json"))); fens = [it["fen"] for it in L.items]
S = np.array([swings(f, l) for f, l in zip(fens, lines)], dtype=np.float64)
fm = json.load(open(os.path.join(K, "chess", "rules", "fresh_meta.json")))["items"]; ffens = [it["fen"] for it in fm]
fres = json.load(open(os.path.join(K, "chess", "rules", "fresh_results.json")))
zl = np.load(os.path.join(L.R, "fresh_cache_partial.npz"), allow_pickle=True); flines = json.loads(str(zl["lines"]))
Sf = np.array([swings(f, l) for f, l in zip(ffens, flines)], dtype=np.float64)
z = np.load(os.path.join(L.R, "fresh_cache.npz"), allow_pickle=True); CXf = z["CX"]
To = totals(np.asarray(CV, dtype=np.float32), CI); Tf = np.load(os.path.join(L.R, "fresh_totals_s10.npy"))
att = np.array(json.load(open(f"{L.RES}/glass2_attr.json"))["attribution"]); top = np.argsort(-att)[:16]
yo = L.outcome(); g = np.where(~np.isnan(yo))[0]; yf = outcome(fres, ffens); gf = np.where(~np.isnan(yf))[0]; y = yf[gf]; v = np.var(y)
e0 = (y - surrogate(L.CX[g], yo[g]).predict(CXf[gf])) ** 2
p16 = surrogate(np.column_stack([L.CX[g], To[g][:, top]]), yo[g]).predict(np.column_stack([CXf[gf], Tf[gf][:, top]]))
p16s = surrogate(np.column_stack([L.CX[g], To[g][:, top], S[g]]), yo[g]).predict(np.column_stack([CXf[gf], Tf[gf][:, top], Sf[gf]]))
# can the 16 totals (plus textbook) predict the swing itself on fresh positions?
ps = surrogate(np.column_stack([L.CX[g], To[g][:, top]]), S[g, 2]).predict(np.column_stack([CXf[gf], Tf[gf][:, top]]))
pc = surrogate(L.CX[g], S[g, 2]).predict(CXf[gf])
pa = surrogate(np.column_stack([L.CX[g], To[g]]), S[g, 2]).predict(np.column_stack([CXf[gf], Tf[gf]]))
r2 = lambda p, t: float(1 - np.mean((p - t) ** 2) / np.var(t))
out = {"top16_gain": round(float((e0.mean() - ((y - p16) ** 2).mean()) / v), 5), "top16_plus_swing_gain": round(float((e0.mean() - ((y - p16s) ** 2).mean()) / v), 5),
       "swing3_r2_from_textbook": round(r2(pc, Sf[gf, 2]), 4), "swing3_r2_from_textbook_plus_top16": round(r2(ps, Sf[gf, 2]), 4),
       "swing3_r2_from_textbook_plus_all512": round(r2(pa, Sf[gf, 2]), 4)}
print(out); json.dump(out, open(f"{L.RES}/line_swing_joint_s10.json", "w"), indent=1)
