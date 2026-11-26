"""Exploratory stability check (RESULTS.md section 9): rank LD2's variables by attribution on 600 other training
positions; overlap with the first top 16, and the fresh-game gain of the new top 16. Usage: python3 rosetta/attr_stability.py"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.environ.setdefault("SITE", "10")
from rosetta import run_feats as F  # noqa: E402
from rosetta.run_feats import L, DEC, CV, CI  # noqa: E402
from rosetta.engine import Defs  # noqa: E402
from rosetta.fresh_where import totals  # noqa: E402
from rosetta.fresh_knowledge import outcome  # noqa: E402
from kepler.core.residual import surrogate  # noqa: E402
K = os.path.expanduser("~/kdata")
D = Defs(L.P, L.names, L.TR, L.TE); rows = D.rs[600:1200]; m = L.get_net()
h = m(L.planes(rows), upto=L.SITE).detach().requires_grad_(True)
_, wdl = m.heads(h); (wdl[:, 0] - wdl[:, 2]).sum().backward()
g = h.grad.detach().numpy().reshape(len(rows), 128, 64)
Z = F.dense_codes(rows); att2 = np.abs((Z * np.einsum("ncs,cf->nsf", g, DEC.numpy())).sum(1)).mean(0)
att1 = np.array(json.load(open(f"{L.RES}/glass2_attr.json"))["attribution"])
t1, t2 = set(np.argsort(-att1)[:16].tolist()), np.argsort(-att2)[:16]
To = totals(np.asarray(CV, dtype=np.float32), CI); Tf = np.load(os.path.join(L.R, "fresh_totals_s10.npy"))
fm = json.load(open(os.path.join(K, "chess", "rules", "fresh_meta.json")))["items"]; ffens = [it["fen"] for it in fm]
fres = json.load(open(os.path.join(K, "chess", "rules", "fresh_results.json")))
z = np.load(os.path.join(L.R, "fresh_cache.npz"), allow_pickle=True); CXf = z["CX"]
yo = L.outcome(); gi = np.where(~np.isnan(yo))[0]; yf = outcome(fres, ffens); gf = np.where(~np.isnan(yf))[0]; y = yf[gf]; v = np.var(y)
e0 = (y - surrogate(L.CX[gi], yo[gi]).predict(CXf[gf])) ** 2
p = surrogate(np.column_stack([L.CX[gi], To[gi][:, t2]]), yo[gi]).predict(np.column_stack([CXf[gf], Tf[gf][:, t2]]))
out = {"overlap_of_top16": len(t1 & set(t2.tolist())), "rank_correlation_all_512": round(float(np.corrcoef(np.argsort(np.argsort(-att1)), np.argsort(np.argsort(-att2)))[0, 1]), 3),
       "new_top16_fresh_gain": round(float((e0.mean() - ((y - p) ** 2).mean()) / v), 5)}
print(out); json.dump(out, open(f"{L.RES}/attr_stability_s{L.SITE}.json", "w"), indent=1)
