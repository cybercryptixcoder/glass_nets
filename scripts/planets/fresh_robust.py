"""Robust view of the fresh check: median absolute error and the share of large misses
(> 2 dex) on the fresh systems that went unstable before 1e6 orbits."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import fresh_eval as fe
from final_law import FORMS
P = fe.P
fl = json.load(open("results/planets/FROZEN_LAW.json")); fn = FORMS["F4 F2 + c5 lEMcross_near"][1]
E = np.load(f"{P}/fresh_ic.npz")["E"]; T = fe.ic_table(E)
nb = fe.csv_rows(f"{P}/fresh_nbody_w*.csv", 3) | fe.csv_rows(f"{P}/colab/fresh_nbody*.csv", 3)
ft = fe.csv_rows(f"{P}/colab/featfresh*.csv", 14); nn = fe.csv_rows(f"{P}/colab/nnfresh*.csv", 4)
ids = np.array(sorted(set(nb) & set(ft) & set(nn)))
y = np.log10([nb[i][0] for i in ids]); cens = np.array([nb[i][1] for i in ids]) > 0
F = np.array([ft[i] for i in ids]); K = ["EMcrossnear", "EMfracstdnear", "EPstdnear", "MMRstrengthnear", "EMcrossfar", "EMfracstdfar", "EPstdfar", "MMRstrengthfar", "MEGNO", "MEGNOstd", "Tsec", "short_stable"]
S = {k: F[:, j] for j, k in enumerate(K)}; S["su"] = np.isnan(S["MEGNO"]) | (S["short_stable"] == 0)
S["lmeg"] = np.log10(np.maximum(np.where(S["su"], 1e4, S["MEGNO"]) - 2, 1e-2))
for k in ["EMfracstdnear", "EMcrossnear"]: S["l" + k] = np.log10(np.maximum(np.nan_to_num(S[k], nan=1e-8), 1e-8))
S.update({k: v[ids] for k, v in T.items()})
law = np.clip(np.where(S["su"], fl["const_short_unstable"], fn(np.array(fl["c"]), S)), 4, 9); net = np.clip([nn[i][0] for i in ids], 4, 9)
u = (y >= 4) & ~cens
out = {"n_unstable": int(u.sum())}
for tag, p in [("law", law), ("network", net)]:
    e = np.abs(p[u] - y[u]); out[tag] = {"median_abs_error": round(float(np.median(e)), 3), "share_miss_over_2dex": round(float(np.mean(e > 2)), 3), "rmse": round(float(np.sqrt(np.mean(e ** 2))), 4)}
print(out); json.dump(out, open("results/planets/fresh_robust.json", "w"), indent=1)
