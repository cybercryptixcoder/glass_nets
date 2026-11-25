"""Like-for-like with the fresh check: score the frozen law and the network on the held-out
random systems as if their integrations had stopped at 1e6 orbits (truth exact below 6,
censored above)."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from short_laws import L
from final_law import FORMS, predict
Q = L("random"); ok = ~np.isnan(Q["nn"]); fl = json.load(open("results/planets/FROZEN_LAW.json"))
law = predict(FORMS["F4 F2 + c5 lEMcross_near"][1], np.array(fl["c"]), fl["const_short_unstable"], Q); net = np.clip(Q["nn"], 4, 9)
y = Q["y"]; st = Q["stable"].astype(bool); keep = Q["a"] & ok; cens = keep & (st | (y >= 6)); u = keep & ~cens
out = {"n_kept": int(keep.sum()), "n_unstable_before_1e6": int(u.sum())}
for tag, p in [("law", law), ("network", net)]:
    r = np.where(cens, np.minimum(p - 6, 0), p - y)
    out[tag] = {"unstable_rmse": round(float(np.sqrt(np.mean((p[u] - y[u]) ** 2))), 4), "censored_rmse_all": round(float(np.sqrt(np.mean(r[keep] ** 2))), 4),
                "false_alarm": round(float(np.mean(p[cens] < 6)), 3), "bias_unstable": round(float(np.mean(p[u] - y[u])), 3)}
print(out); json.dump(out, open("results/planets/random_as_if_1e6.json", "w"), indent=1)
