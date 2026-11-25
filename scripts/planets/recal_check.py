"""Hard check: give the network (and the law) an affine recalibration fitted on the
random DEV half's unstable systems (test-distribution data, which favours both), score
on the FINAL half. Removes the network's +0.33 dex bias on the unstable metric."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from short_laws import L
from final_law import FORMS, predict
P = os.path.expanduser("~/pdata"); Q = L("random"); ok = ~np.isnan(Q["nn"])
DEV = np.load(f"{P}/random_split.npy")[Q["i"]]
fl = json.load(open("results/planets/final_law.json"))["F4 F2 + c5 lEMcross_near | censored"]
law = predict(FORMS["F4 F2 + c5 lEMcross_near"][1], np.array(fl["params"]), fl["short_unstable_const"], Q)
net = np.clip(Q["nn"], 4, 9); u = Q["u"] & ok; y = Q["y"]
out = {}
for tag, p in [("network", net), ("law", law)]:
    d, f = u & DEV, u & ~DEV
    A = np.column_stack([np.ones(d.sum()), p[d]]); c, *_ = np.linalg.lstsq(A, y[d], rcond=None)
    q = np.clip(c[0] + c[1] * p, 4, 9)
    out[tag] = {"final_unstable_raw": round(float(np.sqrt(np.mean((p[f] - y[f]) ** 2))), 4),
                "final_unstable_recalibrated": round(float(np.sqrt(np.mean((q[f] - y[f]) ** 2))), 4), "affine": np.round(c, 3).tolist()}
    print(tag, out[tag])
json.dump(out, open("results/planets/recalibration_check.json", "w"), indent=1)
