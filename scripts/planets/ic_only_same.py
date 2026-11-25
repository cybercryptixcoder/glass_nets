"""Initial-condition-only predictors on the same network-scored random systems:
best IC law (round 2, form B, fit on resonant unstable) and a gradient-boosted tree
on IC variables (fit on resonant unstable). Also correlation with the truth."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from fit_law2 import FORMS, R as RF
from short_laws import L
from final_law import evaluate
from sklearn.ensemble import HistGradientBoostingRegressor as H
Q = L("random"); ok = ~np.isnan(Q["nn"])
laws = json.load(open("results/planets/laws_round2.json")); nm = "B  as A with s=min(Rrms,1)"
icl = np.clip(FORMS[nm][1](np.array(laws[nm]["params"]), Q), 4, 9)
V = ["X0", "X1", "R0", "R1", "M0", "M1", "Xh", "Rm"]
Fr = {k: v for k, v in RF.items()}; m = Fr["m"]
rng = np.random.default_rng(0); sub = rng.choice(np.where(m)[0], 30000, replace=False)
g = H(max_iter=250, learning_rate=0.08, random_state=0).fit(np.column_stack([Fr[v][sub] for v in V]), Fr["y"][sub])
QF = dict(np.load(os.path.expanduser("~/pdata/random_F.npz")))
gb = np.clip(g.predict(np.column_stack([QF[v][Q["i"]] for v in V])), 4, 9)
fl = json.load(open("results/planets/final_law.json"))["F4 F2 + c5 lEMcross_near | censored"]
from final_law import FORMS as FF, predict
law = predict(FF["F4 F2 + c5 lEMcross_near"][1], np.array(fl["params"]), fl["short_unstable_const"], Q)
u = Q["u"] & ok; out = {}
for tag, p in [("ic_law_round2_B", icl), ("ic_tree", gb), ("final_law_F4_censored", law), ("network", np.clip(Q["nn"], 4, 9))]:
    ev = evaluate(p, Q["nn"], Q, ok)
    out[tag] = {"unstable": ev["unstable"]["law"], "with_stable": ev["with_stable"]["law"], "corr_truth_unstable": round(float(np.corrcoef(p[u], Q["y"][u])[0, 1]), 3),
                "bias_unstable": round(float(np.mean(p[u] - Q["y"][u])), 3)}
    print(f"{tag:24s}", out[tag])
json.dump(out, open("results/planets/information_same_systems.json", "w"), indent=1)
