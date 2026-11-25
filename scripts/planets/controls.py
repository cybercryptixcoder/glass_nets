"""Controls for the final law (F4, censored fit), all on the network-scored random systems.
1. must-fail: shuffle the short-integration summaries across systems (keep initial
   conditions) -> the law must lose its advantage.
2. published law (Lammers, Hadden & Murray 2024) and the constant predictor on the same systems.
3. stability of the constants: refit on 5 disjoint fifths of the resonant training set."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from short_laws import L
from final_law import FORMS, fit, predict, evaluate
from law import load as load_ic, pair_feats, lhm_pair
P = os.path.expanduser("~/pdata")
R, Q = L("resonant"), L("random"); ok = ~np.isnan(Q["nn"])
name = "F4 F2 + c5 lEMcross_near"; k, fn = FORMS[name]
c, csu = fit(fn, k, R, True); out = {"law_params": np.round(c, 4).tolist()}
base = evaluate(predict(fn, c, csu, Q), Q["nn"], Q, ok); out["law"] = base
rng = np.random.default_rng(5); Qs = dict(Q); perm = rng.permutation(len(Q["y"]))
for v in ["lmeg", "lEMfracstdnear", "su"]: Qs[v] = Q[v][perm]
out["must_fail_shuffled_short_summaries"] = evaluate(predict(fn, c, csu, Qs), Q["nn"], Q, ok)
E, y, ys, st = load_ic("random"); idx = Q["i"]
lhm = np.minimum(lhm_pair(pair_feats(E, 0)), lhm_pair(pair_feats(E, 1)))[idx]
out["LHM2024_published"] = evaluate(np.clip(lhm, 4, 9), Q["nn"], Q, ok)
const = np.full(len(idx), float(np.median(R["y"][R["u"]])))
out["constant"] = evaluate(const, Q["nn"], Q, ok)
parts = np.array_split(np.random.default_rng(9).permutation(len(R["y"])), 5); cs = []
for p in parts:
    m = np.zeros(len(R["y"]), bool); m[p] = True
    Rp = {q: (v[m] if hasattr(v, "shape") and v.shape == R["y"].shape else v) for q, v in R.items()}
    cp, _ = fit(fn, k, Rp, True); cs.append(cp)
cs = np.array(cs); out["constants_over_5_disjoint_fifths"] = {"mean": np.round(cs.mean(0), 3).tolist(), "std": np.round(cs.std(0), 3).tolist()}
for kk in ["law", "must_fail_shuffled_short_summaries", "LHM2024_published", "constant"]:
    a = out[kk]; print(f"{kk:38s} unstable {a['unstable']['law']:.3f} (network {a['unstable']['network']:.3f})  with stable {a['with_stable']['law']:.3f} (network {a['with_stable']['network']:.3f})")
print("constants over fifths", out["constants_over_5_disjoint_fifths"])
json.dump(out, open("results/planets/controls.json", "w"), indent=1)
