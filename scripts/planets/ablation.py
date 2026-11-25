"""Each term earns its place: refit the law (censored, resonant only) without one term at
a time; score on the 1,736 network-scored random systems."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from short_laws import L
from final_law import fit, predict, evaluate
R, Q = L("resonant"), L("random"); ok = ~np.isnan(Q["nn"])
s_ = lambda T: np.minimum(T["Rm"], 1.0)
terms = {"Xh": lambda T: T["Xh"], "chaos log10(MEGNO-2)": lambda T: T["lmeg"], "forcing log10(sigma_EM/EM_cross)": lambda T: T["lEMfracstdnear"],
         "crossing min(R,1)": s_, "near spacing log10(EM_cross)": lambda T: T["lEMcrossnear"]}
def make(names):
    fs = [terms[n] for n in names]
    return len(fs) + 1, (lambda c, T: c[0] + sum(c[j + 1] * f(T) for j, f in enumerate(fs)))
out = {}
for drop in [None] + list(terms):
    names = [n for n in terms if n != drop]; k, fn = make(names)
    c, csu = fit(fn, max(k, 4) if k >= 4 else k, R, True) if k >= 4 else fit(fn, k, R, True)
    ev = evaluate(predict(fn, c, csu, Q), Q["nn"], Q, ok)
    out[f"without {drop}" if drop else "full law"] = {"unstable": ev["unstable"]["law"], "with_stable": ev["with_stable"]["law"]}
    print(f"{('without ' + drop) if drop else 'full law':45s}", out[f"without {drop}" if drop else "full law"], flush=True)
json.dump(out, open("results/planets/ablation.json", "w"), indent=1)
