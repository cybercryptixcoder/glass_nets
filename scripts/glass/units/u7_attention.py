"""U7: attention units in transformers (UNITS_PROGRAM §8 U7 SPEC). Per layer, head and query position: the attention mass on
each library unit's target set; a unit is labelled when it carries >= 90% of the mass on average."""
import sys, json
from pathlib import Path
from collections import Counter
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import judge, nets

def analyse(target, variant="original", n=4000):
    meta, W, X = judge.load_target(target, variant, 0); X = np.asarray(X[:n]).astype(int); B, P = X.shape
    acts = {}; nets.forward(meta["forward"], W, X, acts=acts)
    layers = sorted(k for k in acts if k.endswith(".pattern")); out = {}; tot_mass = expl = 0.0
    for name in layers:
        A = np.asarray(acts[name]); H = A.size // (B * P * P); A = A.reshape(B, H, P, P)
        for h in range(H):
            for i in range(P):
                a = A[:, h, i, :]                                   # (B, keys)
                am = a.argmax(1); kstar = Counter(am).most_common(1)[0][0]
                sets = {"self": np.eye(P)[i][None].repeat(B, 0).astype(bool),
                        "fixed position": np.eye(P)[kstar][None].repeat(B, 0).astype(bool),
                        "same token": X == X[:, i:i + 1],
                        "pointer (j = x_i)": (np.arange(P)[None] == X[:, i:i + 1]),
                        "inverse pointer (x_j = i)": X == i}
                first = np.zeros((B, P), bool); first[:, 0] = True
                sets.update({u + " + first position": m | first for u, m in list(sets.items()) if u not in ("fixed position",)})
                mass = {u: float((a * m).sum(1).mean()) for u, m in sets.items()}
                if a.max(1).mean() <= 1.5 / P: mass["uniform"] = 1.0
                best = max(mass, key=lambda u: (round(mass[u], 2), -len(u)))   # ties: prefer the simpler unit
                lab = best if mass[best] >= 0.9 else "unlabelled"
                out[f"{name}|h{h}|q{i}"] = {"label": lab, "mass": round(mass[best], 3), "best": best, "fixed_key": int(kstar)}
                tot_mass += 1; expl += (lab != "unlabelled")
    return {"target": target, "variant": variant, "labelled_share": expl / max(tot_mass, 1), "cells": out}

if __name__ == "__main__":
    res = {}
    for t in sys.argv[1:] or ["tracr_reverse", "tracr_hist", "lc_3_4", "lc_4_5", "lc_6_5", "lc_8_6"]:
        for var in (["original"] if t.startswith("tracr") else ["original", "shuffled"]):
            r = analyse(t, var); res[f"{t}|{var}"] = r
            labs = Counter(c["label"] for c in r["cells"].values())
            print(f"{t:12s} {var:9s} labelled {100*r['labelled_share']:.0f}% of (layer, head, query) | {dict(labs.most_common(5))}", flush=True)
    (ROOT / "results/glass/units/u7.json").write_text(json.dumps(res, indent=0))
