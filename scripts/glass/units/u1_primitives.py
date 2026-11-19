"""U1 (docs/glass/UNITS_PROGRAM.md): do single hidden units of AlgZoo RNNs compute simple primitives of the input history?"""
import sys, json, itertools
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import judge, nets

def base_quantities(X, t):
    """Primitive quantities of the history x_1..x_t (t is 1-based)."""
    H = X[:, :t]; s = np.sort(H, 1)
    M = s[:, -1]; S = s[:, -2] if t > 1 else s[:, -1]; m = s[:, 0]
    Hp = X[:, :t - 1] if t > 1 else None
    if t > 1:
        sp = np.sort(Hp, 1); Mp = sp[:, -1]; Sp = sp[:, -2] if t > 2 else sp[:, -1]; xp = X[:, t - 2]
    else:
        Mp = Sp = xp = np.zeros(len(X))
    return {"x_t": X[:, t - 1], "x_prev": xp, "max": M, "2nd": S, "min": m, "max_prev": Mp, "2nd_prev": Sp}

def relu_fit(h, F):
    """Fit h ~ relu(F @ w) (no bias: the RNNs have none). Linear fit on the active set, two refinements."""
    act = h > 1e-9
    if act.sum() < 20:
        return 0.0, None
    w = np.linalg.lstsq(F[act], h[act], rcond=None)[0]
    for _ in range(2):
        pred = F @ w; use = act | (pred > 0)
        w = np.linalg.lstsq(F[use], h[use], rcond=None)[0]
    pred = np.maximum(F @ w, 0); v = h.var()
    return (1 - ((h - pred) ** 2).mean() / v) if v > 0 else 0.0, w

def analyse(target, variant="original", n_inputs=4000):
    meta, W, X = judge.load_target(target, variant, 0); X = X[:n_inputs]
    acts = {}; nets.forward(meta["forward"], W, X, acts=acts)
    T = X.shape[1]; rows = []; tot = 0.0; cov = 0.0
    for t in range(1, T + 1):
        Hs = np.asarray(acts[f"h{t}"]).reshape(len(X), -1); B = base_quantities(X, t); names = list(B)
        for j in range(Hs.shape[1]):
            h = Hs[:, j]; v = h.var()
            if v < 1e-10:
                continue
            best = (-1, None, None)
            for k in (1, 2):
                for combo in itertools.combinations(names, k):
                    r2, w = relu_fit(h, np.stack([B[c] for c in combo], 1))
                    if r2 > best[0] + (0.02 if k == 2 else 0):   # prefer the simpler unit unless 2 terms clearly help
                        best = (r2, combo, w)
            rows.append({"t": t, "unit": j, "var": float(v), "r2": float(best[0]), "prims": list(best[1] or []),
                         "w": None if best[2] is None else [round(float(x), 3) for x in best[2]]})
            tot += v; cov += v * (best[0] >= 0.9)
    return {"target": target, "variant": variant, "coverage": cov / tot if tot else 0.0, "rows": rows}

if __name__ == "__main__":
    targets = sys.argv[1:] or ["zoo_2nd_argmax_2_2", "zoo_2nd_argmax_4_3", "zoo_2nd_argmax_4_4", "zoo_2nd_argmax_8_5",
                               "zoo_2nd_argmax_8_8", "zoo_2nd_argmax_16_10", "zoo_argmedian_8_5", "zoo_argmedian_16_5"]
    out = {}
    for t in targets:
        a = analyse(t); c = analyse(t, "shuffled"); out[t] = {"orig": a, "shuffled": c}
        from collections import Counter
        good = [r for r in a["rows"] if r["r2"] >= 0.9]
        prims = Counter(tuple(sorted(r["prims"])) for r in good)
        print(f"{t:24s} coverage {a['coverage']:.2f} (shuffled {c['coverage']:.2f}) | neuron-steps explained {len(good)}/{len(a['rows'])} | top units: "
              + ", ".join(f"{'+'.join(k)}x{v}" for k, v in prims.most_common(5)), flush=True)
    (ROOT / "results/glass/units/u1.json").write_text(json.dumps(out, indent=0))
