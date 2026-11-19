"""U9: data-driven sparse units for neuron-steps the library leaves unexplained (pre-registered in DIARY).
h ~ relu(sum of <= 3 features), greedy forward selection on half the inputs, held-out R^2 on the other half.
Pool at step t: x_1..x_t, order statistics of x_1..x_t, relu(x_i - x_j) for i != j <= t."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import judge, nets, unitmap

def pool(X, t):
    P = X[:, :t]; names = [f"x{i+1}" for i in range(t)]; cols = [P[:, i] for i in range(t)]
    S = -np.sort(-P, 1); names += [f"ord{k+1}" for k in range(t)]; cols += [S[:, k] for k in range(t)]
    for i in range(t):
        for j in range(t):
            if i != j:
                names.append(f"relu(x{i+1}-x{j+1})"); cols.append(np.maximum(P[:, i] - P[:, j], 0))
    return names, np.stack(cols, 1)

def search(h, F, names, k=3):
    n = len(h) // 2; tr, te = slice(0, n), slice(n, None); chosen = []; best = None
    for _ in range(k):
        cand = []
        for f in range(F.shape[1]):
            if f in chosen: continue
            idx = chosen + [f]; w = unitmap.fit(h[tr], F[tr][:, idx])
            if w is None: continue
            cand.append((unitmap.r2(h[tr], np.maximum(F[tr][:, idx] @ w, 0)), f, w))
        if not cand: break
        r, f, w = max(cand, key=lambda c: c[0]); chosen.append(f); best = (list(chosen), w)
        if r >= 0.999: break
    if best is None: return None
    idx, w = best
    return {"features": [names[i] for i in idx], "w": [float(v) for v in w], "r2_test": unitmap.r2(h[te], np.maximum(F[te][:, idx] @ w, 0))}

def run(target, variant, select):
    meta, W, X = judge.load_target(target, variant, 0); X = np.asarray(X[:8000], dtype=np.float64)
    acts = {}; nets.forward("rnn_relu", W, X, acts=acts); cells = []
    for t in range(2, X.shape[1] + 1):
        H = np.asarray(acts[f"h{t}"]).reshape(len(X), -1); names, F = pool(X, t)
        for u in range(H.shape[1]):
            if not select(t, u) or H[:, u].var() < 1e-12: continue
            r = search(H[:, u], F, names)
            if r: cells.append({"t": t, "unit": u, "var": float(H[:, u].var()), **r})
    v = sum(c["var"] for c in cells); e = sum(c["var"] for c in cells if c["r2_test"] >= 0.9)
    return {"target": target, "variant": variant, "n": len(cells), "explained_share": e / v if v else None, "cells": cells}

def library_fit(target, variant):
    f = ROOT / f"results/glass/units/maps/{target}.json"
    if variant == "original":
        return {(c["t"], c["unit"]): c["fit"] for c in json.loads(f.read_text())["map"]["cells"]}
    m = unitmap.unit_map(target, variant, confidence=False); return {(c["t"], c["unit"]): c["fit"] for c in m["cells"]}

if __name__ == "__main__":
    out = {}
    if sys.argv[1:] == ["known"]:
        lf = library_fit("zoo_2nd_argmax_16_10", "original")
        r = run("zoo_2nd_argmax_16_10", "original", lambda t, u: u in (1, 6, 7, 8) and (lf.get((t, u)) or 0) >= 0.9)
        ok = np.mean([c["r2_test"] >= 0.9 for c in r["cells"]]); out["known"] = r
        print(f"KNOWN ANSWER: {len(r['cells'])} comparator neuron-steps, search reaches R2>=0.9 in {100*ok:.0f}% (bar 90%)")
        for c in r["cells"][:6]: print("  ", c["t"], c["unit"], c["features"], [round(x, 2) for x in c["w"]], round(c["r2_test"], 3))
        (ROOT / "results/glass/units/u9_known.json").write_text(json.dumps(out, indent=0)); sys.exit()
    for t in ["zoo_argmedian_8_5", "zoo_argmedian_8_7", "zoo_argmedian_16_5", "zoo_argmedian_16_7"]:
        for var in ("original", "shuffled"):
            lf = library_fit(t, var)
            r = run(t, var, lambda s, u: (lf.get((s, u)) if lf.get((s, u)) is not None else 0) < 0.9); out[f"{t}|{var}"] = r
            print(f"{t:20s} {var:9s} unexplained neuron-steps {r['n']:3d} | share of their variance now at R2>=0.9: {r['explained_share']:.2f}", flush=True)
    (ROOT / "results/glass/units/u9.json").write_text(json.dumps(out, indent=0))
