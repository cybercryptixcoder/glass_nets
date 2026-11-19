"""U17: replacement curve. Replace library-explained neuron-steps by their units (best fit first) and track decision
fidelity; R* = largest replaced variance share with fidelity >= the network's accuracy (pre-registered in DIARY)."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import judge, unitmap

def unit_fn(c):
    t, p, w = c["t"], c["prims"], np.array(c["w"])
    if p[0].startswith("x_") and p[0][2:].isdigit():
        return lambda X_, t=t, w=w: np.maximum(X_[:, :t] @ w, 0)
    return lambda X_, t=t, p=p, w=w: np.maximum(np.stack([unitmap.library(X_, t)[q] for q in p], 1) @ w, 0)

def curve(tid):
    meta, W, X = judge.load_target(tid); X = np.asarray(X[:8000], dtype=np.float64); Xt = X[4000:]
    acc = float(meta["test_acc"]); m = unitmap.unit_map(tid, confidence=False)
    later = [c for c in m["cells"] if c["t"] > 1]; tot = sum(c["var"] for c in later) or 1.0
    good = sorted([c for c in later if c["fit"] >= 0.9 and c["prims"]], key=lambda c: -c["fit"])
    ref = unitmap.run(W, Xt); rep = {}; share = 0.0; pts = [(0.0, 1.0)]; rstar = 0.0
    for c in good:
        rep[(c["t"], c["unit"])] = unit_fn(c); share += c["var"] / tot
        f = float(judge.same(ref, unitmap.run(W, Xt, rep)).mean()); pts.append((share, f))
        if f >= acc: rstar = share
    return {"acc": acc, "coverage_history": m["coverage_history"], "rstar": rstar, "points": pts}

if __name__ == "__main__":
    out = {}
    for tid in sys.argv[1:]:
        r = curve(tid); out[tid] = r
        print(f"{tid}: acc {r['acc']:.3f} cov {r['coverage_history']:.2f} | R* {r['rstar']:.2f} | max replaceable {r['points'][-1][0]:.2f} at fidelity {r['points'][-1][1]:.3f}", flush=True)
    (ROOT / f"results/glass/units/u17_{len(out)}.json").write_text(json.dumps(out, indent=1))
