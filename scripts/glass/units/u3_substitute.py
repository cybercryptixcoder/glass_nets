"""U3: substitute U1-explained neurons by their primitive formulas inside the running RNN; compare with mean-ablation."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab")); sys.path.insert(0, str(Path(__file__).parent))
import judge, nets
from u1_primitives import base_quantities

def run(W, X, replace=None):
    """replace: {(t, j): fn(X)->values} applied after step t (1-based)."""
    s = np.zeros((len(X), W["W_h"].shape[0]))
    for t in range(1, X.shape[1] + 1):
        s = np.maximum(X[:, t - 1:t] * W["W_x"] + s @ W["W_h"], 0)
        for (tt, j), fn in (replace or {}).items():
            if tt == t:
                s[:, j] = fn(X)
    return s @ W["W_o"]

d = json.loads((ROOT / "results/glass/units/u1.json").read_text())
for target in sys.argv[1:] or ["zoo_2nd_argmax_16_10", "zoo_2nd_argmax_8_5", "zoo_argmedian_16_5"]:
    meta, W, X = judge.load_target(target); X = X[:4000]
    base = judge.decide(run(W, X))
    rows = [r for r in d[target]["orig"]["rows"] if r["t"] > 1 and r["r2"] >= 0.9]
    sub, abl = {}, {}
    acts = {}; nets.forward(meta["forward"], W, X, acts=acts)
    for r in rows:
        t, j, p, w = r["t"], r["unit"], r["prims"], np.array(r["w"])
        sub[(t, j)] = (lambda X, t=t, p=p, w=w: np.maximum(np.stack([base_quantities(X, t)[q] for q in p], 1) @ w, 0))
        mu = float(np.asarray(acts[f"h{t}"]).reshape(len(X), -1)[:, j].mean())
        abl[(t, j)] = (lambda X, mu=mu: np.full(len(X), mu))
    fs = float(judge.same(run(W, X), run(W, X, sub)).mean()); fa = float(judge.same(run(W, X), run(W, X, abl)).mean())
    print(f"{target}: {len(rows)} neuron-steps | substitution keeps {100*fs:.1f}% of decisions | mean-ablation keeps {100*fa:.1f}%", flush=True)

def per_unit(target, n=2000):
    meta, W, X = judge.load_target(target); X = X[:n]; ref = run(W, X)
    rows = [r for r in d[target]["orig"]["rows"] if r["t"] > 1 and r["r2"] >= 0.9]
    out = []
    for r in rows:
        t, j, p, w = r["t"], r["unit"], r["prims"], np.array(r["w"])
        fn = lambda X, t=t, p=p, w=w: np.maximum(np.stack([base_quantities(X, t)[q] for q in p], 1) @ w, 0)
        out.append((float(judge.same(ref, run(W, X, {(t, j): fn})).mean()), r["r2"], t, j, "+".join(p)))
    return sorted(out)
