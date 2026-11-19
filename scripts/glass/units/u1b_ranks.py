"""U1b: are hidden units relu(a*x_t + sum_k c_k * k-th largest of history)?"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import judge, nets

def feats(X, t):
    s = -np.sort(-X[:, :t], 1)                      # descending order statistics of x_1..x_t
    return np.concatenate([X[:, t - 1:t], s], 1)

def fit(h, F):
    act = h > 1e-9
    if act.sum() < 20: return None, 0.0
    w = np.linalg.lstsq(F[act], h[act], rcond=None)[0]
    for _ in range(3):
        use = act | (F @ w > 0); w = np.linalg.lstsq(F[use], h[use], rcond=None)[0]
    p = np.maximum(F @ w, 0); return w, 1 - ((h - p) ** 2).mean() / h.var()

def run(W, X, rep=None):
    s = np.zeros((len(X), W["W_h"].shape[0]))
    for t in range(1, X.shape[1] + 1):
        s = np.maximum(X[:, t - 1:t] * W["W_x"] + s @ W["W_h"], 0)
        if rep and t in rep:
            for j, w in rep[t].items(): s[:, j] = np.maximum(feats(X, t) @ w, 0)
    return s @ W["W_o"]

out = {}
for target in sys.argv[1:] or ["zoo_2nd_argmax_16_10", "zoo_2nd_argmax_8_5", "zoo_argmedian_16_5"]:
    meta, W, X = judge.load_target(target); Xf, Xt = X[:4000], X[4000:8000]   # fit on one half, test on the other
    acts = {}; nets.forward("rnn_relu", W, Xf, acts=acts)
    rep, r2s = {}, []
    for t in range(2, X.shape[1] + 1):
        H = np.asarray(acts[f"h{t}"]).reshape(len(Xf), -1); F = feats(Xf, t)
        for j in range(H.shape[1]):
            if H[:, j].var() < 1e-10: continue
            w, r2 = fit(H[:, j], F); r2s.append(r2)
            if w is not None: rep.setdefault(t, {})[j] = w
    ref = run(W, Xt)
    joint = float(judge.same(ref, run(W, Xt, rep)).mean())
    singles = [float(judge.same(ref, run(W, Xt, {t: {j: w}})).mean()) for t in rep for j, w in rep[t].items()]
    out[target] = {"r2_median": float(np.median(r2s)), "r2_min": float(np.min(r2s)), "joint": joint,
                   "single_median": float(np.median(singles)), "single_min": float(np.min(singles)), "n_units": len(singles)}
    print(target, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in out[target].items()}), flush=True)
(ROOT / "results/glass/units/u1b.json").write_text(json.dumps(out, indent=1))
