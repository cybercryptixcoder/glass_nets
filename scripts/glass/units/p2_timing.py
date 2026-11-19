"""P2': timing-aware units h = relu(a(d)*M + b(d)*S + c(d)*x_t), coefficients per d = steps since the running max."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import judge, nets

def stats(X, t):
    H = X[:, :t]; o = np.argsort(-H, 1); s = np.take_along_axis(H, o, 1)
    M = s[:, 0]; S = s[:, 1] if t > 1 else s[:, 0]
    d = (t - 1 - o[:, 0]); d2 = (t - 1 - (o[:, 1] if t > 1 else o[:, 0]))
    return M, S, X[:, t - 1], d, d2

def fit_table(h, F, key, keys):
    """One relu-fit per value of the timing key (no bias); returns coefficient table."""
    tab = {}
    for k in keys:
        m = key == k
        if m.sum() < 15: continue
        Fk, hk = F[m], h[m]; act = hk > 1e-9
        if act.sum() < 5: tab[k] = np.zeros(F.shape[1]); continue
        w = np.linalg.lstsq(Fk[act], hk[act], rcond=None)[0]
        for _ in range(2):
            use = act | (Fk @ w > 0); w = np.linalg.lstsq(Fk[use], hk[use], rcond=None)[0]
        tab[k] = w
    return tab

def apply(tab, F, key):
    out = np.zeros(len(F))
    for k, w in tab.items():
        m = key == k; out[m] = np.maximum(F[m] @ w, 0)
    return out

def run(W, X, rep=None):
    s = np.zeros((len(X), W["W_h"].shape[0]))
    for t in range(1, X.shape[1] + 1):
        s = np.maximum(X[:, t - 1:t] * W["W_x"] + s @ W["W_h"], 0)
        if rep and t in rep:
            M, S, xt, d, d2 = stats(X, t); F = np.stack([M, S, xt], 1)
            for j, (tab, which) in rep[t].items(): s[:, j] = apply(tab, F, d if which == "d" else d2)
    return s @ W["W_o"]

res = {}
for target in sys.argv[1:] or ["zoo_2nd_argmax_16_10", "zoo_2nd_argmax_8_5", "zoo_argmedian_8_5", "zoo_argmedian_16_5"]:
    meta, W, X = judge.load_target(target); X = X[:8000]; Xf, Xt = X[:4000], X[4000:]
    af, at = {}, {}; nets.forward("rnn_relu", W, Xf, acts=af); nets.forward("rnn_relu", W, Xt, acts=at)
    rep, r2s, known = {}, [], {}
    for t in range(2, X.shape[1] + 1):
        Mf, Sf, xf, df, d2f = stats(Xf, t); Mt, St, xt, dt, d2t = stats(Xt, t)
        Ff, Ft = np.stack([Mf, Sf, xf], 1), np.stack([Mt, St, xt], 1)
        Hf = np.asarray(af[f"h{t}"]).reshape(4000, -1); Ht = np.asarray(at[f"h{t}"]).reshape(4000, -1)
        for j in range(Hf.shape[1]):
            if Ht[:, j].var() < 1e-10: continue
            best = None
            for which, kf, kt in (("d", df, dt), ("d2", d2f, d2t)):
                tab = fit_table(Hf[:, j], Ff, kf, range(t))
                p = apply(tab, Ft, kt); r2 = 1 - ((Ht[:, j] - p) ** 2).mean() / Ht[:, j].var()
                if best is None or r2 > best[0]: best = (r2, tab, which)
            r2s.append(best[0]); rep.setdefault(t, {})[j] = (best[1], best[2])
            if target.endswith("16_10") and j in (1, 6, 7, 8): known.setdefault(j, []).append(round(best[0], 3))
    ref = run(W, Xt); joint = float(judge.same(ref, run(W, Xt, rep)).mean())
    res[target] = {"r2_median": float(np.median(r2s)), "share_r2_ge_0.98": float(np.mean(np.array(r2s) >= 0.98)), "joint_substitution": joint}
    print(target, json.dumps({k: round(v, 3) for k, v in res[target].items()}), ("known units R2 by step: " + json.dumps(known)) if known else "", flush=True)
(ROOT / "results/glass/units/p2_timing.json").write_text(json.dumps(res, indent=1))
