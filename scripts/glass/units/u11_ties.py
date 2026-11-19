"""U11: is ε95 set by near-tie inputs? (pre-registered in DIARY). P3's noise procedure, grid and seeds."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import judge, nets

def noisy(W, X, sd, eps, rng):
    s = np.zeros((len(X), W["W_h"].shape[0]))
    for t in range(1, X.shape[1] + 1):
        s = np.maximum(X[:, t - 1:t] * W["W_x"] + s @ W["W_h"], 0)
        if eps > 0:
            s = s + rng.normal(size=s.shape) * eps * sd[t]
    return s @ W["W_o"]

def margin(X, task):
    S = np.sort(X, 1)[:, ::-1]
    if task == "2nd" and X.shape[1] == 2:
        return S[:, 0] - S[:, 1]                      # with two inputs, only the one gap decides
    if task == "2nd":
        return np.minimum(S[:, 0] - S[:, 1], S[:, 1] - S[:, 2])
    m = X.shape[1] // 2
    return np.minimum(S[:, m - 1] - S[:, m], S[:, m] - S[:, m + 1])

def eps95(W, X, sd, mask, grid=np.geomspace(0.001, 1.0, 31)):
    ref = noisy(W, X, sd, 0, None)
    for e in grid:
        if np.mean([judge.same(ref, noisy(W, X, sd, e, np.random.default_rng(k)))[mask].mean() for k in range(3)]) < 0.95:
            return float(e)
    return 1.0

res = {}
for t in ["zoo_2nd_argmax_2_2", "zoo_2nd_argmax_4_3", "zoo_2nd_argmax_4_4", "zoo_2nd_argmax_8_5", "zoo_2nd_argmax_8_8", "zoo_2nd_argmax_16_10",
          "zoo_argmedian_8_5", "zoo_argmedian_16_5", "zoo_argmedian_16_7"]:
    meta, W, X = judge.load_target(t); X = np.asarray(X[:2000], dtype=np.float64); acts = {}; nets.forward("rnn_relu", W, X, acts=acts)
    sd = {s: np.asarray(acts[f"h{s}"]).reshape(len(X), -1).std(0) for s in range(1, X.shape[1] + 1)}
    mg = margin(X, "2nd" if "2nd" in t else "med"); clear = mg > np.median(mg); low = mg <= np.quantile(mg, 0.1)
    e_all = eps95(W, X, sd, np.ones(len(X), bool)); e_clear = eps95(W, X, sd, clear)
    ref = noisy(W, X, sd, 0, None); flips = np.zeros(len(X), bool)
    for k in range(3):
        flips |= ~judge.same(ref, noisy(W, X, sd, e_all, np.random.default_rng(k)))
    share_low = float(low[flips].mean()) if flips.any() else None
    res[t] = {"eps95_all": e_all, "eps95_clear_half": e_clear, "ratio": e_clear / e_all, "flips_in_lowest_decile": share_low}
    print(f"{t:22s} eps95 all {e_all:.4f} | clear half {e_clear:.4f} | ratio {e_clear/e_all:5.1f} | flips in lowest-margin decile {share_low:.2f}", flush=True)
(ROOT / "results/glass/units/u11_ties.json").write_text(json.dumps(res, indent=1))
