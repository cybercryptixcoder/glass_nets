"""P3: precision requirement eps95 = relative hidden-state noise at which decision agreement falls to 95%."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import judge, nets

def run(W, X, eps, sd, rng):
    s = np.zeros((len(X), W["W_h"].shape[0]))
    for t in range(1, X.shape[1] + 1):
        s = np.maximum(X[:, t - 1:t] * W["W_x"] + s @ W["W_h"], 0)
        if eps > 0: s = s + rng.normal(size=s.shape) * eps * sd[t]
    return s @ W["W_o"]

res = {}
for target in sys.argv[1:] or ["zoo_2nd_argmax_2_2", "zoo_2nd_argmax_4_3", "zoo_2nd_argmax_4_4", "zoo_2nd_argmax_8_5", "zoo_2nd_argmax_8_8",
                               "zoo_2nd_argmax_16_10", "zoo_argmedian_8_5", "zoo_argmedian_16_5", "zoo_argmedian_16_7"]:
    meta, W, X = judge.load_target(target); X = X[:2000]
    acts = {}; nets.forward("rnn_relu", W, X, acts=acts)
    sd = {t: np.asarray(acts[f"h{t}"]).reshape(len(X), -1).std(0) for t in range(1, X.shape[1] + 1)}
    ref = run(W, X, 0, sd, None); curve = []
    for eps in np.geomspace(0.001, 1.0, 31):
        agr = np.mean([judge.same(ref, run(W, X, eps, sd, np.random.default_rng(k))).mean() for k in range(3)])
        curve.append((float(eps), float(agr)))
    e95 = next((e for e, a in curve if a < 0.95), 1.0)
    res[target] = {"eps95": e95, "curve": curve, "params": int(sum(v.size for k, v in W.items() if k != "seq_len"))}
    print(f"{target:22s} params {res[target]['params']:4d}  eps95 = {e95:.4f}", flush=True)
(ROOT / "results/glass/units/p3_precision.json").write_text(json.dumps(res, indent=1))
