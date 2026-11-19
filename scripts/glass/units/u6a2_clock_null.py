"""U6a': clock hypothesis with a shuffle null (300 shuffles per network)."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import judge
from scipy import stats

def err(Wh, n):
    base = 360.0 / n; best = 9.9
    for e in np.linalg.eigvals(Wh):
        if abs(e) >= 0.5 and abs(e.imag) > 1e-6:
            a = abs(np.degrees(np.angle(e)))
            best = min(best, min(abs(a - k * base) / base for k in range(1, n // 2 + 1)))
    return best

out = {}; ps = []
for t in ["zoo_2nd_argmax_4_3", "zoo_2nd_argmax_4_4", "zoo_2nd_argmax_8_5", "zoo_2nd_argmax_8_8", "zoo_2nd_argmax_16_10",
          "zoo_argmedian_8_5", "zoo_argmedian_8_7", "zoo_argmedian_16_5", "zoo_argmedian_16_7"]:
    meta, W, X = judge.load_target(t); n = X.shape[1]; e0 = err(W["W_h"], n)
    null = [err(judge.load_target(t, "shuffled", s)[1]["W_h"], n) for s in range(1, 301)]
    p = (1 + sum(x <= e0 for x in null)) / 301; ps.append(p)
    out[t] = {"n": n, "err": e0, "p": p, "null_median": float(np.median(null))}
    print(f"{t:22s} n={n:2d} err {100*e0:5.1f}% | null median {100*np.median(null):5.1f}% | p={p:.3f}", flush=True)
chi = -2 * np.sum(np.log(ps)); pf = 1 - stats.chi2.cdf(chi, 2 * len(ps))
print(f"networks with p<=0.05: {sum(p <= 0.05 for p in ps)}/9 | Fisher combined p = {pf:.2e}")
out["_fisher_p"] = pf; (ROOT / "results/glass/units/u6a2.json").write_text(json.dumps(out, indent=1))
