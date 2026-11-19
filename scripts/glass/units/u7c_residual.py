"""U7c: residual units in transformers (UNITS_PROGRAM §8 U7 spec, second part; pre-registered in DIARY).

For each attention layer's write attn{l}.out at each position i:
  1. the decision-relevant part = the top-k principal components that keep >= 95% of decisions (others set to the mean);
  2. the held-out R² of those components on categorical features: first order (the token at every position) and
     relational (fixed point, pointer read x_{x_i}, 2-cycle, preimage count of i, same-token count of x_i).
Also ε95 on resid0.mid: the relative noise at which decision agreement falls to 95%."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import judge, nets


def feats(X, i):
    B, P = X.shape; xi = X[:, i]
    ptr = np.where(xi < P, X[np.arange(B), np.clip(xi, 0, P - 1)], -1)
    base = [X[:, k] for k in range(P)]
    rel = [(xi == i).astype(int), ptr, (ptr == i).astype(int), (X == i).sum(1), (X == xi[:, None]).sum(1)]
    return base, rel


def design(cols):
    mats = [np.ones((len(cols[0]), 1))]
    for c in cols:
        v = np.unique(c); mats.append((c[:, None] == v[None, 1:]).astype(float))
    return np.hstack(mats)


def heldout_sse(D, Y, rng):
    idx = rng.permutation(len(Y)); folds = (idx[: len(Y) // 2], idx[len(Y) // 2:]); pred = np.zeros_like(Y)
    for tr, te in (folds, folds[::-1]):
        coef = np.linalg.lstsq(D[tr], Y[tr], rcond=None)[0]; pred[te] = D[te] @ coef
    return float(((Y - pred) ** 2).sum())


def analyse(target, variant, n=6000):
    meta, W, X = judge.load_target(target, variant, 0); judge.GROUPS = meta.get("out_groups")
    X = np.asarray(X[:n]).astype(int); B, P = X.shape; kind = meta["forward"]
    acts = {}; ref = nets.forward(kind, W, X, acts=acts); d = W["W_E"].shape[1]
    out = {"target": target, "variant": variant, "layers": {}}
    for site in sorted(s for s in acts if s.endswith(".out") and s.startswith("attn")):
        A = np.asarray(acts[site]).reshape(B, P, d); cells = []; tot = {"sst": 0.0, "base": 0.0, "full": 0.0}
        for i in range(P):
            a = A[:, i]; mu = a.mean(0); Vt = np.linalg.svd(a - mu, full_matrices=False)[2]
            for k in range(d + 1):
                def f(z, k=k):
                    z = z.reshape(B, P, d).copy(); z[:, i] = mu + ((z[:, i] - mu) @ Vt[:k].T) @ Vt[:k]; return z.reshape(B, -1)
                if judge.same(ref, nets.forward(kind, W, X, hook={site: f})).mean() >= 0.95:
                    break
            if k == 0:
                continue
            Y = (a - mu) @ Vt[:k].T; sst = float((Y ** 2).sum()); base, rel = feats(X, i)
            sb = heldout_sse(design(base), Y, np.random.default_rng(0)); sf = heldout_sse(design(base + rel), Y, np.random.default_rng(0))
            cells.append({"pos": i, "k": k, "r2_base": 1 - sb / sst, "r2_full": 1 - sf / sst})
            tot["sst"] += sst; tot["base"] += sb; tot["full"] += sf
        if cells:
            rb, rf = 1 - tot["base"] / tot["sst"], 1 - tot["full"] / tot["sst"]
            out["layers"][site] = {"cells": cells, "r2_base": rb, "r2_full": rf, "relational_share": (rf - rb) / (1 - rb) if 1 - rb >= 0.01 else None}
    # ε95 on resid0.mid
    sd = np.asarray(acts["resid0.mid"]).std(0)
    for e in np.geomspace(0.001, 1.0, 31):
        agree = np.mean([judge.same(ref, nets.forward(kind, W, X, hook={"resid0.mid": lambda z, r=np.random.default_rng(s): z + r.normal(size=z.shape) * e * sd})).mean() for s in range(3)])
        if agree < 0.95:
            break
    out["eps95_resid"] = float(e) if agree < 0.95 else 1.0
    return out


if __name__ == "__main__":
    res = {}
    for t in sys.argv[1:] or ["tracr_hist", "tracr_reverse", "lc_3_4", "lc_4_5", "lc_6_5", "lc_8_6"]:
        for var in (["original"] if t.startswith("tracr") else ["original", "shuffled"]):
            r = analyse(t, var); res[f"{t}|{var}"] = r
            print(f"{t:13s} {var:9s} eps95_resid {r['eps95_resid']:.3f} |", "; ".join(
                f"{s}: base {v['r2_base']:.2f} full {v['r2_full']:.2f} rel {v['relational_share'] if v['relational_share'] is None else round(v['relational_share'], 2)} k={[c['k'] for c in v['cells']]}"
                for s, v in r["layers"].items()), flush=True)
    (ROOT / "results/glass/units/u7c.json").write_text(json.dumps(res, indent=1))
