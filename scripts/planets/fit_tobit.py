"""Round 3: fit the law as the median of an intrinsic (chaotic) scatter model.
log10 t ~ Normal(mu(x), sigma), observed only for systems that survive 1e4 orbits
(truncation at 4, same selection the network gets from its own short integration),
right-censored at 9 (stable systems). Maximum likelihood on the resonant set
(unstable AND stable systems). Point prediction = median of y given y >= 4,
clipped to [4, 9], exactly as the network's median is clipped."""
import numpy as np, json, os, sys
from scipy.optimize import minimize
from scipy.stats import norm
sys.path.insert(0, "scripts/planets")
from fit_law2 import FORMS, Q, R, DEV, comb, sat
def sel(F):  # all systems that survived 1e4 orbits (unstable in [4,9) or stable)
    return (F["y"] >= 4) | F["stable"].astype(bool)
def sub(F, m): return {k: v[m] for k, v in F.items() if v.shape == F["y"].shape}
def nll(theta, fn, S, stable, y):
    c, ls = theta[:-1], theta[-1]; s = np.exp(ls); mu = fn(c, S)
    lt = norm.logsf((4 - mu) / s)
    lu = norm.logpdf((y - mu) / s) - ls
    lc = norm.logsf((9 - mu) / s)
    return -np.mean(np.where(stable, lc, lu) - lt)
def predict(fn, c, ls, S):
    mu = fn(c, S); s = np.exp(ls); a = (4 - mu) / s
    return np.clip(mu + s * norm.isf(0.5 * norm.sf(a)), 4, 9)
def fit(fn, k, F, init):
    m = sel(F); S = sub(F, m); st = F["stable"][m].astype(bool); y = np.where(st, 9.0, F["y"][m])
    best = None
    for c0 in init:
        th0 = np.r_[c0[:k], 0.2]
        r = minimize(nll, th0, args=(fn, S, st, y), method="L-BFGS-B", options={"maxiter": 500})
        if best is None or r.fun < best.fun: best = r
    return best.x[:-1], best.x[-1], best.fun
def metrics(pred, F, m):
    y, st = F["y"][m], F["stable"][m].astype(bool)
    u = (~st) & (y < 9) & (y >= 4); a = (y >= 4) | st; t = np.where(st, 9.0, y)
    return round(float(np.sqrt(np.mean((pred[u] - y[u]) ** 2))), 4), round(float(np.sqrt(np.mean((pred[a] - t[a]) ** 2))), 4)
if __name__ == "__main__":
    names = sys.argv[1:] or list(FORMS)
    old = json.load(open("results/planets/laws_round2.json"))
    out = {}
    for name in names:
        k, fn = FORMS[name]
        inits = [np.array(old[name]["params"]), np.r_[8, 6, -4.5, -2, np.ones(4) * 0.3]]
        c, ls, f = fit(fn, k, R, inits)
        mR = sel(R); mD = sel(Q) & DEV
        pr = predict(fn, c, ls, sub(R, mR)); pd = predict(fn, c, ls, sub(Q, mD))
        out[name] = {"params": c.round(4).tolist(), "sigma": round(float(np.exp(ls)), 4), "nll": f,
                     "resonant(unstable,with_stable)": metrics(pr, sub(R, mR), np.ones(mR.sum(), bool)),
                     "random_dev(unstable,with_stable)": metrics(pd, sub(Q, mD), np.ones(mD.sum(), bool))}
        print(name[:45], out[name], flush=True)
    json.dump(out, open("results/planets/laws_round3_tobit.json", "w"), indent=1)
