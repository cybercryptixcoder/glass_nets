"""Round 4 (final): compact physical law using the 1e4-orbit integration's physical
summaries. Fit on resonant only; model choice on the random DEV half; FINAL half and
the whole network-scored set reported. Two fits of each form:
  primary  = unstable systems only (the pre-registered metric's population),
  censored = stable systems included, penalised only when predicted below 9.
Network compared on identical systems; paired bootstrap of the RMSE difference."""
import os, sys, json, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(__file__))
from short_laws import L
P = os.path.expanduser("~/pdata")
s_ = lambda T: np.minimum(T["Rm"], 1.0)
FORMS = {
 "F1 c0 + c1 Xh - c2 lmeg - c3 lEMf": (4, lambda c, T: c[0] + c[1] * T["Xh"] - c[2] * T["lmeg"] - c[3] * T["lEMfracstdnear"]),
 "F2 F1 + c4 s": (5, lambda c, T: c[0] + c[1] * T["Xh"] - c[2] * T["lmeg"] - c[3] * T["lEMfracstdnear"] + c[4] * s_(T)),
 "F3 c0 + (c1 + c2 s) Xh + c3 s - c4 lmeg - c5 lEMf": (6, lambda c, T: c[0] + (c[1] + c[2] * s_(T)) * T["Xh"] + c[3] * s_(T) - c[4] * T["lmeg"] - c[5] * T["lEMfracstdnear"]),
 "F4 F2 + c5 lEMcross_near": (6, lambda c, T: c[0] + c[1] * T["Xh"] - c[2] * T["lmeg"] - c[3] * T["lEMfracstdnear"] + c[4] * s_(T) + c[5] * T["lEMcrossnear"]),
 "F6 F2 + c5 M (free mass exponent of the spacing)": (6, lambda c, T: c[0] + c[1] * T["Xh"] - c[2] * T["lmeg"] - c[3] * T["lEMfracstdnear"] + c[4] * s_(T) + c[5] * T["M"]),
 "F5 F2 + c5 lMMRstrength_near": (6, lambda c, T: c[0] + c[1] * T["Xh"] - c[2] * T["lmeg"] - c[3] * T["lEMfracstdnear"] + c[4] * s_(T) + c[5] * T["lMMRstrengthnear"]),
}
def predict(fn, c, csu, T):
    return np.clip(np.where(T["su"], csu, fn(c, T)), 4, 9)
def fit(fn, k, T, censored):
    st = T["stable"].astype(bool)
    m = (T["u"] | (st & censored)) & ~T["su"]
    S = {q: v[m] for q, v in T.items() if hasattr(v, "shape") and v.shape == T["y"].shape}
    y, sm = S["y"], S["stable"].astype(bool)
    def res(c):
        p = np.clip(fn(c, S), 4, 9)
        return np.where(sm, np.minimum(p - 9, 0), p - y)
    best = None
    for c0 in [np.r_[6, 1.5, 0.4, 0.4, np.zeros(k - 4)], np.r_[7, 3, 0.3, 0.3, np.zeros(k - 4) - 0.5]]:
        r = least_squares(res, c0[:k], max_nfev=2000)
        if best is None or r.cost < best.cost: best = r
    csu = float(np.median(T["y"][T["su"] & T["u"]]))
    return best.x, csu
def rm(p, y): return float(np.sqrt(np.mean((p - y) ** 2)))
def evaluate(p, nn, T, m, B=2000, seed=0):
    st = T["stable"].astype(bool); t = np.where(st, 9.0, T["y"])
    u = T["u"] & m; a = T["a"] & m; nnc = np.clip(nn, 4, 9); out = {}
    for tag, s, tr in [("unstable", u, T["y"]), ("with_stable", a, t)]:
        i = np.where(s)[0]; rng = np.random.default_rng(seed); d = []
        for _ in range(B):
            j = rng.choice(i, len(i)); d.append(rm(p[j], tr[j]) - rm(nnc[j], tr[j]))
        out[tag] = {"n": int(len(i)), "law": round(rm(p[i], tr[i]), 4), "network": round(rm(nnc[i], tr[i]), 4),
                    "law_minus_network": round(rm(p[i], tr[i]) - rm(nnc[i], tr[i]), 4),
                    "ci95": [round(float(np.quantile(d, .025)), 4), round(float(np.quantile(d, .975)), 4)]}
    return out
if __name__ == "__main__":
    R, Q = L("resonant"), L("random")
    DEV = np.load(f"{P}/random_split.npy")[Q["i"]]
    ok = ~np.isnan(Q["nn"])
    res = {"n_train_resonant": int(R["a"].sum()), "n_train_unstable": int(R["u"].sum())}
    for name, (k, fn) in FORMS.items():
        for cens in (False, True):
            c, csu = fit(fn, k, R, cens)
            p = predict(fn, c, csu, Q)
            key = f"{name} | {'censored' if cens else 'primary'}"
            res[key] = {"params": np.round(c, 4).tolist(), "short_unstable_const": round(csu, 3),
                        "dev": evaluate(p, Q["nn"], Q, ok & DEV), "final": evaluate(p, Q["nn"], Q, ok & ~DEV), "all": evaluate(p, Q["nn"], Q, ok)}
            d, f, a = res[key]["dev"], res[key]["final"], res[key]["all"]
            print(f"{key:62s} dev u {d['unstable']['law']:.3f}/{d['unstable']['network']:.3f} s {d['with_stable']['law']:.3f}/{d['with_stable']['network']:.3f} | "
                  f"final u {f['unstable']['law']:.3f}/{f['unstable']['network']:.3f} s {f['with_stable']['law']:.3f}/{f['with_stable']['network']:.3f} | params {np.round(c, 2)}", flush=True)
    os.makedirs("results/planets", exist_ok=True)
    json.dump(res, open("results/planets/final_law.json", "w"), indent=1)
