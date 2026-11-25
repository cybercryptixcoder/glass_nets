"""Law search, round 2. Finding from the binned table (DIARY 20:55): the slope of
log t in the three-body spacing falls as the relative eccentricity approaches orbit
crossing, and saturates once orbits cross (R >= 1). Forms below use a saturating
eccentricity s = min(R, 1). Fit on resonant only. Model selection uses the random
DEV half only; the random FINAL half is scored once at the end (split saved)."""
import numpy as np, json, os, sys
from scipy.optimize import least_squares
P = os.path.expanduser("~/pdata")
def load(n):
    F = dict(np.load(f"{P}/{n}_F.npz")); return F
R, Q = load("resonant"), load("random")
mq = Q["m"]
if not os.path.exists(f"{P}/random_split.npy"):
    perm = np.random.default_rng(2026).permutation(len(Q["y"]))
    dev = np.zeros(len(Q["y"]), bool); dev[perm[: len(perm) // 2]] = True
    np.save(f"{P}/random_split.npy", dev)
DEV = np.load(f"{P}/random_split.npy")
def sat(x, k=None): return np.minimum(x, 1.0)
def comb(F, how):
    a, b = F["R0"], F["R1"]
    if how == "max": return np.maximum(a, b)
    if how == "rms": return np.sqrt((a * a + b * b) / 2)
    if how == "sum": return a + b
    if how == "mean": return (a + b) / 2
FORMS = {
 "A  c0+(c1+c2 s)Xh+c3 s, s=min(Rmax,1)": (4, lambda c, F: c[0] + (c[1] + c[2] * sat(comb(F, "max"))) * F["Xh"] + c[3] * sat(comb(F, "max"))),
 "B  as A with s=min(Rrms,1)": (4, lambda c, F: c[0] + (c[1] + c[2] * sat(comb(F, "rms"))) * F["Xh"] + c[3] * sat(comb(F, "rms"))),
 "C  as A with s=min(Rmean,1)": (4, lambda c, F: c[0] + (c[1] + c[2] * sat(comb(F, "mean"))) * F["Xh"] + c[3] * sat(comb(F, "mean"))),
 "D  as A, separate s0,s1": (6, lambda c, F: c[0] + (c[1] + c[2] * sat(F["R0"]) + c[3] * sat(F["R1"])) * F["Xh"] + c[4] * sat(F["R0"]) + c[5] * sat(F["R1"])),
 "E  as A + mass c4 M": (5, lambda c, F: c[0] + (c[1] + c[2] * sat(comb(F, "max"))) * F["Xh"] + c[3] * sat(comb(F, "max")) + c[4] * F["M"]),
 "F  smooth sat s=R/(R+c4)": (5, lambda c, F: (lambda s: c[0] + (c[1] + c[2] * s) * F["Xh"] + c[3] * s)(comb(F, "max") / (comb(F, "max") + abs(c[4])))),
 "G  power sat s=min(R,1)^c4": (5, lambda c, F: (lambda s: c[0] + (c[1] + c[2] * s) * F["Xh"] + c[3] * s)(sat(comb(F, "max")) ** abs(c[4]))),
 "H  D + separate pair spacings via Xh weight": (6, lambda c, F: c[0] + (c[1] + c[2] * sat(comb(F, "max"))) * F["Xh"] + c[3] * sat(comb(F, "max")) + c[4] * (F["X0"] - F["X1"]) ** 2 + c[5] * (F["X0"] - F["X1"])),
}
def fit(fn, k, F, mask):
    y = F["y"][mask]
    sub = {key: v[mask] for key, v in F.items() if v.shape == F["y"].shape}
    res = lambda c: np.clip(fn(c, sub), 4, 9) - y
    best = None
    for s0 in [np.r_[7, 4, -3, -1, np.ones(k - 4) * 0.5], np.r_[6, 3, -2, -0.5, np.ones(k - 4) * 0.1]]:
        r = least_squares(res, s0[:k], loss="linear", max_nfev=400)
        if best is None or r.cost < best.cost: best = r
    return best.x
def score(fn, c, F, mask):
    sub = {key: v[mask] for key, v in F.items() if v.shape == F["y"].shape}
    return float(np.sqrt(np.mean((np.clip(fn(c, sub), 4, 9) - F["y"][mask]) ** 2)))
if __name__ == "__main__":
    out = {}
    for name, (k, fn) in FORMS.items():
        c = fit(fn, k, R, R["m"])
        out[name] = {"params": c.round(4).tolist(), "resonant": score(fn, c, R, R["m"]), "random_dev": score(fn, c, Q, mq & DEV)}
        print(f"{name:50s} k={k} res {out[name]['resonant']:.3f} dev {out[name]['random_dev']:.3f}  {np.round(c, 3)}", flush=True)
    os.makedirs("results/planets", exist_ok=True)
    json.dump(out, open("results/planets/laws_round2.json", "w"), indent=1)
