"""PL1: baselines and law search for compact three-planet instability times (ledger row PL1).

Fit on the resonant dataset only; judge on the random dataset only. Metric: RMSE of log10(t/P1) over systems with
4 <= log10 t < 9 that are not stable (all predictions clipped to [4, 9] for every method).
"""
import json
import os
import sys

import numpy as np

PDATA = os.path.expanduser("~/pdata")
MEARTH = 3.0035e-6     # Earth / Sun mass ratio (star mass is 1 in these simulations)


def load(name):
    z = np.load(os.path.join(PDATA, f"{name}_ic.npz"))
    E = z["E"]                                  # n x 3 x [mu, a, e, inc, pomega, l, P, Omega]
    y = np.log10(z["t"])
    ys = np.log10(z["t_shadow"])
    return E, y, ys, z["stable"]


def pair_feats(E, i):
    mu1, mu2 = E[:, i, 0], E[:, i + 1, 0]
    a1, a2 = E[:, i, 1], E[:, i + 1, 1]
    e1, e2 = E[:, i, 2], E[:, i + 1, 2]
    w1, w2 = E[:, i, 4], E[:, i + 1, 4]
    Pr = E[:, i + 1, 6] / E[:, i, 6]
    ecross = (Pr ** (2 / 3) - 1) / (Pr ** (2 / 3) + 1)
    frac = (a2 - a1) / (a2 + a1)
    Z = np.abs(e2 * np.exp(1j * w2) - e1 * np.exp(1j * w1))          # relative eccentricity vector
    j1 = np.round(Pr / (Pr - 1))                                        # nearest first-order j:j-1
    d1 = Pr / (j1 / (j1 - 1)) - 1
    j2 = np.round(2 * Pr / (Pr - 1))                                    # nearest second-order j:j-2 (odd j)
    j2 = np.where(j2 % 2 == 0, j2 + np.sign(2 * Pr / (Pr - 1) - j2 + 1e-12), j2)
    d2 = Pr / (j2 / (j2 - 2)) - 1
    return {"mu1": mu1, "mu2": mu2, "Pr": Pr, "ecross": ecross, "frac": frac, "e1": e1, "e2": e2,
            "r1": e1 / ecross, "r2": e2 / ecross, "Z": Z, "rz": Z / ecross, "d1": d1, "d2": d2, "j1": j1}


def lhm_pair(f, A=11.9, B=-7.67, C=5.20, D=-3.26, mass="mu1", ecc="r1"):
    mu = f[mass] if isinstance(mass, str) else mass
    x = f["frac"] * mu ** -0.25
    r = f[ecc]
    return (A + B * r) * np.log10(np.maximum(x, 1e-6)) + C + D * r - np.log10(mu / MEARTH)


def rmse(p, y):
    return float(np.sqrt(np.mean((np.clip(p, 4, 9) - y) ** 2)))


def evaluation_mask(y, stable):
    return (y >= 4) & (y < 9) & (~stable)


def main():
    Er, yr, ysr, str_ = load("resonant")
    Et, yt, yst, stt = load("random")
    mr, mt = evaluation_mask(yr, str_), evaluation_mask(yt, stt)
    out = {"n_resonant_eval": int(mr.sum()), "n_random_eval": int(mt.sum())}
    # chaotic floor from shadow integrations: per-realization scatter
    for nm, y, ys, m in (("resonant", yr, ysr, mr), ("random", yt, yst, mt)):
        both = m & (ys >= 4) & (ys < 9)
        out[f"chaotic_floor_{nm}"] = float(np.std(y[both] - ys[both]) / np.sqrt(2))
    # LHM 2024 as published, per pair, system = minimum over the two adjacent pairs
    fr = [pair_feats(Er, 0), pair_feats(Er, 1)]
    ft = [pair_feats(Et, 0), pair_feats(Et, 1)]
    for nm, kw in (("LHM published (inner mass, inner e)", {}),):
        pt = np.minimum(lhm_pair(ft[0], **kw), lhm_pair(ft[1], **kw))
        pr = np.minimum(lhm_pair(fr[0], **kw), lhm_pair(fr[1], **kw))
        out[nm] = {"random_rmse": rmse(pt[mt], yt[mt]), "resonant_rmse": rmse(pr[mr], yr[mr])}
    out["constant (mean of resonant)"] = {"random_rmse": rmse(np.full(mt.sum(), yr[mr].mean()), yt[mt])}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
