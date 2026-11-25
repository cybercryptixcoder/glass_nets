"""PL1: fit compact laws on the resonant dataset, judge them on the random dataset.

Each law gives a time per adjacent pair; the system's time is a soft minimum over the two pairs (the weakest pair
decides). Variables (all dimensionless):
  X = log10[(a2-a1)/(a2+a1) / mu^(1/4)]   spacing in units of the pair's mass ratio to the 1/4 (Petit 2020, LHM 2024)
  R = |e2 - e1| (vectors) / e_cross        how close the orbits are to crossing
  M = log10 mu                              pair mass ratio (m_i + m_i+1) / M_star
  D = |distance to nearest first-order MMR| / mu^(2/3)
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

P = os.path.expanduser("~/pdata")
OUT = "results/planets"


def data(name):
    D = dict(np.load(f"{P}/{name}_phys.npz"))
    m = (D["y"] >= 4) & (D["y"] < 9) & (~D["stable"])
    return D, m


def softmin(a, b, tau=0.25):
    lo = np.minimum(a, b)
    return lo - tau * np.log(np.exp(-(a - lo) / tau) + np.exp(-(b - lo) / tau))


LAWS = {
    # name: (pair function(c, D, i), n_params, initial)
    "L0 spacing only: c0 + c1 X": (lambda c, D, i: c[0] + c[1] * D[f"X{i}"], [5, 10]),
    "L1 LHM form: c0 + (c1 + c2 R) X + c3 R + c4 M": (
        lambda c, D, i: c[0] + (c[1] + c[2] * D[f"R{i}"]) * D[f"X{i}"] + c[3] * D[f"R{i}"] + c[4] * D[f"M{i}"], [5, 10, -5, -3, 0]),
    "L2 L1 + resonance: + c5 exp(-D/c6)": (
        lambda c, D, i: c[0] + (c[1] + c[2] * D[f"R{i}"]) * D[f"X{i}"] + c[3] * D[f"R{i}"] + c[4] * D[f"M{i}"]
        + c[5] * np.exp(-D[f"D{i}"] / abs(c[6])), [5, 10, -5, -3, 0, -1, 1]),
}


def tri(D):
    """Three-body combination (Petit et al. 2020): harmonic combination of the two pair spacings, in units of the
    triplet mass ratio to the 1/4."""
    f0 = 10 ** (D["X0"] + D["M0"] / 4)
    f1 = 10 ** (D["X1"] + D["M1"] / 4)
    h = f0 * f1 / (f0 + f1)
    mu = (10 ** D["M0"] + 10 ** D["M1"]) / 2
    return np.log10(h / mu ** 0.25), np.log10(mu)


def gaps(D, k):
    """Pair gaps reduced by eccentricity: frac * (1 - k R), floored; R = relative eccentricity / crossing value."""
    f0 = 10 ** (D["X0"] + D["M0"] / 4)
    f1 = 10 ** (D["X1"] + D["M1"] / 4)
    return f0 * np.clip(1 - k * D["R0"], 1e-3, None), f1 * np.clip(1 - k * D["R1"], 1e-3, None)


def xeff(D, k, beta=0.25, w=1.0):
    g0, g1 = gaps(D, k)
    h = (w * g0) * g1 / (w * g0 + g1)
    mu = (10 ** D["M0"] + 10 ** D["M1"]) / 2
    return np.log10(h / mu ** beta)


SYSTEM = {   # laws that act on the whole triplet directly (no min over pairs)
    "L6 effective gap: c0 + c1 log10[H(gap0, gap1) / mu^(1/4)], gap = frac (1 - c2 R)": (
        lambda c, D: c[0] + c[1] * xeff(D, c[2]), [8, 4, 1]),
    "L7 L6 with free mass exponent c3": (
        lambda c, D: c[0] + c[1] * xeff(D, c[2], c[3]), [8, 4, 1, 0.25]),
    "L8 L7 + curvature c4 x^2": (
        lambda c, D: c[0] + c[1] * xeff(D, c[2], c[3]) + c[4] * xeff(D, c[2], c[3]) ** 2, [8, 4, 1, 0.25, 0]),
    "L9 L7 + inner/outer weight c4": (
        lambda c, D: c[0] + c[1] * xeff(D, c[2], c[3], abs(c[4])) , [8, 4, 1, 0.25, 1]),
    "L3 three-body spacing: c0 + (c1 + c2 Rm) Xh + c3 Rm + c4 M": (
        lambda c, D: c[0] + (c[1] + c[2] * np.maximum(D["R0"], D["R1"])) * tri(D)[0]
        + c[3] * np.maximum(D["R0"], D["R1"]) + c[4] * tri(D)[1], [5, 10, -5, -3, 0]),
    "L4 three-body + both crossings: c0 + c1 Xh + c2 R0 + c3 R1 + c4 M + c5 Xh (R0+R1)": (
        lambda c, D: c[0] + c[1] * tri(D)[0] + c[2] * D["R0"] + c[3] * D["R1"] + c[4] * tri(D)[1]
        + c[5] * tri(D)[0] * (D["R0"] + D["R1"]), [5, 10, -2, -2, 0, -3]),
    "L5 three-body, crossing as log: c0 + c1 Xh + c2 log10(1 - Rm) + c3 M": (
        lambda c, D: c[0] + c[1] * tri(D)[0] + c[2] * np.log10(np.clip(1 - np.maximum(D["R0"], D["R1"]), 1e-3, 1)) + c[3] * tri(D)[1],
        [5, 10, 3, 0]),
}


def predict(fn, c, D):
    try:
        return softmin(fn(c, D, 0), fn(c, D, 1))
    except TypeError:
        return fn(c, D)


def rmse(p, y):
    return float(np.sqrt(np.mean((np.clip(p, 4, 9) - y) ** 2)))


def main(names=None):
    R, mr = data("resonant")
    T, mt = data("random")
    Rm = {k: v[mr] for k, v in R.items()}
    Tm = {k: v[mt] for k, v in T.items()}
    res = {}
    for name, (fn, c0) in {**LAWS, **SYSTEM}.items():
        if names and not any(name.startswith(n) for n in names):
            continue
        f = lambda c: np.clip(predict(fn, c, Rm), 4, 9) - Rm["y"]
        sol = least_squares(f, np.array(c0, float), loss="linear", max_nfev=4000)
        res[name] = {"params": [round(float(x), 4) for x in sol.x], "resonant_rmse": rmse(predict(fn, sol.x, Rm), Rm["y"]),
                     "random_rmse": rmse(predict(fn, sol.x, Tm), Tm["y"])}
        print(f"{name:55s} resonant {res[name]['resonant_rmse']:.3f}  random {res[name]['random_rmse']:.3f}  {res[name]['params']}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/laws_v1.json" if not names else f"{OUT}/laws_{'_'.join(names)}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
