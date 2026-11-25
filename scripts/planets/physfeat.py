"""PL1: physics feature table for every system (both datasets), saved once so fits are fast."""
import os, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from law import load, pair_feats, evaluation_mask, PDATA
def table(E):
    cols = {}
    for i in (0, 1):
        f = pair_feats(E, i)
        mus = f["mu1"] + f["mu2"]
        cols[f"X{i}"] = np.log10(f["frac"] * mus ** -0.25)          # spacing in units of mass^(1/4)
        cols[f"H{i}"] = np.log10(f["frac"] * mus ** (-1 / 3))       # spacing in Hill-like units
        cols[f"R{i}"] = f["rz"]                                     # relative eccentricity / crossing value
        cols[f"Ri{i}"] = f["r1"]; cols[f"Ro{i}"] = f["r2"]
        cols[f"M{i}"] = np.log10(mus)
        cols[f"P{i}"] = f["Pr"]
        cols[f"D{i}"] = np.abs(f["d1"]) / mus ** (2 / 3)           # distance to nearest first-order MMR, in mass^(2/3) units
        cols[f"d{i}"] = f["d1"]; cols[f"dd{i}"] = f["d2"]
    cols["inc"] = E[:, :, 3].max(1)
    return cols
for name in ("resonant", "random"):
    E, y, ys, st = load(name)
    c = table(E)
    np.savez(os.path.join(PDATA, f"{name}_phys.npz"), y=y, ys=ys, stable=st, **c)
    print(name, "saved", len(y))
