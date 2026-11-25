"""Merge short-integration features (local + Colab CSVs) with the initial-condition
table and the network's predictions. Writes ~/pdata/short_{set}.npz."""
import os, glob, numpy as np
P = os.path.expanduser("~/pdata")
KEYS = ["EMcrossnear", "EMfracstdnear", "EPstdnear", "MMRstrengthnear", "EMcrossfar", "EMfracstdfar",
        "EPstdfar", "MMRstrengthfar", "MEGNO", "MEGNOstd", "Tsec", "short_stable"]
def rows():
    out = {"resonant": {}, "random": {}}
    for f in glob.glob(f"{P}/feat_random_*.csv"):
        for l in open(f):
            if l[:1].isdigit():
                p = l.strip().split(","); out["random"][int(p[0])] = [float(x) for x in p[1:]]
    for f in glob.glob(f"{P}/colab/*.csv"):
        for l in open(f):
            p = l.strip().split(",")
            if p[0] in out and len(p) == 14: out[p[0]][int(p[1])] = [float(x) for x in p[2:]]
    return out
if __name__ == "__main__":
    R = rows()
    for s, d in R.items():
        F = dict(np.load(f"{P}/{s}_F.npz"))
        idx = np.array(sorted(d)); V = np.array([d[i] for i in idx])
        T = {k: V[:, j] for j, k in enumerate(KEYS)}
        T.update({k: v[idx] for k, v in F.items() if v.shape == F["y"].shape}); T["i"] = idx
        if s == "random":
            nn = np.loadtxt(f"{P}/nn_random_batch.csv", delimiter=",", ndmin=2); m = dict(zip(nn[:, 0].astype(int), nn[:, 1]))
            T["nn"] = np.array([m.get(i, np.nan) for i in idx])
        np.savez(f"{P}/short_{s}.npz", **T); print(s, len(idx))
