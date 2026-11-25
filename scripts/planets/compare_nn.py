"""Compare laws with the network on the same random-set systems (both metrics)."""
import numpy as np, sys, json, os
sys.path.insert(0, "scripts/planets")
from fit_law2 import FORMS, Q, DEV
P = os.path.expanduser("~/pdata")
def nn_table(name="random"):
    rows = np.loadtxt(f"{P}/nn_{name}_batch.csv", delimiter=",", ndmin=2)
    return rows[:, 0].astype(int), rows[:, 1]
def metrics(pred, y, stable):
    p = np.clip(pred, 4, 9); u = (~stable) & (y < 9) & (y >= 4)
    t = np.where(stable, 9.0, y); a = (y >= 4) | stable
    return {"unstable": float(np.sqrt(np.mean((p[u] - y[u]) ** 2))), "n_unstable": int(u.sum()),
            "with_stable": float(np.sqrt(np.mean((p[a] - t[a]) ** 2))), "n_all": int(a.sum())}
if __name__ == "__main__":
    idx, nn = nn_table()
    y, st = Q["y"][idx], Q["stable"][idx].astype(bool)
    print("network", metrics(nn, y, st))
    laws = json.load(open("results/planets/laws_round2.json"))
    sub = {k: v[idx] for k, v in Q.items() if v.shape == Q["y"].shape}
    for name in laws:
        c = np.array(laws[name]["params"]); k, fn = FORMS[name]
        print(name[:45], metrics(fn(c, sub), y, st))
