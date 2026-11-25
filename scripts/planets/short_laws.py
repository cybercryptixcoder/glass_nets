"""Round 4: laws using physical summaries of the 1e4-orbit integration (the network's
own information budget). Fit on resonant (unstable, 4 <= y < 9), judge on the random
systems the network scored, both metrics, network on identical systems."""
import os, sys, json, numpy as np
from scipy.optimize import least_squares
P = os.path.expanduser("~/pdata")
def L(n):
    T = dict(np.load(f"{P}/short_{n}.npz"))
    T["su"] = np.isnan(T["MEGNO"]) | (T["short_stable"] == 0)        # went unstable inside the short integration
    meg = np.where(T["su"], 1e4, T["MEGNO"])
    T["lmeg"] = np.log10(np.maximum(meg - 2, 1e-2))                 # chaos excess: MEGNO - 2 ~ t / (2 Lyapunov time)
    T["lmegr"] = np.log10(np.maximum(meg, 1e-2))
    for k in ["EMfracstdnear", "EMfracstdfar", "EPstdnear", "EPstdfar", "EMcrossnear", "EMcrossfar", "Tsec", "MEGNOstd"]:
        T["l" + k] = np.log10(np.maximum(np.nan_to_num(T[k], nan=1e-8), 1e-8))
    for k in ["MMRstrengthnear", "MMRstrengthfar"]:
        T["l" + k] = np.log10(np.maximum(np.nan_to_num(T[k], nan=1e-8), 1e-8))
    T["u"] = (T["y"] >= 4) & (T["y"] < 9) & (~T["stable"].astype(bool))
    T["a"] = (T["y"] >= 4) | T["stable"].astype(bool)
    return T
def rm(p, y): return float(np.sqrt(np.mean((np.clip(p, 4, 9) - y) ** 2)))
def both(p, T):
    st = T["stable"].astype(bool); t = np.where(st, 9.0, T["y"])
    return round(rm(p[T["u"]], T["y"][T["u"]]), 4), round(rm(p[T["a"]], t[T["a"]]), 4)
if __name__ == "__main__":
    R, Q = L("resonant"), L("random")
    ok = ~np.isnan(Q["nn"])
    print("train n (unstable)", R["u"].sum(), "test n unstable", (Q["u"] & ok).sum(), "all", (Q["a"] & ok).sum())
    print("network", both(Q["nn"], Q))
    V = ["lmeg", "lmegr", "lEMfracstdnear", "lEMfracstdfar", "lEPstdnear", "lEPstdfar", "lEMcrossnear", "lEMcrossfar",
         "lTsec", "lMEGNOstd", "lMMRstrengthnear", "lMMRstrengthfar", "Xh", "Rm", "M"]
    u = R["u"] & ~R["su"]
    for v in V:
        print(f"  corr y {v:18s} resonant {np.corrcoef(R[v][u], R['y'][u])[0,1]:+.2f} random {np.corrcoef(Q[v][Q['u']], Q['y'][Q['u']])[0,1]:+.2f}")
    from sklearn.ensemble import HistGradientBoostingRegressor as H
    A = np.column_stack([R[v] for v in V]); B = np.column_stack([Q[v] for v in V])
    g = H(max_iter=300, learning_rate=0.05, random_state=0).fit(A[u], R["y"][u])
    print("tree ceiling (short + IC vars)", both(g.predict(B), Q))
    def lin(vs):
        A = np.column_stack([np.ones(u.sum())] + [R[v][u] for v in vs]); c, *_ = np.linalg.lstsq(A, R["y"][u], rcond=None)
        B = np.column_stack([np.ones(len(Q["y"]))] + [Q[v] for v in vs]); p = B @ c
        return c, np.where(Q["su"], np.median(R["y"][R["su"] & R["u"]]), p)
    for vs in [["lmeg"], ["lmeg", "Xh"], ["lmeg", "lEMfracstdnear"], ["lmeg", "lEMfracstdnear", "Xh"], ["lmeg", "lEMfracstdnear", "lEMcrossnear"],
               ["lmeg", "lEMfracstdnear", "lEPstdnear", "lEMcrossnear"], ["lmeg", "lEMfracstdnear", "lEMfracstdfar", "lEMcrossnear", "lEMcrossfar"],
               ["lmeg", "lEMfracstdnear", "lEMcrossnear", "Xh", "Rm"], ["lmegr", "lEMfracstdnear", "lEMcrossnear"]]:
        c, p = lin(vs); print(f"  linear {'+'.join(vs):60s}", both(p, Q), np.round(c, 2))
