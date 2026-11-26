"""Look-ahead profile across depth: at each site, can the network's state on a square tell whether that square is the
destination of the network's own move (ply 1), the opponent's reply (ply 2) or the network's follow-up (ply 3),
beyond what the board itself says? Probes: ridge on the 128-dim square vector; baseline: gradient-boosted trees on the
53 static board primitives; combined: both. Control: the untrained (weights-permuted) LD2.
Usage: python3 rosetta/probe_lookahead.py"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.engine import Defs  # noqa: E402
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    t0 = time.time()
    D = Defs(L.P, L.names, L.TR, L.TE, n_search=1500, n_score=1500)
    static = [k for k, n in enumerate(L.names) if not n.startswith("la_")]
    tg = {n: L.names.index(n) for n in ("la_m1_to", "la_m2_to", "la_m3_to")}
    Ps, Pt = np.asarray(L.P[D.rs], np.float32).reshape(-1, len(L.names)), np.asarray(L.P[D.rt], np.float32).reshape(-1, len(L.names))
    out = {"targets": list(tg), "sites": {}}
    base = {}
    for n, k in tg.items():
        g = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(Ps[:, static], Ps[:, k] > 0.5)
        base[n] = roc_auc_score(Pt[:, k] > 0.5, g.predict_proba(Pt[:, static])[:, 1])
    out["board_only_auc"] = {n: round(v, 4) for n, v in base.items()}
    print("board only", out["board_only_auc"], flush=True)
    mu = L.get_net(untrained_seed=21)
    for site in (2, 4, 6, 8, 10):
        A = np.load(os.path.join(L.C, "LD2", f"acts_site{site}.npy"), mmap_mode="r")
        Xs = np.asarray(A[D.rs], np.float32).reshape(len(D.rs), 128, 64).transpose(0, 2, 1).reshape(-1, 128)
        Xt = np.asarray(A[D.rt], np.float32).reshape(len(D.rt), 128, 64).transpose(0, 2, 1).reshape(-1, 128)
        Hu = L.site_acts(mu, np.concatenate([D.rs, D.rt]), site=site).reshape(-1, 128, 64).transpose(0, 2, 1).reshape(-1, 128)
        Us, Ut = Hu[:len(Xs)], Hu[len(Xs):]
        res = {}
        for n, k in tg.items():
            ys, yt = Ps[:, k] > 0.5, Pt[:, k] > 0.5
            def lr(a, b):
                mu_, sd = a.mean(0), a.std(0) + 1e-6
                m = LogisticRegression(max_iter=300, C=1.0).fit((a - mu_) / sd, ys)
                return roc_auc_score(yt, m.predict_proba((b - mu_) / sd)[:, 1])
            comb_s = np.column_stack([Xs, Ps[:, static]]); comb_t = np.column_stack([Xt, Pt[:, static]])
            g = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(comb_s, ys)
            res[n] = {"network_linear": round(lr(Xs, Xt), 4), "untrained_linear": round(lr(Us, Ut), 4),
                      "board_plus_network": round(roc_auc_score(yt, g.predict_proba(comb_t)[:, 1]), 4), "board_only": round(base[n], 4)}
        out["sites"][site] = res
        print("site", site, json.dumps(res), round(time.time() - t0), "s", flush=True)
    json.dump(out, open(f"{L.RES}/lookahead_profile.json", "w"), indent=1)
