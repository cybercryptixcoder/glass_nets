"""Where does LD2's extra knowledge go? On the 16,000 fresh positions, predict game results from the textbook library
plus the board-wide totals of LD2's own variables (true SAE codes at block 10), using only the K variables its value
depends on most (attribution ranking from glass2.py, computed on training positions), for K = 8, 16, 64 and all 512.
Train on the 16,000 original game positions, score on the fresh ones, bootstrap over fresh positions; LD2's own value
is the ceiling. Exploratory (diary 06:2x): reported as shares, no pass line.
Usage: python3 rosetta/fresh_where.py"""
import json, os, sys, time
import chess, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.run_feats import sae, CV, CI  # noqa: E402
from rosetta.fresh_knowledge import outcome  # noqa: E402
from kepler.core.residual import surrogate  # noqa: E402
K = os.path.expanduser("~/kdata")


def totals(V, I, m=512):
    n = V.shape[0]
    idx = (np.arange(n)[:, None, None] * m + I.astype(np.int64)).reshape(-1)
    return np.bincount(idx, weights=V.reshape(-1).astype(np.float64), minlength=n * m).reshape(n, m)


if __name__ == "__main__":
    t0 = time.time()
    fm = json.load(open(os.path.join(K, "chess", "rules", "fresh_meta.json")))["items"]
    fres = json.load(open(os.path.join(K, "chess", "rules", "fresh_results.json")))
    fens = [it["fen"] for it in fm]
    z = np.load(os.path.join(L.R, "fresh_cache.npz"), allow_pickle=True); CXf, qf = z["CX"], z["q"]
    cache = os.path.join(L.R, "fresh_totals_s10.npy")
    if os.path.exists(cache):
        Tf = np.load(cache)
    else:
        m = L.get_net(); Tf = []
        with torch.no_grad():
            for j in range(0, len(fens), 500):
                x = torch.tensor(np.stack([L.encode(chess.Board(f)) for f in fens[j:j + 500]]))
                h = m(x, upto=L.SITE).flatten(2).numpy()                                   # [b, 128, 64]
                Vb = torch.tensor(h.transpose(0, 2, 1).reshape(-1, 128), dtype=torch.float32)
                v, i = sae.codes(Vb)
                Tf.append(totals(v.numpy().reshape(len(h), 64, -1), i.numpy().reshape(len(h), 64, -1)))
                if j % 4000 == 0:
                    print("fresh", j, round(time.time() - t0), "s", flush=True)
        Tf = np.concatenate(Tf); np.save(cache, Tf)
    To = totals(np.asarray(CV, dtype=np.float32), CI)                                       # [20000, 512] original items
    att = np.array(json.load(open(f"{L.RES}/glass2_attr.json"))["attribution"]); order = np.argsort(-att)
    yo = L.outcome(); g = np.where(~np.isnan(yo))[0]
    yf = outcome(fres, fens); gf = np.where(~np.isnan(yf))[0]; y = yf[gf]; v = np.var(y)
    base = surrogate(L.CX[g], yo[g]).predict(CXf[gf]); e0 = (y - base) ** 2
    preds = {"network value (ceiling)": surrogate(np.column_stack([L.CX[g], L.q[g]]), yo[g]).predict(np.column_stack([CXf[gf], qf[gf]]))}
    for Kt in (8, 16, 64, 512):
        cols = order[:Kt]
        preds[f"top {Kt} variables' board totals"] = surrogate(np.column_stack([L.CX[g], To[g][:, cols]]), yo[g]).predict(np.column_stack([CXf[gf], Tf[gf][:, cols]]))
        print(Kt, round(time.time() - t0), "s", flush=True)
    # Controls (added after the first result, 06:19): 16 variables drawn at random (3 draws), and the 16 least important.
    for sd in range(3):
        cols = np.random.default_rng(100 + sd).choice(512, 16, replace=False)
        preds[f"control: 16 random variables (draw {sd})"] = surrogate(np.column_stack([L.CX[g], To[g][:, cols]]), yo[g]).predict(np.column_stack([CXf[gf], Tf[gf][:, cols]]))
    cols = order[-16:]
    preds["control: 16 least important variables"] = surrogate(np.column_stack([L.CX[g], To[g][:, cols]]), yo[g]).predict(np.column_stack([CXf[gf], Tf[gf][:, cols]]))
    rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    out = {"n_fresh_positions": int(len(gf)), "textbook_r2_on_fresh": round(float(1 - e0.mean() / v), 4), "gains": {}}
    for k, p in preds.items():
        e = (y - p) ** 2
        bt = [(e0[i].mean() - e[i].mean()) / np.var(y[i]) for i in B]
        out["gains"][k] = {"gain": round(float((e0.mean() - e.mean()) / v), 5), "ci95": [round(float(np.quantile(bt, .025)), 5), round(float(np.quantile(bt, .975)), 5)]}
    ceil = out["gains"]["network value (ceiling)"]["gain"]
    for k in out["gains"]:
        out["gains"][k]["share_of_network_gain"] = round(out["gains"][k]["gain"] / ceil, 3) if ceil else None
    print(json.dumps(out, indent=1), round(time.time() - t0), "s", flush=True)
    json.dump(out, open(f"{L.RES}/fresh_where_s{L.SITE}.json", "w"), indent=1)
