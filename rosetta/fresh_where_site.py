"""Exploratory (RESULTS.md section 9): the few-variable test one block earlier. Train LD2's dictionary at block S (same
recipe), rank its variables by attribution to LD2's value (gradient taken through the remaining blocks via the edit
hook, 600 training positions), and compare the fresh-game gain of the top 16 board totals with block 10's.
Usage: S=8 python3 rosetta/fresh_where_site.py"""
import json, os, sys, time
import chess, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.sae import train  # noqa: E402
from rosetta.fresh_where import totals  # noqa: E402
from rosetta.fresh_knowledge import outcome  # noqa: E402
from kepler.core.residual import surrogate  # noqa: E402
K = os.path.expanduser("~/kdata"); S = int(os.environ.get("S", "8"))

if __name__ == "__main__":
    t0 = time.time(); torch.manual_seed(0); m = L.get_net()
    X = np.load(os.path.join(K, "chess", "LD2", f"acts_site{S}.npy"), mmap_mode="r")
    N = X.shape[0]; H = np.asarray(X, dtype=np.float32).reshape(N, 128, 64)
    V = H.transpose(0, 2, 1).reshape(-1, 128)
    sae = train(V, epochs=6)
    with torch.no_grad():
        vs, is_ = [], []
        for j in range(0, len(V), 65536):
            v, i = sae.codes(torch.tensor(V[j:j + 65536])); vs.append(v.numpy()); is_.append(i.numpy())
    To = totals(np.concatenate(vs).reshape(N, 64, -1), np.concatenate(is_).reshape(N, 64, -1))
    print("dictionary", round(time.time() - t0), "s", flush=True)
    tr = np.asarray(L.TR)[:600]; x = L.planes(tr)
    with torch.no_grad():
        hs = m(x, upto=S)
    hs = hs.detach().requires_grad_(True)
    _, wdl = m(x, edit=lambda site, h: hs if site == S else h)
    (wdl[:, 0] - wdl[:, 2]).sum().backward()
    g = hs.grad.detach().numpy().reshape(len(tr), 128, 64)
    with torch.no_grad():
        v, i = sae.codes(torch.tensor(hs.detach().numpy().reshape(len(tr), 128, 64).transpose(0, 2, 1).reshape(-1, 128)))
    Z = np.zeros((len(tr) * 64, 512), np.float32); np.put_along_axis(Z, i.numpy().astype(np.int64), v.numpy(), axis=1); Z = Z.reshape(len(tr), 64, 512)
    att = np.abs((Z * np.einsum("ncs,cf->nsf", g, sae.dec.weight.detach().numpy())).sum(1)).mean(0); order = np.argsort(-att)
    fm = json.load(open(os.path.join(K, "chess", "rules", "fresh_meta.json")))["items"]; fens = [it["fen"] for it in fm]
    fres = json.load(open(os.path.join(K, "chess", "rules", "fresh_results.json")))
    Tf = []
    with torch.no_grad():
        for j in range(0, len(fens), 500):
            hn = m(torch.tensor(np.stack([L.encode(chess.Board(f)) for f in fens[j:j + 500]])), upto=S).flatten(2).numpy()
            v, i = sae.codes(torch.tensor(hn.transpose(0, 2, 1).reshape(-1, 128), dtype=torch.float32))
            Tf.append(totals(v.numpy().reshape(len(hn), 64, -1), i.numpy().reshape(len(hn), 64, -1)))
    Tf = np.concatenate(Tf)
    z = np.load(os.path.join(L.R, "fresh_cache.npz"), allow_pickle=True); CXf, qf = z["CX"], z["q"]
    yo = L.outcome(); gi = np.where(~np.isnan(yo))[0]; yf = outcome(fres, fens); gf = np.where(~np.isnan(yf))[0]; y = yf[gf]; var = np.var(y)
    e0 = (y - surrogate(L.CX[gi], yo[gi]).predict(CXf[gf])) ** 2
    fit = lambda A, Af: surrogate(np.column_stack([L.CX[gi], A[gi]]), yo[gi]).predict(np.column_stack([CXf[gf], Af[gf]]))
    ceil = float((e0.mean() - ((y - fit(L.q[:, None], qf[:, None])) ** 2).mean()) / var)
    out = {"site": S, "network_value_gain": round(ceil, 5), "top": {}}
    for Kt in (8, 16, 64):
        p = fit(To[:, order[:Kt]], Tf[:, order[:Kt]]); gk = float((e0.mean() - ((y - p) ** 2).mean()) / var)
        out["top"][str(Kt)] = {"gain": round(gk, 5), "share_of_network_gain": round(gk / ceil, 3)}
    print(out, round(time.time() - t0), "s", flush=True)
    json.dump(out, open(f"{L.RES}/fresh_where_LD2_s{S}.json", "w"), indent=1)
