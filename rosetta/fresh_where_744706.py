"""Replication of RESULTS.md section 9 in the second network 744706 (registered in the DIARY before the run): train its
sparse dictionary at block 10 (same recipe as LD2's: 512 variables, k = 8) on the original positions, rank variables by
attribution to 744706's own value (600 training positions), and on the 16,000 fresh positions compare textbook concepts
plus the board totals of the top 8 / 16 / 64 / all variables against 744706's own value (ceiling), with three random
16-variable draws and the 16 least important as controls. Usage: python3 rosetta/fresh_where_744706.py"""
import json, os, sys, time
import chess, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.sae import train  # noqa: E402
from rosetta.fresh_where import totals  # noqa: E402
from rosetta.fresh_knowledge import outcome  # noqa: E402
from kepler.core.residual import surrogate  # noqa: E402
K = os.path.expanduser("~/kdata")

if __name__ == "__main__":
    t0 = time.time(); torch.manual_seed(0)
    m = L.get_net("744706"); N = len(L.items); rows = np.arange(N)
    H = L.site_acts(m, rows).reshape(N, 128, 64); V = H.transpose(0, 2, 1).reshape(-1, 128).astype(np.float32)
    print("acts", round(time.time() - t0), "s", flush=True)
    sae = train(V, epochs=6)
    with torch.no_grad():
        vs, is_ = [], []
        for j in range(0, len(V), 65536):
            v, i = sae.codes(torch.tensor(V[j:j + 65536])); vs.append(v.numpy()); is_.append(i.numpy())
    To = totals(np.concatenate(vs).reshape(N, 64, -1), np.concatenate(is_).reshape(N, 64, -1))
    print("dictionary", round(time.time() - t0), "s", flush=True)
    # attribution to 744706's value on 600 training positions
    tr = np.asarray(L.TR)[:600]
    h = m(L.planes(tr), upto=L.SITE).detach().requires_grad_(True)
    _, wdl = m.heads(h); (wdl[:, 0] - wdl[:, 2]).sum().backward()
    g = h.grad.detach().numpy().reshape(len(tr), 128, 64)
    Vt = torch.tensor(H[tr].transpose(0, 2, 1).reshape(-1, 128).astype(np.float32))
    with torch.no_grad():
        v, i = sae.codes(Vt)
    Z = np.zeros((len(Vt), 512), np.float32); np.put_along_axis(Z, i.numpy().astype(np.int64), v.numpy(), axis=1); Z = Z.reshape(len(tr), 64, 512)
    proj = np.einsum("ncs,cf->nsf", g, sae.dec.weight.detach().numpy())
    att = np.abs((Z * proj).sum(1)).mean(0); order = np.argsort(-att)
    # the network's own value on original and fresh positions; fresh totals
    q2 = L.value(m, rows)
    fm = json.load(open(os.path.join(K, "chess", "rules", "fresh_meta.json")))["items"]
    fres = json.load(open(os.path.join(K, "chess", "rules", "fresh_results.json")))
    fens = [it["fen"] for it in fm]
    z = np.load(os.path.join(L.R, "fresh_cache.npz"), allow_pickle=True); CXf = z["CX"]
    Tf, qf = [], []
    with torch.no_grad():
        for j in range(0, len(fens), 500):
            x = torch.tensor(np.stack([L.encode(chess.Board(f)) for f in fens[j:j + 500]]))
            hb = m(x, upto=L.SITE); _, wdl = m.heads(hb); qf.append((wdl[:, 0] - wdl[:, 2]).numpy())
            hn = hb.flatten(2).numpy(); v, i = sae.codes(torch.tensor(hn.transpose(0, 2, 1).reshape(-1, 128), dtype=torch.float32))
            Tf.append(totals(v.numpy().reshape(len(hn), 64, -1), i.numpy().reshape(len(hn), 64, -1)))
    Tf = np.concatenate(Tf); qf = np.concatenate(qf)
    print("fresh", round(time.time() - t0), "s", flush=True)
    yo = L.outcome(); gi = np.where(~np.isnan(yo))[0]
    yf = outcome(fres, fens); gf = np.where(~np.isnan(yf))[0]; y = yf[gf]; var = np.var(y)
    e0 = (y - surrogate(L.CX[gi], yo[gi]).predict(CXf[gf])) ** 2
    fit = lambda A, Af: surrogate(np.column_stack([L.CX[gi], A[gi]]), yo[gi]).predict(np.column_stack([CXf[gf], Af[gf]]))
    preds = {"network value (ceiling)": fit(q2[:, None], qf[:, None])}
    for Kt in (8, 16, 64, 512):
        preds[f"top {Kt} variables' board totals"] = fit(To[:, order[:Kt]], Tf[:, order[:Kt]])
    for sd in range(3):
        c = np.random.default_rng(100 + sd).choice(512, 16, replace=False)
        preds[f"control: 16 random variables (draw {sd})"] = fit(To[:, c], Tf[:, c])
    preds["control: 16 least important variables"] = fit(To[:, order[-16:]], Tf[:, order[-16:]])
    rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    out = {"network": "744706", "n_fresh_positions": int(len(gf)), "gains": {}}
    for k, p in preds.items():
        e = (y - p) ** 2; bt = [(e0[i].mean() - e[i].mean()) / np.var(y[i]) for i in B]
        out["gains"][k] = {"gain": round(float((e0.mean() - e.mean()) / var), 5), "ci95": [round(float(np.quantile(bt, .025)), 5), round(float(np.quantile(bt, .975)), 5)]}
    ceil = out["gains"]["network value (ceiling)"]["gain"]
    for k, d in out["gains"].items():
        d["share_of_network_gain"] = round(d["gain"] / ceil, 3) if ceil else None
    print(json.dumps(out, indent=1), round(time.time() - t0), "s", flush=True)
    json.dump(out, open(f"{L.RES}/fresh_where_744706_s{L.SITE}.json", "w"), indent=1)
