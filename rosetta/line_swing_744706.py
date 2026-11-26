"""Replication of the line-swing result in 744706 (registered in the DIARY before the run): its own 3-ply greedy line on
the original positions (744706_lines.json) and on the 16,000 fresh positions (computed here), its own value as the
ceiling, the same random-move and static-tactics controls. Usage: python3 rosetta/line_swing_744706.py"""
import json, os, sys, time
import chess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.line_swing import swings  # noqa: E402
from rosetta.line_swing_static import static  # noqa: E402
from rosetta.fresh_knowledge import outcome  # noqa: E402
from kepler.targets.chess.leela import evaluate  # noqa: E402
from kepler.core.residual import surrogate  # noqa: E402
K = os.path.expanduser("~/kdata")

if __name__ == "__main__":
    t0 = time.time(); net = L.get_net("744706")
    lines = json.load(open(os.path.join(L.R, "744706_lines.json"))); fens = [it["fen"] for it in L.items]
    S = np.array([swings(f, l) for f, l in zip(fens, lines)], dtype=np.float64)
    q2 = L.value(net, np.arange(len(fens)))
    fm = json.load(open(os.path.join(K, "chess", "rules", "fresh_meta.json")))["items"]; ffens = [it["fen"] for it in fm]
    fres = json.load(open(os.path.join(K, "chess", "rules", "fresh_results.json")))
    boards = [chess.Board(f) for f in ffens]; flines = [[] for _ in boards]; cur = boards; qf = None
    for ply in range(3):
        ev = evaluate(net, cur, batch=256)
        if ply == 0:
            qf = np.array([e["q"] for e in ev], dtype=np.float64)
        nxt = []
        for i, (b, e) in enumerate(zip(cur, ev)):
            if len(e["moves"]) == 0 or (ply > 0 and flines[i][-1] is None):
                flines[i].append(None); nxt.append(b); continue
            mv = e["moves"][int(np.argmax(e["p"]))]; flines[i].append(mv.uci()); b2 = b.copy(stack=False); b2.push(mv); nxt.append(b2)
        cur = nxt; print("ply", ply + 1, round(time.time() - t0), "s", flush=True)
    json.dump(flines, open(os.path.join(L.R, "744706_fresh_lines.json"), "w"))
    Sf = np.array([swings(f, l) for f, l in zip(ffens, flines)], dtype=np.float64)
    rng = np.random.default_rng(0)
    Sr = np.array([swings(f, [], rng) for f in fens], dtype=np.float64); Srf = np.array([swings(f, [], rng) for f in ffens], dtype=np.float64)
    z = np.load(os.path.join(L.R, "fresh_cache.npz"), allow_pickle=True); CXf, Pf = z["CX"], z["P"]
    St = np.concatenate([static(L.P[j:j + 2000]) for j in range(0, len(fens), 2000)]); Stf = static(Pf)
    yo = L.outcome(); g = np.where(~np.isnan(yo))[0]; yf = outcome(fres, ffens); gf = np.where(~np.isnan(yf))[0]; y = yf[gf]; v = np.var(y)
    fit = lambda A, Af: surrogate(np.column_stack([L.CX[g]] + [a[g] for a in A]), yo[g]).predict(np.column_stack([CXf[gf]] + [a[gf] for a in Af]))
    e0 = (y - surrogate(L.CX[g], yo[g]).predict(CXf[gf])) ** 2
    pr = {"network value (ceiling)": fit([q2[:, None]], [qf[:, None]]), "material won along 744706's own line": fit([S], [Sf]),
          "control: along random legal moves": fit([Sr], [Srf]), "static tactics": fit([St], [Stf]), "static tactics plus line swing": fit([St, S], [Stf, Sf])}
    E = {k: (y - p) ** 2 for k, p in pr.items()}
    rb = np.random.default_rng(0); B = [rb.integers(0, len(y), len(y)) for _ in range(1000)]
    out = {"network": "744706", "n_fresh_positions": int(len(gf)), "gains": {}}
    for k, e in E.items():
        bt = [(e0[i].mean() - e[i].mean()) / np.var(y[i]) for i in B]
        out["gains"][k] = {"gain": round(float((e0.mean() - e.mean()) / v), 5), "ci95": [round(float(np.quantile(bt, .025)), 5), round(float(np.quantile(bt, .975)), 5)]}
    a, b = E["static tactics"], E["static tactics plus line swing"]
    bt = [(a[i].mean() - b[i].mean()) / np.var(y[i]) for i in B]
    out["swing_added_over_static"] = {"gain": round(float((a.mean() - b.mean()) / v), 5), "ci95": [round(float(np.quantile(bt, .025)), 5), round(float(np.quantile(bt, .975)), 5)]}
    ceil = out["gains"]["network value (ceiling)"]["gain"]
    for k, d in out["gains"].items():
        d["share_of_network_gain"] = round(d["gain"] / ceil, 3)
    print(json.dumps(out, indent=1), round(time.time() - t0), "s", flush=True)
    json.dump(out, open(f"{L.RES}/line_swing_744706_s{L.SITE}.json", "w"), indent=1)
