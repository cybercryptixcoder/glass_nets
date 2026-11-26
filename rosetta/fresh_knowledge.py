"""Knowledge test on FRESH games: do Rosetta's translated variables predict who wins beyond the textbook library, on
16,000 positions from games never used anywhere in Rosetta (Kepler's fresh Lichess set)? Train on the 16,000 original
game positions; score on the fresh ones; bootstrap over fresh positions. Also LD2's own value as the ceiling.
Usage: python3 rosetta/fresh_knowledge.py [min_fidelity]"""
import glob, json, os, sys, time
import chess, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.engine import spec_map  # noqa: E402
from rosetta.chess_lib import build  # noqa: E402
from kepler.targets.chess.concepts import concepts  # noqa: E402
from kepler.targets.chess.leela import evaluate  # noqa: E402
from kepler.core.residual import surrogate  # noqa: E402
K = os.path.expanduser("~/kdata")


def outcome(results, fens):
    y = np.full(len(fens), np.nan)
    for i, (r, f) in enumerate(zip(results, fens)):
        if r in ("1-0", "0-1", "1/2-1/2"):
            s = 1.0 if r == "1-0" else 0.0 if r == "0-1" else 0.5
            y[i] = s if f.split()[1] == "w" else 1 - s
    return y


if __name__ == "__main__":
    t0 = time.time(); thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    fm = json.load(open(os.path.join(K, "chess", "rules", "fresh_meta.json")))["items"]
    fres = json.load(open(os.path.join(K, "chess", "rules", "fresh_results.json")))
    fens = [it["fen"] for it in fm]
    cache = os.path.join(L.R, "fresh_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); Pf, CXf, qf = z["P"], z["CX"], z["q"]
    else:
        net = L.get_net(); boards = [chess.Board(f) for f in fens]
        lines = [[] for _ in boards]; cur = boards; qf = None
        for ply in range(3):
            ev = evaluate(net, cur, batch=256)
            if ply == 0:
                qf = np.array([e["q"] for e in ev], dtype=np.float32)
            nxt = []
            for i, (b, e) in enumerate(zip(cur, ev)):
                if len(e["moves"]) == 0 or (ply > 0 and lines[i][-1] is None):
                    lines[i].append(None); nxt.append(b); continue
                mv = e["moves"][int(np.argmax(e["p"]))]; lines[i].append(mv.uci()); b2 = b.copy(stack=False); b2.push(mv); nxt.append(b2)
            cur = nxt
            print("ply", ply + 1, round(time.time() - t0), "s", flush=True)
        Pf = build(fens, lines, procs=2).astype(np.float16)
        CXf = np.array([concepts(chess.Board(f)) for f in fens], dtype=np.float64)
        np.savez(cache, P=Pf, CX=CXf, q=qf)
    print("fresh features ready", round(time.time() - t0), "s", flush=True)
    E = []
    for f in glob.glob(f"{L.RES}/dict_s{L.SITE}_w*.json"):
        E += [e for e in json.load(open(f)) if e.get("program") and abs(e["heldout_fidelity"]) >= thr]
    E.sort(key=lambda e: e["feature"])
    T = np.column_stack([spec_map(L.P, e["program"]).sum(1) for e in E]).astype(np.float64)
    Tf = np.column_stack([spec_map(Pf, e["program"]).sum(1) for e in E]).astype(np.float64)
    yo = L.outcome(); g = np.where(~np.isnan(yo))[0]                                  # all original game positions
    yf = outcome(fres, fens); gf = np.where(~np.isnan(yf))[0]
    base = surrogate(L.CX[g], yo[g]).predict(CXf[gf])
    aug = surrogate(np.column_stack([L.CX[g], T[g]]), yo[g]).predict(np.column_stack([CXf[gf], Tf[gf]]))
    netv = surrogate(np.column_stack([L.CX[g], L.q[g]]), yo[g]).predict(np.column_stack([CXf[gf], qf[gf]]))
    y = yf[gf]; e0, e1, e2 = (y - base) ** 2, (y - aug) ** 2, (y - netv) ** 2
    rng = np.random.default_rng(0); bt1, bt2 = [], []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y)); v = np.var(y[i])
        bt1.append((e0[i].mean() - e1[i].mean()) / v); bt2.append((e0[i].mean() - e2[i].mean()) / v)
    v = np.var(y)
    out = {"n_fresh_positions": int(len(gf)), "n_variables": len(E), "min_fidelity": thr,
           "textbook_r2_on_fresh": round(float(1 - e0.mean() / v), 4),
           "translated_variables_gain": round(float((e0.mean() - e1.mean()) / v), 5), "ci95": [round(float(np.quantile(bt1, .025)), 5), round(float(np.quantile(bt1, .975)), 5)],
           "network_value_gain": round(float((e0.mean() - e2.mean()) / v), 5), "network_ci95": [round(float(np.quantile(bt2, .025)), 5), round(float(np.quantile(bt2, .975)), 5)]}
    print(json.dumps(out), round(time.time() - t0), "s", flush=True)
    json.dump(out, open(f"{L.RES}/fresh_knowledge_s{L.SITE}.json", "w"), indent=1)
