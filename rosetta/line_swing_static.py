"""Control for the line-swing result (registered in the DIARY before the run): is "what LD2's own line wins" just static
tactics? Static tactical totals per position, value-weighted (P1 N3 B3 R5 Q9): their pieces outnumbered (attacked by
more than defended), ours outnumbered, theirs attacked by a lower piece, ours attacked by a lower piece. (The textbook
concepts already hold hanging pieces, hanging value, captures available and the best free capture.) Fresh games:
textbook + static, textbook + swing, textbook + static + swing; bootstrap of the swing's addition over static.
Usage: python3 rosetta/line_swing_static.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.line_swing import swings  # noqa: E402
from rosetta.fresh_knowledge import outcome  # noqa: E402
from kepler.core.residual import surrogate  # noqa: E402
K = os.path.expanduser("~/kdata")
VALS = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}


def static(P):
    ix = {n: i for i, n in enumerate(L.names)}
    P = np.asarray(P, dtype=np.float32)
    vo = sum(v * P[:, :, ix["own_" + k]] for k, v in VALS.items()); vt = sum(v * P[:, :, ix["opp_" + k]] for k, v in VALS.items())
    return np.column_stack([(P[:, :, ix["outnumbered_opp"]] * vt).sum(1), (P[:, :, ix["outnumbered_own"]] * vo).sum(1),
                            (P[:, :, ix["opp_att_by_lower"]] * vt).sum(1), (P[:, :, ix["own_att_by_lower"]] * vo).sum(1)]).astype(np.float64)


if __name__ == "__main__":
    lines = json.load(open(os.path.join(L.R, "LD2_lines.json"))); fens = [it["fen"] for it in L.items]
    S = np.array([swings(f, l) for f, l in zip(fens, lines)], dtype=np.float64)
    fm = json.load(open(os.path.join(K, "chess", "rules", "fresh_meta.json")))["items"]; ffens = [it["fen"] for it in fm]
    fres = json.load(open(os.path.join(K, "chess", "rules", "fresh_results.json")))
    zl = np.load(os.path.join(L.R, "fresh_cache_partial.npz"), allow_pickle=True); flines = json.loads(str(zl["lines"]))
    Sf = np.array([swings(f, l) for f, l in zip(ffens, flines)], dtype=np.float64)
    z = np.load(os.path.join(L.R, "fresh_cache.npz"), allow_pickle=True); CXf, Pf = z["CX"], z["P"]
    St = np.concatenate([static(L.P[j:j + 2000]) for j in range(0, len(fens), 2000)]); Stf = static(Pf)
    yo = L.outcome(); g = np.where(~np.isnan(yo))[0]; yf = outcome(fres, ffens); gf = np.where(~np.isnan(yf))[0]; y = yf[gf]; v = np.var(y)
    fit = lambda A, Af: surrogate(np.column_stack([L.CX[g]] + [a[g] for a in A]), yo[g]).predict(np.column_stack([CXf[gf]] + [a[gf] for a in Af]))
    e0 = (y - surrogate(L.CX[g], yo[g]).predict(CXf[gf])) ** 2
    pr = {"static tactics": fit([St], [Stf]), "line swing": fit([S], [Sf]), "static tactics plus line swing": fit([St, S], [Stf, Sf])}
    E = {k: (y - p) ** 2 for k, p in pr.items()}
    rng = np.random.default_rng(0); B = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    out = {"gains": {}}
    for k, e in E.items():
        bt = [(e0[i].mean() - e[i].mean()) / np.var(y[i]) for i in B]
        out["gains"][k] = {"gain": round(float((e0.mean() - e.mean()) / v), 5), "ci95": [round(float(np.quantile(bt, .025)), 5), round(float(np.quantile(bt, .975)), 5)]}
    a, b = E["static tactics"], E["static tactics plus line swing"]
    bt = [(a[i].mean() - b[i].mean()) / np.var(y[i]) for i in B]
    out["swing_added_over_static"] = {"gain": round(float((a.mean() - b.mean()) / v), 5), "ci95": [round(float(np.quantile(bt, .025)), 5), round(float(np.quantile(bt, .975)), 5)]}
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open(f"{L.RES}/line_swing_static_s{L.SITE}.json", "w"), indent=1)
