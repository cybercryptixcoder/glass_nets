"""Hypothesis (06:22, registered in the diary before the run): the evaluation variables Rosetta cannot define carry what
LD2's own forcing line wins. Material swing (side to move's material balance after 1, 2 and 3 plies of LD2's own greedy
line, minus now; P1 N3 B3 R5 Q9). (1) On held-out positions: correlation of each of the 16 most important variables'
board totals with the swing, raw and after removing what the 61 textbook concepts predict (linear fit on training).
(2) On the 16,000 fresh positions: textbook concepts plus the three swings, against LD2's own value (ceiling), trained
on the original games. Control for (2): the same swings along a line of random legal moves (seeded).
Usage: python3 rosetta/line_swing.py"""
import json, os, sys
import chess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.run_feats import CV, CI  # noqa: E402
from rosetta.fresh_where import totals  # noqa: E402
from rosetta.fresh_knowledge import outcome  # noqa: E402
from kepler.core.residual import surrogate  # noqa: E402
K = os.path.expanduser("~/kdata")
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def mat(b, side):
    return sum(VAL[p.piece_type] * (1 if p.color == side else -1) for p in b.piece_map().values())


def swings(fen, line, rng=None):
    b = chess.Board(fen); side = b.turn; m0 = mat(b, side); out = []
    for ply in range(3):
        mv = None
        if rng is not None:
            lm = list(b.legal_moves)
            mv = lm[rng.integers(len(lm))] if lm else None
        elif ply < len(line) and line[ply]:
            try:
                mv = chess.Move.from_uci(line[ply]); mv = mv if mv in b.legal_moves else None
            except ValueError:
                mv = None
        if mv is not None:
            b.push(mv)
        out.append(mat(b, side) - m0)
    return out


if __name__ == "__main__":
    lines = json.load(open(os.path.join(L.R, "LD2_lines.json")))
    fens = [it["fen"] for it in L.items]
    S = np.array([swings(f, l) for f, l in zip(fens, lines)], dtype=np.float64)                  # [20000, 3]
    To = totals(np.asarray(CV, dtype=np.float32), CI)
    att = np.array(json.load(open(f"{L.RES}/glass2_attr.json"))["attribution"]); order = np.argsort(-att)
    TR, TE = np.asarray(L.TR), np.asarray(L.TE)
    X = np.column_stack([np.ones(len(L.CX)), L.CX])
    out = {"part1_heldout_correlations": []}
    for f in order[:16]:
        t = To[:, int(f)]
        w, *_ = np.linalg.lstsq(X[TR], t[TR], rcond=None); rt = t - X @ w                          # what textbook concepts miss
        ws, *_ = np.linalg.lstsq(X[TR], S[TR, 2], rcond=None); rs = S[:, 2] - X @ ws
        r = {"feature": int(f), "corr_with_swing3": round(float(np.corrcoef(t[TE], S[TE, 2])[0, 1]), 3),
             "corr_after_textbook_removed": round(float(np.corrcoef(rt[TE], rs[TE])[0, 1]), 3)}
        out["part1_heldout_correlations"].append(r); print(r, flush=True)
    # part 2: fresh games
    fm = json.load(open(os.path.join(K, "chess", "rules", "fresh_meta.json")))["items"]
    fres = json.load(open(os.path.join(K, "chess", "rules", "fresh_results.json")))
    ffens = [it["fen"] for it in fm]
    z = np.load(os.path.join(L.R, "fresh_cache.npz"), allow_pickle=True); CXf, qf = z["CX"], z["q"]
    zl = np.load(os.path.join(L.R, "fresh_cache_partial.npz"), allow_pickle=True); flines = json.loads(str(zl["lines"]))
    Sf = np.array([swings(f, l) for f, l in zip(ffens, flines)], dtype=np.float64)
    rng = np.random.default_rng(0)
    Sr = np.array([swings(f, [], rng) for f in fens], dtype=np.float64); Srf = np.array([swings(f, [], rng) for f in ffens], dtype=np.float64)
    yo = L.outcome(); g = np.where(~np.isnan(yo))[0]
    yf = outcome(fres, ffens); gf = np.where(~np.isnan(yf))[0]; y = yf[gf]; v = np.var(y)
    base = surrogate(L.CX[g], yo[g]).predict(CXf[gf]); e0 = (y - base) ** 2
    preds = {"network value (ceiling)": surrogate(np.column_stack([L.CX[g], L.q[g]]), yo[g]).predict(np.column_stack([CXf[gf], qf[gf]])),
             "material won along LD2's own line": surrogate(np.column_stack([L.CX[g], S[g]]), yo[g]).predict(np.column_stack([CXf[gf], Sf[gf]])),
             "control: along random legal moves": surrogate(np.column_stack([L.CX[g], Sr[g]]), yo[g]).predict(np.column_stack([CXf[gf], Srf[gf]])),
             "network value plus line": surrogate(np.column_stack([L.CX[g], L.q[g], S[g]]), yo[g]).predict(np.column_stack([CXf[gf], qf[gf], Sf[gf]]))}
    rngb = np.random.default_rng(0); B = [rngb.integers(0, len(y), len(y)) for _ in range(1000)]
    out["part2_fresh_games"] = {"n_fresh_positions": int(len(gf)), "share_swinging": round(float((Sf[gf, 2] != 0).mean()), 3), "gains": {}}
    for k, p in preds.items():
        e = (y - p) ** 2; bt = [(e0[i].mean() - e[i].mean()) / np.var(y[i]) for i in B]
        out["part2_fresh_games"]["gains"][k] = {"gain": round(float((e0.mean() - e.mean()) / v), 5),
                                                 "ci95": [round(float(np.quantile(bt, .025)), 5), round(float(np.quantile(bt, .975)), 5)]}
    ceil = out["part2_fresh_games"]["gains"]["network value (ceiling)"]["gain"]
    for k, d in out["part2_fresh_games"]["gains"].items():
        d["share_of_network_gain"] = round(d["gain"] / ceil, 3)
    print(json.dumps(out["part2_fresh_games"], indent=1), flush=True)
    json.dump(out, open(f"{L.RES}/line_swing_s{L.SITE}.json", "w"), indent=1)
