"""Is the network's unexplained knowledge calculation? Tactical-resolution features from a material quiescence search.

    python3 scripts/kepler/chess_calc.py

Features (side to move's view):
  qs_gain     material won by resolving all captures (alpha-beta quiescence search, depth <= 6, checks evaded) minus
              the static material balance
  threat      material the opponent would win by captures if it were their move (null move), as a loss for us
  qs_mated    the capture/evasion search finds the side to move checkmated
Tested with the same frozen K3 criteria (gain on both networks, decodability vs untrained copy, non-restatement).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import chess  # noqa: E402

from chess_dark import Evaluator  # noqa: E402
from chess_known_test import load_chess, replication_q  # noqa: E402
from kepler.core.residual import gain, r2, surrogate  # noqa: E402

V = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
MATE = 100


def material(b):
    s = 0
    for pt, v in V.items():
        s += v * (len(b.pieces(pt, b.turn)) - len(b.pieces(pt, not b.turn)))
    return s


def mvv_lva(b, m):
    victim = b.piece_type_at(m.to_square) or chess.PAWN
    attacker = b.piece_type_at(m.from_square)
    return V[victim] * 10 - V[attacker]


def qsearch(b, alpha, beta, depth, nodes):
    nodes[0] += 1
    if b.is_check():
        moves = list(b.legal_moves)
        if not moves:
            return -MATE
        best = -MATE
        for m in moves:
            b.push(m)
            s = -qsearch(b, -beta, -alpha, depth - 1, nodes) if depth > 0 else -material(b)
            b.pop()
            best = max(best, s)
            alpha = max(alpha, s)
            if alpha >= beta or nodes[0] > 3000:
                break
        return best
    stand = material(b)
    if depth <= 0 or stand >= beta or nodes[0] > 3000:
        return stand
    alpha = max(alpha, stand)
    caps = sorted((m for m in b.legal_moves if b.is_capture(m)), key=lambda m: -mvv_lva(b, m))
    for m in caps:
        b.push(m)
        s = -qsearch(b, -beta, -alpha, depth - 1, nodes)
        b.pop()
        if s >= beta:
            return s
        alpha = max(alpha, s)
    return alpha


def features(b):
    st = material(b)
    q = qsearch(b, -MATE - 1, MATE + 1, 6, [0])
    out = {"qs_gain": float(np.clip(q - st, -20, 20)), "qs_mated": float(q <= -MATE)}
    if b.is_check():
        out["threat"] = 0.0
    else:
        nb = b.copy(stack=False)
        nb.push(chess.Move.null())
        qo = qsearch(nb, -MATE - 1, MATE + 1, 6, [0])
        out["threat"] = float(np.clip(-(qo - material(nb)), -20, 20))   # negative = opponent threatens to win material
    return out


def main():
    items, X, names, groups = load_chess()
    q = np.array([it["q"] for it in items])
    q2 = replication_q(items)
    ev = Evaluator(items, X, names, q, q2, groups)
    t0 = time.time()
    F = [features(b) for b in ev.boards]
    print(f"features for {len(F)} positions in {time.time() - t0:.0f}s")
    res = {}
    cols = {}
    for k in ("qs_gain", "threat", "qs_mated"):
        vals = np.array([f[k] for f in F])
        cols[k] = vals
        t = ev.test(vals)
        res[k] = t
        print(f"{k}: gain {t['gain_LD2']['gain']:+.4f} {np.round(t['gain_LD2']['ci95'], 4)} rep {t['gain_744706']['gain']:+.4f} "
              f"dec {t['decodability']['trained']:.2f}/{t['decodability']['control']:.2f} maxcorr {abs(t['max_corr_library'][1]):.2f} "
              f"({t['max_corr_library'][0]}) -> {'ACCEPTED' if t['accepted'] else 'no'}")
    # all three together
    ntr = ev.ntr
    Xs = X[ev.idx]
    A = np.column_stack([cols[k] for k in cols])
    for lab, y, base in (("LD2", q[ev.idx], ev.base1), ("744706", q2[ev.idx], ev.base2)):
        g, _ = gain(Xs[:ntr], y[:ntr], Xs[ntr:], y[ntr:], A[:ntr], A[ntr:], base_pred=base)
        res[f"all_three_{lab}"] = g
        print(f"all three calculation features, {lab}: R2 {g['r2_base']:.3f} -> {g['r2_aug']:.3f} (gain {g['gain']:+.4f}, CI {np.round(g['ci95'], 4)})")
    os.makedirs("results/kepler/chess", exist_ok=True)
    json.dump(res, open("results/kepler/chess/calculation_features.json", "w"), indent=1, default=float)
    np.savez(os.path.join(os.environ.get("KDATA", os.path.expanduser("~/kdata")), "chess", "LD2", "calc_eval.npz"),
             **cols, idx=ev.idx)


if __name__ == "__main__":
    main()
