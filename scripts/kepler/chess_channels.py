"""Causal and counterfactual evidence for the residual-predicting channels (K4 follow-up, evidence for the scientist).

    python3 scripts/kepler/chess_channels.py

1. Cause: set a channel to its mean at the last site and measure the change of LD2's value. If the channel carries the
   knowledge classical concepts miss, the change should track the residual (it should undo the disagreement).
2. Counterfactual drivers: in the positions where a channel is most active, remove each non-king piece in turn and
   measure the drop of the channel's total activity. Aggregated by piece, side and region, this says what turns the
   channel on, in board terms.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import chess  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from chess_known_test import load_chess  # noqa: E402
from kepler.core.residual import surrogate  # noqa: E402
from kepler.targets.chess.leela import LeelaNet, encode  # noqa: E402

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))
CHANNELS = [22, 15, 80, 122]


def main():
    torch.set_num_threads(2)
    items, X, names, groups = load_chess()
    q = np.array([it["q"] for it in items])
    C = np.load(os.path.join(KDATA, "chess", "LD2", "calc_all.npz"))["C"]
    src = np.array([it["source"] for it in items])
    quiet = np.where((src == "game") & (X[:, names.index("in_check")] == 0) & (C[:, 0] == 0) & (C[:, 1] == 0))[0]
    L = np.column_stack([X, C])
    pred = np.zeros(len(quiet))
    for tr, te in GroupKFold(5).split(quiet, groups=groups[quiet]):
        pred[te] = surrogate(L[quiet[tr]], q[quiet[tr]]).predict(L[quiet[te]])
    r = q[quiet] - pred
    net = LeelaNet(os.path.join(KDATA, "chess", "LD2.pb.gz"))
    A = np.load(os.path.join(KDATA, "chess", "LD2", "acts_site10.npy"), mmap_mode="r")
    rng = np.random.default_rng(0)
    sub = rng.choice(len(quiet), 1500, replace=False)
    boards = [chess.Board(items[quiet[i]]["fen"]) for i in sub]
    x = torch.tensor(np.stack([encode(b) for b in boards]))
    with torch.no_grad():
        h10 = net(x, upto=10)
        _, wdl0 = net.heads(h10)
        q0 = (wdl0[:, 0] - wdl0[:, 2]).numpy()
    means = np.asarray(A[quiet[:3000]]).astype(np.float32).reshape(-1, 128, 64).mean(0)   # per channel, per square
    out = {}
    for ch in CHANNELS:
        with torch.no_grad():
            h = h10.clone()
            h[:, ch] = torch.tensor(means[ch].reshape(8, 8))
            _, wdl = net.heads(h)
            dq = (wdl[:, 0] - wdl[:, 2]).numpy() - q0
        rs = r[sub]
        c = float(np.corrcoef(dq, -rs)[0, 1])     # undoing the disagreement means dq has the sign of -residual
        # share of the residual removed when the channel is reset: regress residual on dq
        frac = float(np.sum(-dq * rs) / np.sum(rs * rs))
        tot = np.asarray(A[quiet[sub]]).astype(np.float32).reshape(-1, 128, 64)[:, ch].sum(1)
        # counterfactual drivers in the 120 most active positions
        top = np.argsort(-tot)[:120]
        agg = {}
        for k in top:
            b = boards[k]
            base_t = float(h10[k, ch].sum())
            pieces = [(sq, p) for sq, p in b.piece_map().items() if p.piece_type != chess.KING]
            bb = []
            for sq, p in pieces:
                b2 = b.copy(stack=False)
                b2.remove_piece_at(sq)
                bb.append(encode(b2))
            with torch.no_grad():
                hh = net(torch.tensor(np.stack(bb)), upto=10)[:, ch].sum((1, 2)).numpy()
            for (sq, p), v in zip(pieces, hh):
                side = "own" if p.color == b.turn else "opp"
                rel_rank = chess.square_rank(sq) if b.turn == chess.WHITE else 7 - chess.square_rank(sq)
                half = "our half" if rel_rank <= 3 else "their half"
                key = f"{side} {chess.piece_name(p.piece_type)} in {half}"
                agg.setdefault(key, []).append(base_t - float(v))
        drivers = sorted(((k, float(np.mean(v)), len(v)) for k, v in agg.items() if len(v) >= 20), key=lambda t: -abs(t[1]))
        out[ch] = {"ablation_corr_with_undoing_residual": c, "ablation_share_of_residual_removed": frac,
                   "mean_abs_dq": float(np.mean(np.abs(dq))), "drivers": drivers[:10]}
        print(f"channel {ch}: reset -> value change tracks the residual (corr {c:+.3f}, share removed {frac:+.3f}, mean |dq| {np.mean(np.abs(dq)):.3f})")
        for k, v, n in drivers[:6]:
            print(f"    removing {k:28s} changes the channel by {-v:+.2f} (n={n})")
    json.dump(out, open("results/kepler/chess/channel_evidence.json", "w"), indent=1)


if __name__ == "__main__":
    main()
