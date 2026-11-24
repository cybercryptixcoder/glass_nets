"""K10 known-answer checks for the Leela transformer port (kepler/targets/chess/leela_t.py).

Registered in docs/glass/CHANGES.md (K10) before this script was written. All four are required before the network is
used for anything:
  (1) start position: at least 2 of e2e4, d2d4, g1f3, c2c4 among the top 4 moves, and |q| < 0.3
  (2) mate in one is the top move: back-rank mate for White, and the colour-mirrored position for Black
  (3) White a queen up (White to move): q > 0.5; Black a queen up (White to move): q < -0.5
  (4) on the first 1,000 game positions of the K3 sample: value correlation with LD2 > 0.8, top-move agreement > 50%
Must fail: the same checks on a copy whose encoder weights are shuffled.
Also reported, not judged: the mate-in-one rate on 50 mateIn1 puzzles, and the speed.

Run:  python3 scripts/kepler/chess_t1_check.py [path-to-net]   (default $KDATA/chess/t1-256x10.pb.gz)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import chess  # noqa: E402
import torch  # noqa: E402

from kepler.targets.chess.leela import LeelaNet, evaluate  # noqa: E402
from kepler.targets.chess.leela_t import LeelaTransformer  # noqa: E402

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))
OUT = "results/kepler/chess/t1_check.json"

START = chess.STARTING_FEN
BACK_RANK = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
WHITE_QUEEN_UP = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_QUEEN_UP = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQ - 0 1"


def top(ev, k=1):
    order = np.argsort(-ev["p"])[:k]
    return [ev["moves"][i].uci() for i in order]


def mates(board):
    out = set()
    for m in board.legal_moves:
        board.push(m)
        if board.is_checkmate():
            out.add(m.uci())
        board.pop()
    return out


def mate_in_one_puzzles(n=50):
    """The first n mateIn1 puzzles of the local puzzle file, at the position the solver faces."""
    import io
    import zstandard
    path = os.path.join(KDATA, "chess", "puzzles_part.csv.zst")
    out = []
    with open(path, "rb") as f:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8", errors="ignore")
        for line in text:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 8 or "mateIn1" not in parts[7].split():
                continue
            try:
                b = chess.Board(parts[1])
                moves = parts[2].split()
                b.push_uci(moves[0])        # the opponent's move; the solver is now to move
            except Exception:
                continue
            if mates(b):
                out.append(b)
            if len(out) >= n:
                break
    return out


def checks(net, ld2_items, ld2_ev):
    res = {}
    ev = evaluate(net, [chess.Board(START)])[0]
    t4 = top(ev, 4)
    hits = len(set(t4) & {"e2e4", "d2d4", "g1f3", "c2c4"})
    res["start"] = {"top4": t4, "q": ev["q"], "pass": hits >= 2 and abs(ev["q"]) < 0.3}
    bw = chess.Board(BACK_RANK)
    bb = bw.mirror()
    ew, eb = evaluate(net, [bw, bb])
    res["mate_white"] = {"fen": bw.fen(), "top": top(ew)[0], "mates": sorted(mates(bw)), "pass": top(ew)[0] in mates(bw)}
    res["mate_black"] = {"fen": bb.fen(), "top": top(eb)[0], "mates": sorted(mates(bb)), "pass": top(eb)[0] in mates(bb)}
    ewq, ebq = evaluate(net, [chess.Board(WHITE_QUEEN_UP), chess.Board(BLACK_QUEEN_UP)])
    res["queen_up"] = {"white_up_q": ewq["q"], "black_up_q": ebq["q"], "pass": ewq["q"] > 0.5 and ebq["q"] < -0.5}
    boards = [chess.Board(it["fen"]) for it in ld2_items]
    t0 = time.time()
    evs = evaluate(net, boards)
    secs = time.time() - t0
    q = np.array([e["q"] for e in evs])
    q2 = np.array([e["q"] for e in ld2_ev])
    corr = float(np.corrcoef(q, q2)[0, 1])
    agree = float(np.mean([top(a)[0] == top(b)[0] for a, b in zip(evs, ld2_ev)]))
    res["agreement_LD2"] = {"n": len(boards), "value_corr": corr, "top_move_agreement": agree,
                            "pass": corr > 0.8 and agree > 0.5, "seconds": round(secs, 1)}
    res["pass"] = all(res[k]["pass"] for k in ("start", "mate_white", "mate_black", "queen_up", "agreement_LD2"))
    return res, evs


def shuffled_copy(path):
    net = LeelaTransformer(path)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for layer in net.enc:
            for p in layer.parameters():
                flat = p.data.reshape(-1)
                p.data = flat[torch.randperm(flat.numel(), generator=g)].reshape(p.shape)
    return net


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(KDATA, "chess", "t1-256x10.pb.gz")
    meta = json.load(open(os.path.join(KDATA, "chess", "LD2", "meta.json")))
    items = [it for it in meta["items"] if "ply" in it][:1000]
    ld2 = LeelaNet(os.path.join(KDATA, "chess", "LD2.pb.gz"))
    with torch.no_grad():
        ld2_ev = evaluate(ld2, [chess.Board(it["fen"]) for it in items])
        net = LeelaTransformer(path)
        out = {"net": os.path.basename(path), "activations": net.activations}
        out["checks"], evs = checks(net, items, ld2_ev)
        print("trained:", json.dumps(out["checks"], indent=1))
        out["must_fail_shuffled"], _ = checks(shuffled_copy(path), items, ld2_ev)
        print("shuffled pass:", out["must_fail_shuffled"]["pass"])
        puz = mate_in_one_puzzles(50)
        pe = evaluate(net, puz)
        out["info_mate_in_one_puzzles"] = {"n": len(puz), "solved": int(sum(top(e)[0] in mates(b) for e, b in zip(pe, puz)))}
        pl = evaluate(ld2, puz)
        out["info_mate_in_one_puzzles"]["LD2_solved"] = int(sum(top(e)[0] in mates(b) for e, b in zip(pl, puz)))
        played = [(e, l, it["played"]) for e, l, it in zip(evs, ld2_ev, items) if "played" in it]
        if played:   # the K3 sample predates the `played` field; the behaviour study (K8/K11) measures this instead
            out["info_human_move_match"] = {"n": len(played),
                                            "t1": float(np.mean([top(e)[0] == m for e, _, m in played])),
                                            "LD2": float(np.mean([top(l)[0] == m for _, l, m in played]))}
    out["verdict"] = "PASS" if out["checks"]["pass"] and not out["must_fail_shuffled"]["pass"] else "FAIL"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: out.get(k) for k in ("verdict", "info_mate_in_one_puzzles", "info_human_move_match")}, indent=1))


if __name__ == "__main__":
    main()
