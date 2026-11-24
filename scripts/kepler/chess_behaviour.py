"""What does the network PLAY differently from humans, in the very same positions? (behavioural strategy mining)

    python3 scripts/kepler/chess_behaviour.py

For 16,000 real game positions: LD2's top move vs the move the human actually played (and the replication network's
top move). Move features from the mover's point of view; paired comparison per feature with a bootstrap interval;
stratified by the human's rating. A tendency counts only if both networks show it and it holds in every rating band.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import chess  # noqa: E402
import torch  # noqa: E402

from kepler.targets.chess.leela import LeelaNet, encode, move_index  # noqa: E402

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))
V = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}


def feats(b, m):
    us = b.turn
    p = b.piece_at(m.from_square)
    fr = chess.square_rank(m.from_square) if us else 7 - chess.square_rank(m.from_square)
    tr = chess.square_rank(m.to_square) if us else 7 - chess.square_rank(m.to_square)
    ff = chess.square_file(m.from_square)
    cap = b.is_capture(m)
    f = {"capture": cap, "check": b.gives_check(m), "castle": b.is_castling(m), "promotion": m.promotion is not None}
    pt = p.piece_type
    f["pawn move"] = pt == chess.PAWN
    f["flank pawn push (a,b,g,h)"] = pt == chess.PAWN and not cap and ff in (0, 1, 6, 7)
    f["rook-pawn push (a,h)"] = pt == chess.PAWN and not cap and ff in (0, 7)
    f["centre pawn push (c-f)"] = pt == chess.PAWN and not cap and ff in (2, 3, 4, 5)
    f["knight move"] = pt == chess.KNIGHT
    f["bishop move"] = pt == chess.BISHOP
    f["rook move"] = pt == chess.ROOK
    f["queen move"] = pt == chess.QUEEN
    f["king move (not castling)"] = pt == chess.KING and not b.is_castling(m)
    f["retreat (toward own side)"] = pt != chess.PAWN and tr < fr and not cap
    f["rook lift (leaves back rank)"] = pt == chess.ROOK and fr == 0 and tr > 0
    quiet = not cap and not f["check"] and not f["promotion"] and not f["castle"]
    f["quiet move"] = quiet
    # moved piece lands where a cheaper enemy piece attacks it, or attacked and undefended (material offered)
    b2 = b.copy(stack=False)
    b2.push(m)
    att = b2.attackers(not us, m.to_square)
    dfd = b2.attackers(us, m.to_square)
    offered = bool(att) and (any(V[b2.piece_type_at(a)] < V[pt] for a in att) or not dfd) and pt != chess.KING
    gained = V[b.piece_type_at(m.to_square)] if cap and b.piece_type_at(m.to_square) else 0
    f["offers material (sacrifice-like)"] = offered and V[pt] > gained
    return f


def main():
    torch.set_num_threads(2)
    meta = json.load(open(os.path.join(KDATA, "chess", "LD2", "meta.json")))
    items = [it for it in meta["items"] if it["source"] == "game"]
    played = json.load(open(os.path.join(KDATA, "chess", "LD2", "played_moves.json")))
    boards = [chess.Board(it["fen"]) for it in items]
    rep = LeelaNet(os.path.join(KDATA, "chess", "744706.pb.gz"))
    rep_top = []
    with torch.no_grad():
        for i in range(0, len(boards), 256):
            bs = boards[i:i + 256]
            lg, _ = rep(torch.tensor(np.stack([encode(b) for b in bs])))
            for b, l in zip(bs, lg):
                mv = list(b.legal_moves)
                rep_top.append(mv[int(np.argmax([float(l[move_index(rep, b, m)]) for m in mv]))])
    elo = []
    for it, b in zip(items, boards):
        e = it["white_elo"] if b.turn else it["black_elo"]
        elo.append(int(e) if str(e).isdigit() else 0)
    elo = np.array(elo)
    F_l, F_h, F_r = [], [], []
    for it, b, pm, rm in zip(items, boards, played, rep_top):
        F_l.append(feats(b, chess.Move.from_uci(it["top_move"])))
        F_h.append(feats(b, chess.Move.from_uci(pm)))
        F_r.append(feats(b, rm))
    keys = list(F_l[0])
    L = np.array([[f[k] for k in keys] for f in F_l], dtype=float)
    H = np.array([[f[k] for k in keys] for f in F_h], dtype=float)
    Rr = np.array([[f[k] for k in keys] for f in F_r], dtype=float)
    agree = float(np.mean([it["top_move"] == pm for it, pm in zip(items, played)]))
    rng = np.random.default_rng(0)
    bands = [("< 1500", elo < 1500), ("1500-1799", (elo >= 1500) & (elo < 1800)), (">= 1800", elo >= 1800)]
    out = {"n": len(items), "leela_agrees_with_human": agree, "features": {}}
    print(f"{len(items)} positions; LD2 plays the human's move in {agree:.1%}")
    for j, k in enumerate(keys):
        d = L[:, j] - H[:, j]
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(400)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        band = {nm: (float(L[m, j].mean()), float(H[m, j].mean())) for nm, m in bands if m.sum() > 200}
        rep_d = float((Rr[:, j] - H[:, j]).mean())
        consistent = all((lv - hv) * np.sign(d.mean()) > 0 for lv, hv in band.values()) and np.sign(rep_d) == np.sign(d.mean())
        out["features"][k] = {"leela": float(L[:, j].mean()), "human": float(H[:, j].mean()), "replication_net": float(Rr[:, j].mean()),
                              "diff": float(d.mean()), "ci95": [float(lo), float(hi)], "by_rating": band,
                              "consistent_all_bands_and_both_nets": bool(consistent and (lo > 0 or hi < 0))}
    for k, v in sorted(out["features"].items(), key=lambda t: -abs(t[1]["diff"] / max(t[1]["human"], 1e-3))):
        ratio = v["leela"] / max(v["human"], 1e-6)
        print(f"{k:34s} LD2 {v['leela']:.3f} | 744706 {v['replication_net']:.3f} | human {v['human']:.3f}  x{ratio:4.2f} "
              f"[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] {'CONSISTENT' if v['consistent_all_bands_and_both_nets'] else ''}  "
              + " ".join(f"{nm}: {a:.3f}/{h:.3f}" for nm, (a, h) in v['by_rating'].items()))
    # examples for the strongest consistent tendency
    json.dump(out, open("results/kepler/chess/behaviour.json", "w"), indent=1)


if __name__ == "__main__":
    main()
