"""Rosetta's chess library: per-square primitives in the network's frame (row 0 = the side to move's first rank,
square index = rank * 8 + file, as in the Leela input planes and kepler/targets/chess/squares.py).

Three kinds of primitives:
- static board facts (Kepler's 42 descriptors plus threats, hanging pieces, mobility and check squares);
- look-ahead: the squares of the network's OWN best line (its top move, the opponent's top reply, its follow-up),
  computed by running the network itself, so Rosetta can recognise search inside the network;
- learned: definitions accepted in earlier rounds (vocabulary growth), added by the engine at run time.
"""
import os
import sys

import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kepler.targets.chess.squares import NAMES as KNAMES, descriptors, V  # noqa: E402

EXTRA = ["hanging_opp", "hanging_own", "opp_att_by_lower", "own_att_by_lower", "mob_us", "mob_them",
         "check_from_us", "check_from_them", "control", "outnumbered_own", "outnumbered_opp"]
LOOK = ["la_m1_from", "la_m1_to", "la_m2_from", "la_m2_to", "la_m3_from", "la_m3_to", "la_m1_attacks", "la_m3_attacks"]
NAMES = list(KNAMES) + EXTRA + LOOK


def _net_sq(sq, flip):
    return chess.square_mirror(sq) if flip else sq


def static_extra(board):
    us, them = board.turn, not board.turn
    flip = us == chess.BLACK
    out = np.zeros((64, len(EXTRA)), dtype=np.float32)
    mob_us = np.zeros(64)
    chk_us = np.zeros(64)
    for m in board.legal_moves:
        t = _net_sq(m.to_square, flip)
        mob_us[t] += 1
        if board.gives_check(m):
            chk_us[t] = 1
    b2 = board.copy(stack=False)
    b2.turn = them
    b2.ep_square = None
    mob_th = np.zeros(64)
    chk_th = np.zeros(64)
    for m in b2.pseudo_legal_moves:
        t = _net_sq(m.to_square, flip)
        mob_th[t] += 1
        try:
            if b2.gives_check(m):
                chk_th[t] = 1
        except Exception:
            pass
    for sq in chess.SQUARES:
        k = _net_sq(sq, flip)
        p = board.piece_at(sq)
        au, at = board.attackers(us, sq), board.attackers(them, sq)
        row = out[k]
        if p is not None:
            if p.color == them:
                row[0] = float(len(au) > 0 and len(at) == 0)
                row[2] = float(any(V[board.piece_type_at(a)] < V[p.piece_type] for a in au))
                row[10] = float(len(au) > len(at) and len(au) > 0)
            else:
                row[1] = float(len(at) > 0 and len(au) == 0)
                row[3] = float(any(V[board.piece_type_at(a)] < V[p.piece_type] for a in at))
                row[9] = float(len(at) > len(au) and len(at) > 0)
        row[8] = len(au) - len(at)
    out[:, 4] = mob_us
    out[:, 5] = mob_th
    out[:, 6] = chk_us
    out[:, 7] = chk_th
    return out


def look_ahead(board, line):
    """line: list of up to three python-chess moves (network's top move, opponent's top reply, network's follow-up),
    in real coordinates. Squares are returned in the ORIGINAL position's network frame."""
    flip = board.turn == chess.BLACK
    out = np.zeros((64, len(LOOK)), dtype=np.float32)
    b = board.copy(stack=False)
    for j, m in enumerate(line[:3]):
        if m is None or m not in b.legal_moves:
            break
        out[_net_sq(m.from_square, flip), 2 * j] = 1
        out[_net_sq(m.to_square, flip), 2 * j + 1] = 1
        b.push(m)
        if j in (0, 2):
            for a in b.attacks(m.to_square):
                out[_net_sq(a, flip), 6 + (j // 2)] = 1
    return out


def primitives(fen, line):
    board = chess.Board(fen)
    return np.concatenate([descriptors(board), static_extra(board), look_ahead(board, line)], axis=1)


def _work(args):
    fen, line = args
    return primitives(fen, [chess.Move.from_uci(u) if u else None for u in line])


def build(fens, lines, procs=2):
    from multiprocessing import Pool
    with Pool(procs) as pool:
        rows = pool.map(_work, list(zip(fens, lines)), chunksize=64)
    return np.stack(rows).astype(np.float32)
