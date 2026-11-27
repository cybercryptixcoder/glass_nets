"""Chess boards: the starting vocabulary (bench/accept/VOCABULARY.md, domain `chess`).

Frozen, self-contained copy of the board features the earlier sprints used (Kepler descriptors, kepler/targets/chess/
squares.py, and Rosetta static extras, rosetta/chess_lib.py), so later edits there cannot move the vocabulary.
Squares are in the side to move's frame: index = rank * 8 + file, rank 0 = the mover's first rank, as in the Leela
input planes. The look-ahead primitives of Rosetta (the network's own line) are not base features: a definition
reaches the network's own choice only through a part reference (@id) to a part of the blueprint that computes it.
"""
import chess
import numpy as np

V = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
PT = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
DESC_NAMES = (["empty"] + [f"own_{chess.piece_symbol(p).upper()}" for p in PT] + [f"opp_{chess.piece_symbol(p).upper()}" for p in PT] +
         ["rel_rank", "file", "edge_dist", "center_dist", "att_us", "att_them", "min_att_us", "min_att_them",
          "pawn_att_us", "pawn_att_them", "dist_own_king", "dist_opp_king", "own_king_zone", "opp_king_zone",
          "piece_reach", "pinned", "defended", "attacked", "passed_pawn", "isolated_pawn", "doubled_pawn",
          "safe_from_opp_pawns", "safe_from_own_pawns", "open_file", "semiopen_us", "semiopen_them", "light_square",
          "behind_own_pawn", "in_front_of_opp_pawn"])


def _kdist(a, b):
    if a is None or b is None:
        return 8
    return max(abs(chess.square_file(a) - chess.square_file(b)), abs(chess.square_rank(a) - chess.square_rank(b)))


def descriptors(board):
    """[64, len(DESC_NAMES)] float32; row k = network square k (mover's frame)."""
    us, them = board.turn, not board.turn
    flip = us == chess.BLACK
    out = np.zeros((64, len(DESC_NAMES)), dtype=np.float32)
    ku, kt = board.king(us), board.king(them)
    pawns_us, pawns_them = board.pieces(chess.PAWN, us), board.pieces(chess.PAWN, them)
    files_us = [chess.square_file(p) for p in pawns_us]
    files_them = [chess.square_file(p) for p in pawns_them]
    zone_u = chess.SquareSet(chess.BB_KING_ATTACKS[ku]) if ku is not None else chess.SquareSet()
    zone_t = chess.SquareSet(chess.BB_KING_ATTACKS[kt]) if kt is not None else chess.SquareSet()
    fwd = 1 if us == chess.WHITE else -1
    for k in range(64):
        sq = chess.square_mirror(k) if flip else k
        f, r = chess.square_file(sq), chess.square_rank(sq)
        rel = r if us == chess.WHITE else 7 - r
        row = out[k]
        p = board.piece_at(sq)
        if p is None:
            row[0] = 1
        else:
            row[1 + PT.index(p.piece_type) + (0 if p.color == us else 6)] = 1
        j = 13
        au, at = board.attackers(us, sq), board.attackers(them, sq)
        row[j:j + 4] = [rel, f, min(f, 7 - f), max(abs(f - 3.5), abs(rel - 3.5))]
        row[j + 4] = len(au)
        row[j + 5] = len(at)
        row[j + 6] = min((V[board.piece_type_at(a)] for a in au), default=0)
        row[j + 7] = min((V[board.piece_type_at(a)] for a in at), default=0)
        row[j + 8] = float(any(board.piece_type_at(a) == chess.PAWN for a in au))
        row[j + 9] = float(any(board.piece_type_at(a) == chess.PAWN for a in at))
        row[j + 10] = _kdist(sq, ku)
        row[j + 11] = _kdist(sq, kt)
        row[j + 12] = float(sq in zone_u)
        row[j + 13] = float(sq in zone_t)
        if p is not None:
            row[j + 14] = len(board.attacks(sq))
            row[j + 15] = float(board.is_pinned(p.color, sq))
            row[j + 16] = float(len(board.attackers(p.color, sq)) > 0)
            row[j + 17] = float(len(board.attackers(not p.color, sq)) > 0)
            if p.piece_type == chess.PAWN:
                own_files = files_us if p.color == us else files_them
                opp = pawns_them if p.color == us else pawns_us
                d = 1 if p.color == chess.WHITE else -1
                row[j + 18] = float(not any(abs(chess.square_file(o) - f) <= 1 and (chess.square_rank(o) - r) * d > 0 for o in opp))
                row[j + 19] = float((f - 1) not in own_files and (f + 1) not in own_files)
                row[j + 20] = float(own_files.count(f) > 1)
        # can an opponent pawn ever attack this square? (no opponent pawn on adjacent files that can still advance to it)
        row[j + 21] = float(not any(abs(chess.square_file(o) - f) == 1 and (chess.square_rank(o) - r) * fwd > 0 for o in pawns_them))
        row[j + 22] = float(not any(abs(chess.square_file(o) - f) == 1 and (r - chess.square_rank(o)) * fwd > 0 for o in pawns_us))
        row[j + 23] = float(f not in files_us and f not in files_them)
        row[j + 24] = float(f not in files_us)
        row[j + 25] = float(f not in files_them)
        row[j + 26] = float((f + r) % 2 == 1)
        row[j + 27] = float(any(chess.square_file(o) == f and (chess.square_rank(o) - r) * fwd > 0 for o in pawns_us))
        row[j + 28] = float(any(chess.square_file(o) == f and (r - chess.square_rank(o)) * fwd > 0 for o in pawns_them))
    return out


EXTRA_NAMES = ["hanging_opp", "hanging_own", "opp_att_by_lower", "own_att_by_lower", "mob_us", "mob_them",
         "check_from_us", "check_from_them", "control", "outnumbered_own", "outnumbered_opp"]


def _net_sq(sq, flip):
    return chess.square_mirror(sq) if flip else sq


def static_extra(board):
    us, them = board.turn, not board.turn
    flip = us == chess.BLACK
    out = np.zeros((64, len(EXTRA_NAMES)), dtype=np.float32)
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




SEE_NAMES = ["see_us", "see_them"]


def _least_attacker(board, color, sq):
    best = None
    for a in board.attackers(color, sq):
        v = V[board.piece_type_at(a)]
        if best is None or v < best[0]:
            best = (v, a)
    return best


def see(board, sq, side):
    """Static exchange evaluation ("play out the exchange"): the material `side` wins by capturing on `sq` with its
    least valuable attacker, both sides then recapturing with their least valuable attackers and each free to stop
    (standard swap algorithm; x-rays are seen because attackers are recomputed after each capture; pins and
    promotions are ignored). 0 when there is nothing of the other side's to capture there, or no attacker."""
    target = board.piece_at(sq)
    if target is None or target.color == side:
        return 0.0
    b = board.copy(stack=False)
    first = _least_attacker(b, side, sq)
    if first is None:
        return 0.0
    gain = [float(V[target.piece_type])]
    on_sq_value, color = first[0], not side
    ptype = b.piece_type_at(first[1])
    b.remove_piece_at(first[1]); b.set_piece_at(sq, chess.Piece(ptype, side))
    while True:
        nxt = _least_attacker(b, color, sq)
        if nxt is None:
            break
        gain.append(on_sq_value - gain[-1])
        ptype = b.piece_type_at(nxt[1])
        b.remove_piece_at(nxt[1]); b.set_piece_at(sq, chess.Piece(ptype, color))
        on_sq_value, color = nxt[0], not color
    for d in range(len(gain) - 1, 0, -1):
        gain[d - 1] = -max(-gain[d - 1], gain[d])
    return gain[0]


def see_features(board):
    """[64, 2]: see_us (the side to move capturing on the square) and see_them (the opponent capturing there, as if
    it were its move), in the mover's frame."""
    us = board.turn
    flip = us == chess.BLACK
    out = np.zeros((64, 2), dtype=np.float32)
    for sq in chess.SQUARES:
        k = _net_sq(sq, flip)
        out[k, 0] = see(board, sq, us)
        out[k, 1] = see(board, sq, not us)
    return out
