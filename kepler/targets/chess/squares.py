"""Per-square board descriptors in the network's frame (row 0 = the side to move's first rank).

Used to turn a network unit's per-square activity into a readable rule: a shallow tree over these descriptors.
"""
import chess
import numpy as np

V = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
PT = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
NAMES = (["empty"] + [f"own_{chess.piece_symbol(p).upper()}" for p in PT] + [f"opp_{chess.piece_symbol(p).upper()}" for p in PT] +
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
    """[64, len(NAMES)] float32; row k = network square k (mover's frame)."""
    us, them = board.turn, not board.turn
    flip = us == chess.BLACK
    out = np.zeros((64, len(NAMES)), dtype=np.float32)
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
