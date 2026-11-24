"""Classical chess knowledge as computable concepts (the 'human library' Kepler measures novelty against).

Every concept is computed from the side to move's point of view ("us" = the mover), matching Leela's input frame.
The list follows the terms of classical engine evaluation (material, mobility, king safety, pawn structure, piece
placement, threats) as used by McGrath et al. (2022) for AlphaZero, plus simple tactical motifs.
"""
import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]


def _king_zone(board, color):
    k = board.king(color)
    if k is None:
        return chess.SquareSet()
    return chess.SquareSet(chess.BB_KING_ATTACKS[k] | chess.BB_SQUARES[k])


def _pawn_files(board, color):
    return [chess.square_file(s) for s in board.pieces(chess.PAWN, color)]


def _passed(board, color):
    out = []
    opp = board.pieces(chess.PAWN, not color)
    for s in board.pieces(chess.PAWN, color):
        f, r = chess.square_file(s), chess.square_rank(s)
        blocked = False
        for o in opp:
            of, orank = chess.square_file(o), chess.square_rank(o)
            if abs(of - f) <= 1 and ((orank > r) if color == chess.WHITE else (orank < r)):
                blocked = True
                break
        if not blocked:
            out.append(r if color == chess.WHITE else 7 - r)   # relative rank 1..6
    return out


def _pawn_structure(board, color):
    files = _pawn_files(board, color)
    fs = set(files)
    doubled = sum(files.count(f) - 1 for f in fs)
    isolated = sum(1 for f in files if (f - 1) not in fs and (f + 1) not in fs)
    islands = 0
    for f in range(8):
        if f in fs and (f - 1) not in fs:
            islands += 1
    passed = _passed(board, color)
    return doubled, isolated, islands, len(passed), (max(passed) if passed else 0)


def _hanging_and_threats(board, color):
    """Pieces of `color` attacked by the other side and undefended; and attacked by a lower-valued piece."""
    hang_n = hang_v = lower_n = 0
    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for s in board.pieces(pt, color):
            attackers = board.attackers(not color, s)
            if not attackers:
                continue
            if not board.attackers(color, s):
                hang_n += 1
                hang_v += VAL[pt]
            if any(VAL[board.piece_type_at(a)] < VAL[pt] for a in attackers):
                lower_n += 1
    return hang_n, hang_v, lower_n


def _shield(board, color):
    k = board.king(color)
    if k is None:
        return 0, 0
    kf, kr = chess.square_file(k), chess.square_rank(k)
    step = 1 if color == chess.WHITE else -1
    shield = 0
    open_files = 0
    pawns = board.pieces(chess.PAWN, color)
    for f in range(max(0, kf - 1), min(7, kf + 1) + 1):
        if not any(chess.square_file(p) == f for p in pawns):
            open_files += 1
        for d in (1, 2):
            r = kr + step * d
            if 0 <= r <= 7 and chess.square(f, r) in pawns:
                shield += 1
    return shield, open_files


def _mobility(board, color):
    if board.turn == color:
        return board.legal_moves.count()
    if board.is_check():
        return 0
    b = board.copy(stack=False)
    b.push(chess.Move.null())
    return b.legal_moves.count()


def _outposts(board, color):
    n = 0
    for s in board.pieces(chess.KNIGHT, color):
        r = chess.square_rank(s) if color == chess.WHITE else 7 - chess.square_rank(s)
        if r < 4:
            continue
        defended_by_pawn = any(board.piece_type_at(a) == chess.PAWN for a in board.attackers(color, s))
        f = chess.square_file(s)
        can_be_chased = False
        for o in board.pieces(chess.PAWN, not color):
            of, orank = chess.square_file(o), chess.square_rank(o)
            ahead = (orank > chess.square_rank(s)) if color == chess.WHITE else (orank < chess.square_rank(s))
            if abs(of - f) == 1 and ahead:
                can_be_chased = True
        if defended_by_pawn and not can_be_chased:
            n += 1
    return n


def _rooks(board, color):
    open_f = semi = seventh = 0
    own = set(_pawn_files(board, color))
    opp = set(_pawn_files(board, not color))
    for s in board.pieces(chess.ROOK, color):
        f = chess.square_file(s)
        if f not in own and f not in opp:
            open_f += 1
        elif f not in own:
            semi += 1
        r = chess.square_rank(s) if color == chess.WHITE else 7 - chess.square_rank(s)
        if r == 6:
            seventh += 1
    return open_f, semi, seventh


def _king_center_dist(board, color):
    k = board.king(color)
    if k is None:
        return 0
    f, r = chess.square_file(k), chess.square_rank(k)
    return max(abs(f - 3.5), abs(r - 3.5)) - 0.5


def _space(board, color):
    """Squares in the opponent's half attacked by our pawns or pieces (attack-count proxy for space)."""
    half = chess.BB_RANK_5 | chess.BB_RANK_6 | chess.BB_RANK_7 if color == chess.WHITE else \
        chess.BB_RANK_4 | chess.BB_RANK_3 | chess.BB_RANK_2
    n = 0
    for s in chess.SquareSet(half):
        if board.is_attacked_by(color, s):
            n += 1
    return n


def _tactics(board):
    """Motifs available to the side to move: mates, checks, captures, forks, best undefended capture."""
    mate1 = checks = captures = forks = 0
    best_free = 0
    us = board.turn
    for m in board.legal_moves:
        cap = board.piece_type_at(m.to_square) if board.is_capture(m) else None
        if cap is not None or board.is_en_passant(m):
            captures += 1
            if cap is not None and not board.attackers(not us, m.to_square):
                best_free = max(best_free, VAL[cap])
        gives = board.gives_check(m)
        board.push(m)
        if gives:
            checks += 1
            if board.is_checkmate():
                mate1 += 1
        # fork: the moved piece attacks two or more enemy pieces worth >= 3 (or the king + a piece)
        attacked = [board.piece_type_at(s) for s in board.attacks(m.to_square) if board.color_at(s) == (not us)]
        big = [t for t in attacked if t is not None and (VAL[t] >= 3 or t == chess.KING)]
        if len(big) >= 2:
            forks += 1
        board.pop()
    return mate1, checks, captures, forks, best_free


def _threat_mate(board):
    if board.is_check():
        return 0
    b = board.copy(stack=False)
    b.push(chess.Move.null())
    for m in b.legal_moves:
        if b.gives_check(m):
            b.push(m)
            mate = b.is_checkmate()
            b.pop()
            if mate:
                return 1
    return 0


NAMES = None


def concepts(board):
    us, them = board.turn, not board.turn
    c = {}
    for pt, nm in [(chess.PAWN, "P"), (chess.KNIGHT, "N"), (chess.BISHOP, "B"), (chess.ROOK, "R"), (chess.QUEEN, "Q")]:
        c[f"material_{nm}"] = len(board.pieces(pt, us)) - len(board.pieces(pt, them))
    c["material"] = sum(VAL[pt] * (len(board.pieces(pt, us)) - len(board.pieces(pt, them))) for pt in VAL)
    c["phase_nonpawn"] = sum(VAL[pt] * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
                             for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    c["n_pieces"] = len(board.piece_map())
    c["bishop_pair_us"] = int(len(board.pieces(chess.BISHOP, us)) >= 2)
    c["bishop_pair_them"] = int(len(board.pieces(chess.BISHOP, them)) >= 2)
    c["mobility_us"] = _mobility(board, us)
    c["mobility_them"] = _mobility(board, them)
    c["in_check"] = int(board.is_check())
    zu, zt = _king_zone(board, us), _king_zone(board, them)
    c["king_zone_attacks_on_us"] = sum(len(board.attackers(them, s)) for s in zu)
    c["king_zone_attacks_on_them"] = sum(len(board.attackers(us, s)) for s in zt)
    c["pawn_shield_us"], c["open_files_near_king_us"] = _shield(board, us)
    c["pawn_shield_them"], c["open_files_near_king_them"] = _shield(board, them)
    c["king_center_dist_us"] = _king_center_dist(board, us)
    c["king_center_dist_them"] = _king_center_dist(board, them)
    c["castling_rights_us"] = int(board.has_castling_rights(us))
    c["castling_rights_them"] = int(board.has_castling_rights(them))
    for side, col in [("us", us), ("them", them)]:
        d, i, isl, npass, maxp = _pawn_structure(board, col)
        c[f"doubled_{side}"], c[f"isolated_{side}"], c[f"pawn_islands_{side}"] = d, i, isl
        c[f"passed_{side}"], c[f"passed_max_rank_{side}"] = npass, maxp
        o, s, sev = _rooks(board, col)
        c[f"rook_open_{side}"], c[f"rook_semiopen_{side}"], c[f"rook_7th_{side}"] = o, s, sev
        c[f"knight_outposts_{side}"] = _outposts(board, col)
        c[f"center_attacks_{side}"] = sum(len(board.attackers(col, s)) for s in CENTER)
        c[f"space_{side}"] = _space(board, col)
        hn, hv, ln = _hanging_and_threats(board, col)
        c[f"hanging_{side}"], c[f"hanging_value_{side}"], c[f"attacked_by_lower_{side}"] = hn, hv, ln
        c[f"pinned_{side}"] = sum(1 for s in chess.SquareSet(board.occupied_co[col]) if board.is_pinned(col, s)
                                  and board.piece_type_at(s) != chess.KING)
    m1, ch, cap, fk, free = _tactics(board)
    c["mate_in_1_available"], c["checks_available"], c["captures_available"] = int(m1 > 0), ch, cap
    c["forks_available"], c["best_free_capture"] = fk, free
    c["threat_mate_against_us"] = _threat_mate(board)
    c["opposite_bishops"] = int(len(board.pieces(chess.BISHOP, us)) == 1 and len(board.pieces(chess.BISHOP, them)) == 1
                                and (bool(board.pieces(chess.BISHOP, us) & chess.BB_LIGHT_SQUARES)
                                     != bool(board.pieces(chess.BISHOP, them) & chess.BB_LIGHT_SQUARES)))
    c["halfmove_clock"] = board.halfmove_clock
    return c


def concept_names():
    global NAMES
    if NAMES is None:
        NAMES = list(concepts(chess.Board()).keys())
    return NAMES
