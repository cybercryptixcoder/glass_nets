"""K11: strategy rules from the strong network (t1-256x10) by contrast with human play.

Registered in docs/glass/CHANGES.md (K11) before this script was written.

    python3 scripts/kepler/chess_rules.py eval        # networks' moves and move-type policy mass on the K8 sample
    python3 scripts/kepler/chess_rules.py mine        # candidate rules + replication + rating bands + must-fail
    python3 scripts/kepler/chess_rules.py test K      # cause (board edits) and worth (t1 value of human moves), rule K

A rule is "in situation C, play move type M". All move and situation features are from the mover's point of view.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import chess  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from chess_behaviour import feats as k8_feats  # noqa: E402
from kepler.targets.chess.leela import LeelaNet, encode, move_index  # noqa: E402
from kepler.targets.chess.leela_t import LeelaTransformer  # noqa: E402

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))
WORK = os.path.join(KDATA, "chess", "rules")
OUT = "results/kepler/chess/rules"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rel_rank(sq, us):
    return chess.square_rank(sq) if us else 7 - chess.square_rank(sq)


def move_feats(b, m):
    """K8's move types plus pawn moves near either king, minor pieces to the rim, trade offers, king walks."""
    f = k8_feats(b, m)
    us = b.turn
    p = b.piece_at(m.from_square)
    pt = p.piece_type
    cap = b.is_capture(m)
    ff, tf = chess.square_file(m.from_square), chess.square_file(m.to_square)
    ok, ek = b.king(us), b.king(not us)
    f["pawn push near own king"] = pt == chess.PAWN and not cap and ok is not None and abs(ff - chess.square_file(ok)) <= 1
    f["pawn push near enemy king"] = pt == chess.PAWN and not cap and ek is not None and abs(ff - chess.square_file(ek)) <= 1
    f["minor piece to the rim (a/h file)"] = pt in (chess.KNIGHT, chess.BISHOP) and tf in (0, 7)
    f["piece into enemy half"] = pt not in (chess.PAWN, chess.KING) and rel_rank(m.to_square, us) >= 4 and rel_rank(m.from_square, us) < 4
    f["king steps forward"] = pt == chess.KING and not b.is_castling(m) and rel_rank(m.to_square, us) > rel_rank(m.from_square, us)
    f["pawn capture toward centre"] = pt == chess.PAWN and cap and abs(tf - 3.5) < abs(ff - 3.5)
    f["pawn capture away from centre"] = pt == chess.PAWN and cap and abs(tf - 3.5) > abs(ff - 3.5)
    trade = False
    if pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN) and not cap:
        b2 = b.copy(stack=False)
        b2.push(m)
        trade = any(b2.piece_type_at(a) == pt for a in b2.attackers(not us, m.to_square)) and bool(b2.attackers(us, m.to_square))
    f["offers a like-for-like trade"] = trade
    return {k: bool(v) for k, v in f.items()}


def _isolated_doubled(b, color):
    files = [chess.square_file(q) for q in b.pieces(chess.PAWN, color)]
    doubled = sum(files.count(f) - 1 for f in set(files))
    iso = sum(1 for f in files if (f - 1) not in files and (f + 1) not in files)
    return iso, doubled


def move_feats_c(b, m):
    """K11c: moves that classical principles advise against (mover's view)."""
    us = b.turn
    p = b.piece_at(m.from_square)
    pt = p.piece_type
    cap = b.is_capture(m)
    victim = b.piece_type_at(m.to_square) if cap else None
    fr, tr = rel_rank(m.from_square, us), rel_rank(m.to_square, us)
    ff = chess.square_file(m.from_square)
    ok = b.king(us)
    queens_on = bool(b.pieces(chess.QUEEN, us)) and bool(b.pieces(chess.QUEEN, not us))
    nonpawn = sum(len(b.pieces(t, c)) * v for t, v in ((chess.KNIGHT, 3), (chess.BISHOP, 3), (chess.ROOK, 5), (chess.QUEEN, 9))
                  for c in (chess.WHITE, chess.BLACK))
    b2 = b.copy(stack=False)
    b2.push(m)
    i0, d0 = _isolated_doubled(b, us)
    i1, d1 = _isolated_doubled(b2, us)
    att, dfd = b2.attackers(not us, m.to_square), b2.attackers(us, m.to_square)
    kf, kr = chess.square_file(ok), rel_rank(ok, us)
    castled = kr <= 1 and (kf >= 5 or kf <= 2)
    passed = False
    if pt == chess.PAWN:
        step = 1 if us == chess.WHITE else -1
        r = chess.square_rank(m.from_square)
        passed = not any(chess.square_file(q) in (ff - 1, ff, ff + 1) and (chess.square_rank(q) - r) * step > 0
                         for q in b.pieces(chess.PAWN, not us))
    lever = pt == chess.PAWN and not cap and any(b2.piece_type_at(t) == chess.PAWN and b2.color_at(t) != us
                                                 for t in b2.attacks(m.to_square))
    f = {
        "bishop takes knight": pt == chess.BISHOP and victim == chess.KNIGHT,
        "gives up the bishop pair": pt == chess.BISHOP and victim == chess.KNIGHT and len(b.pieces(chess.BISHOP, us)) >= 2,
        "creates own doubled pawns": d1 > d0,
        "creates own isolated pawn": i1 > i0,
        "pawn push in front of own castled king": pt == chess.PAWN and not cap and castled and abs(ff - kf) <= 1,
        "king walk with queens on": pt == chess.KING and not b.is_castling(m) and queens_on and nonpawn >= 40,
        "early queen sortie (into enemy half, castling rights left)": pt == chess.QUEEN and tr >= 4 and fr < 4
            and b.has_castling_rights(us),
        "retreat to the home rank": pt not in (chess.PAWN, chess.KING) and tr == 0 and fr > 0 and not cap,
        "pawn sacrifice (pushed pawn en prise)": pt == chess.PAWN and not cap and bool(att) and not dfd,
        "exchange sacrifice (rook takes a defended minor)": pt == chess.ROOK and victim in (chess.KNIGHT, chess.BISHOP) and bool(att),
        "declines a legal castle": any(b.is_castling(x) for x in b.legal_moves) and not b.is_castling(m),
        "space push (pawn to the 5th rank or beyond)": pt == chess.PAWN and not cap and tr >= 4,
        "passed-pawn push": pt == chess.PAWN and not cap and passed,
        "knight retreat to the back two ranks": pt == chess.KNIGHT and tr <= 1 and fr >= 2,
        "queen trade (captures or offers the queen)": pt == chess.QUEEN and (victim == chess.QUEEN or
            any(b2.piece_type_at(a) == chess.QUEEN for a in att) and bool(dfd)),
        "piece to the 6th/7th rank": pt not in (chess.PAWN, chess.KING) and tr >= 5 and fr < 5,
        "pawn lever (push attacks an enemy pawn)": lever,
    }
    return {k: bool(v) for k, v in f.items()}


FEATS = move_feats


def king_conditions(b):
    """Situation conditions about king placement (mover's view)."""
    us = b.turn
    out = {}
    for who, side in (("own", us), ("enemy", not us)):
        k = b.king(side)
        f, r = chess.square_file(k), rel_rank(k, side)
        out[f"{who} king castled short (f-h file, home ranks)"] = f >= 5 and r <= 1
        out[f"{who} king castled long (a-c file, home ranks)"] = f <= 2 and r <= 1
        out[f"{who} king in the centre (d/e file)"] = f in (3, 4)
        out[f"{who} king off its home ranks"] = r >= 2
    ok, ek = chess.square_file(b.king(us)), chess.square_file(b.king(not us))
    out["kings on opposite wings"] = (ok >= 5 and ek <= 2) or (ok <= 2 and ek >= 5)
    out["kings on the same wing"] = (ok >= 5 and ek >= 5) or (ok <= 2 and ek <= 2)
    out["queens off"] = not b.pieces(chess.QUEEN, us) and not b.pieces(chess.QUEEN, not us)
    return out


def load_sample():
    meta = json.load(open(os.path.join(KDATA, "chess", "LD2", "meta.json")))
    idx = [i for i, it in enumerate(meta["items"]) if it["source"] == "game"]
    items = [meta["items"][i] for i in idx]
    played = json.load(open(os.path.join(KDATA, "chess", "LD2", "played_moves.json")))
    return idx, items, played


def policy_eval(net, boards, batch=256):
    """Top move (uci), q, and the per-legal-move probabilities for each board."""
    tops, qs, probs = [], [], []
    with torch.no_grad():
        for i in range(0, len(boards), batch):
            bs = boards[i:i + batch]
            logits, wdl = net(torch.tensor(np.stack([encode(b) for b in bs])))
            for b, lg, w in zip(bs, logits, wdl):
                mv = list(b.legal_moves)
                li = torch.stack([lg[move_index(net, b, m)] for m in mv])
                p = F.softmax(li, 0).numpy()
                tops.append(mv[int(np.argmax(p))].uci())
                qs.append(float(w[0] - w[2]))
                probs.append(p)
    return tops, np.array(qs, dtype=np.float32), probs


def cmd_eval():
    os.makedirs(WORK, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    idx, items, played = load_sample()
    boards = [chess.Board(it["fen"]) for it in items]
    log(f"{len(boards)} positions")
    t1 = LeelaTransformer(os.path.join(KDATA, "chess", "t1-256x10.pb.gz"))
    t1_top, t1_q, t1_p = policy_eval(t1, boards)
    log("t1 done")
    rep = LeelaNet(os.path.join(KDATA, "chess", "744706.pb.gz"))
    rep_top, rep_q, _ = policy_eval(rep, boards)
    log("744706 done")
    keys = None
    T, H, L, R, MASS, KC = [], [], [], [], [], []
    for it, b, pm, tt, rt, p in zip(items, boards, played, t1_top, rep_top, t1_p):
        mv = list(b.legal_moves)
        mf = [move_feats(b, m) for m in mv]
        keys = keys or list(mf[0])
        A = np.array([[f[k] for k in keys] for f in mf], dtype=np.float32)
        MASS.append(p @ A)
        uci = [m.uci() for m in mv]
        T.append(A[uci.index(tt)])
        H.append(move_feats(b, chess.Move.from_uci(pm)) if pm in uci else {k: False for k in keys})
        L.append(A[uci.index(it["top_move"])] if it["top_move"] in uci else np.zeros(len(keys)))
        R.append(A[uci.index(rt)])
        KC.append(king_conditions(b))
    H = np.array([[h[k] for k in keys] for h in H], dtype=np.float32)
    kc_keys = list(KC[0])
    KCm = np.array([[c[k] for k in kc_keys] for c in KC], dtype=np.float32)
    elo = np.array([int(e) if str(e).isdigit() else 0 for e in
                    [it["white_elo"] if b.turn else it["black_elo"] for it, b in zip(items, boards)]])
    np.savez(os.path.join(WORK, "eval.npz"), T=np.array(T), H=H, L=np.array(L), R=np.array(R), MASS=np.array(MASS),
             KC=KCm, elo=elo, t1_q=t1_q, rep_q=rep_q, idx=np.array(idx))
    json.dump({"move_types": keys, "king_conditions": kc_keys, "t1_top": t1_top, "rep_top": rep_top},
              open(os.path.join(WORK, "eval_meta.json"), "w"))
    log("saved; t1 plays the human move in", float(np.mean([a == b for a, b in zip(t1_top, played)])))


def cmd_evalc():
    E = dict(np.load(os.path.join(WORK, "eval.npz")))
    meta = json.load(open(os.path.join(WORK, "eval_meta.json")))
    _, items, played = load_sample()
    keys = None
    out = {"T": [], "H": [], "L": [], "R": []}
    for i, it in enumerate(items):
        b = chess.Board(it["fen"])
        legal = {m.uci() for m in b.legal_moves}
        for nm, mv in (("T", meta["t1_top"][i]), ("H", played[i]), ("L", it["top_move"]), ("R", meta["rep_top"][i])):
            f = move_feats_c(b, chess.Move.from_uci(mv)) if mv in legal else None
            keys = keys or (list(f) if f else None)
            out[nm].append([f[k] for k in keys] if f else [False] * 17)
    np.savez(os.path.join(WORK, "eval_c.npz"), **{k: np.array(v, dtype=np.float32) for k, v in out.items()})
    json.dump({"move_types": keys}, open(os.path.join(WORK, "eval_c_meta.json"), "w"))
    log("saved", {k: np.array(v).mean(0).round(3).tolist() for k, v in out.items()})


def load_eval():
    E = dict(np.load(os.path.join(WORK, "eval.npz")))
    meta = json.load(open(os.path.join(WORK, "eval_meta.json")))
    if os.environ.get("MOVES") == "c":
        global FEATS
        FEATS = move_feats_c
        Ec = dict(np.load(os.path.join(WORK, "eval_c.npz")))
        E.update(Ec)
        E.pop("MASS", None)
        meta["move_types"] = json.load(open(os.path.join(WORK, "eval_c_meta.json")))["move_types"]
    return E, meta


def concept_conditions(idx):
    """Binary situations from the 61-concept library: binary concepts at each value, others at the 75th / 25th
    percentile; plus the king-placement conditions (and their negations)."""
    cz = np.load(os.path.join(KDATA, "chess", "LD2", "concepts.npz"))
    X = cz["X"][idx].astype(np.float64)
    names = [str(n) for n in cz["names"]]
    conds, cn, spec = [], [], []
    for j, n in enumerate(names):
        x = X[:, j]
        vals = np.unique(x)
        if len(vals) <= 1:
            continue
        if len(vals) == 2:
            for v in vals:
                conds.append(x == v); cn.append(f"{n} = {v:g}"); spec.append((n, "eq", float(v)))
            continue
        lo, hi = np.percentile(x, [25, 75])
        conds.append(x >= hi); cn.append(f"{n} high (>= {hi:g})"); spec.append((n, "ge", float(hi)))
        conds.append(x <= lo); cn.append(f"{n} low (<= {lo:g})"); spec.append((n, "le", float(lo)))
    return conds, cn, spec


def all_conditions(E, meta):
    conds, cn, spec = concept_conditions(E["idx"])
    for j, k in enumerate(meta["king_conditions"]):
        conds.append(E["KC"][:, j] > 0.5); cn.append(k); spec.append((k, "king", 1.0))
        conds.append(E["KC"][:, j] < 0.5); cn.append("not: " + k); spec.append((k, "king", 0.0))
    return np.array(conds), cn, spec


BANDS = [("< 1500", lambda e: (e > 0) & (e < 1500)), ("1500-1799", lambda e: (e >= 1500) & (e < 1800)),
         ("1800+", lambda e: e >= 1800)]


Z_LINE = None   # K11b: set from the label-swap control (MODE=b)


def mined_name():
    if os.environ.get("MOVES") == "c":
        return "mined_c.json"
    return "mined_b.json" if os.environ.get("MODE") == "b" else "mined.json"


def z_matrix(T, H, C):
    """Interaction z for every (situation, move type) at once."""
    d = (T - H).astype(np.float64)
    Cf = C.astype(np.float64)
    n = Cf.sum(1)[:, None]
    nn = len(T) - n
    s1, s2 = Cf @ d, Cf @ (d * d)
    t1, t2 = d.sum(0)[None, :], (d * d).sum(0)[None, :]
    m_in, m_out = s1 / np.maximum(n, 1), (t1 - s1) / np.maximum(nn, 1)
    v_in = s2 / np.maximum(n, 1) - m_in ** 2
    v_out = (t2 - s2) / np.maximum(nn, 1) - m_out ** 2
    z = (m_in - m_out) / np.sqrt(np.maximum(v_in / np.maximum(n, 1) + v_out / np.maximum(nn, 1), 1e-12))
    ok = (n >= 300) & (nn >= 300)
    return np.where(ok, z, -np.inf)


def null_line(T, H, C, reps=20):
    rng = np.random.default_rng(7)
    mx = []
    for _ in range(reps):
        s = (rng.random(len(T)) < 0.5)[:, None]
        mx.append(float(z_matrix(np.where(s, H, T), np.where(s, T, H), C).max()))
    return max(mx), mx


def mine(T, H, L, R, C, cn, keys, elo):
    rows = []
    for ci in range(len(C)):
        c = C[ci]
        n, nn = int(c.sum()), int((~c).sum())
        if n < 300 or nn < 300:
            continue
        for mi, mk in enumerate(keys):
            tc, hc = T[c, mi].mean(), H[c, mi].mean()
            if tc < 0.05 or tc < (1.5 * hc if Z_LINE is None else hc):
                continue
            d = T[:, mi] - H[:, mi]
            inter = d[c].mean() - d[~c].mean()
            z = inter / max(np.sqrt(d[c].var() / n + d[~c].var() / nn), 1e-9)
            if Z_LINE is not None and z <= Z_LINE:
                continue
            rep = {}
            for nm, A in (("LD2", L), ("744706", R)):
                da = A[:, mi] - H[:, mi]
                rep[nm] = {"net_minus_human_in_C": float(da[c].mean()), "interaction": float(da[c].mean() - da[~c].mean()),
                           "net_rate_in_C": float(A[c, mi].mean())}
            bands = {}
            for bn, f in BANDS:
                m = c & f(elo)
                bands[bn] = {"n": int(m.sum()), "t1": float(T[m, mi].mean()) if m.sum() else None,
                             "human": float(H[m, mi].mean()) if m.sum() else None}
            rep_ok = all(v["net_minus_human_in_C"] > 0 and v["interaction"] > 0 for v in rep.values())
            band_ok = all(b["n"] >= 30 and b["t1"] > b["human"] for b in bands.values())
            rows.append({"condition": cn[ci], "cond_index": ci, "move_type": mk, "move_index": mi, "support": n,
                         "t1_rate_in_C": float(tc), "human_rate_in_C": float(hc), "ratio": float(tc / max(hc, 1e-9)),
                         "t1_rate_outside_C": float(T[~c, mi].mean()), "human_rate_outside_C": float(H[~c, mi].mean()),
                         "interaction": float(inter), "z": float(z), "replication": rep, "bands": bands,
                         "passes_replication": rep_ok, "passes_bands": band_ok})
    rows.sort(key=lambda r: -r["z"])
    return rows


def cmd_mine():
    E, meta = load_eval()
    keys = meta["move_types"]
    C, cn, spec = all_conditions(E, meta)
    T, H, L, R, elo = E["T"], E["H"], E["L"], E["R"], E["elo"]
    log(f"{len(C)} situations x {len(keys)} move types; t1 vs human overall:")
    for mi, k in enumerate(keys):
        log(f"   {k:36s} t1 {T[:, mi].mean():.3f}  LD2 {L[:, mi].mean():.3f}  744706 {R[:, mi].mean():.3f}  human {H[:, mi].mean():.3f}")
    global Z_LINE
    null_info = None
    if os.environ.get("MODE") == "b":
        Z_LINE, mx = null_line(T, H, C)
        null_info = {"z_line": Z_LINE, "null_max_z": mx}
        log(f"K11b: line from 20 label-swap nulls: z > {Z_LINE:.2f} (null maxima {min(mx):.2f}-{max(mx):.2f})")
    rows = mine(T, H, L, R, C, cn, keys, elo)
    passing = [r for r in rows if r["passes_replication"] and r["passes_bands"]]
    log(f"{len(rows)} candidates meet the selection rule; {len(passing)} pass replication and every rating band")
    for i, r in enumerate(passing[:25]):
        log(f" #{i:2d} z {r['z']:5.1f} | {r['condition']:52s} -> {r['move_type']:34s} t1 {r['t1_rate_in_C']:.3f} "
            f"human {r['human_rate_in_C']:.3f} (x{r['ratio']:.1f}; outside C t1 {r['t1_rate_outside_C']:.3f} human "
            f"{r['human_rate_outside_C']:.3f}) n={r['support']}")
    # must fail: net and human labels swapped at random per position (jointly for the three networks)
    rng = np.random.default_rng(0)
    s = rng.random(len(T)) < 0.5
    sw = lambda A: np.where(s[:, None], H, A)
    # each network compared with the human; on swapped positions the two labels trade places
    Ts, Ls, Rs = sw(T), sw(L), sw(R)
    HsT, HsL, HsR = np.where(s[:, None], T, H), np.where(s[:, None], L, H), np.where(s[:, None], R, H)
    null = []
    for ci in range(len(C)):
        c = C[ci]
        if c.sum() < 300 or (~c).sum() < 300:
            continue
        for mi in range(len(keys)):
            tc, hc = Ts[c, mi].mean(), HsT[c, mi].mean()
            if tc < 0.05 or tc < 1.5 * hc:
                continue
            ok = True
            for A, HA in ((Ls, HsL), (Rs, HsR)):
                da = A[:, mi] - HA[:, mi]
                ok &= da[c].mean() > 0 and (da[c].mean() - da[~c].mean()) > 0
            for bn, f in BANDS:
                m = c & f(elo)
                ok &= m.sum() >= 30 and Ts[m, mi].mean() > HsT[m, mi].mean()
            null.append(bool(ok))
    log(f"must-fail (labels swapped at random): {len(null)} meet the selection rule, {sum(null)} pass replication+bands")
    os.makedirs(OUT, exist_ok=True)
    json.dump({"n_situations": int(len(C)), "move_types": keys, "n_candidates": len(rows), "n_passing": len(passing),
               "overall": {k: {"t1": float(T[:, i].mean()), "LD2": float(L[:, i].mean()), "744706": float(R[:, i].mean()),
                               "human": float(H[:, i].mean())} for i, k in enumerate(keys)},
               "must_fail_swapped": {"candidates": len(null), "passing": int(sum(null))}, "mode": os.environ.get("MODE", "K11"),
               "null_line": null_info,
               "passing": passing[:60]}, open(os.path.join(OUT, mined_name()), "w"), indent=1)


def holds(spec, board):
    from kepler.targets.chess.concepts import concept_names, concepts
    name, kind, v = spec
    if kind == "king":
        return bool(king_conditions(board)[name]) == bool(v)
    c = concepts(board)
    x = c[name] if isinstance(c, dict) else c[concept_names().index(name)]
    return {"eq": x == v, "ge": x >= v, "le": x <= v}[kind]


def edits(b, top, rng, k=14):
    """Single-piece edits that keep the network's chosen move legal and the check status unchanged:
    remove a piece, or move it to a random empty square."""
    out = []
    pieces = [sq for sq, p in b.piece_map().items() if p.piece_type != chess.KING and sq != top.from_square]
    empties = [sq for sq in chess.SQUARES if b.piece_at(sq) is None and sq != top.to_square]
    for side in (b.turn, not b.turn):          # king relocations (needed for king-placement situations)
        if top.from_square == b.king(side):
            continue
        for e in rng.permutation(empties)[:6]:
            b2 = b.copy(stack=False)
            kp = b2.remove_piece_at(b.king(side))
            b2.set_piece_at(int(e), kp)
            b2.castling_rights = b2.clean_castling_rights()
            b2.ep_square = None
            if b2.is_valid() and top in b2.legal_moves and b2.is_check() == b.is_check():
                out.append(("king", chess.square_name(b.king(side)), kp.symbol(), b2))
                break
    for sq in rng.permutation(pieces):
        for kind in ("remove", "move"):
            b2 = b.copy(stack=False)
            p = b2.remove_piece_at(int(sq))
            if kind == "move":
                ok = [e for e in empties if not (p.piece_type == chess.PAWN and chess.square_rank(e) in (0, 7))]
                b2.set_piece_at(int(rng.choice(ok)), p)
            b2.castling_rights = b2.clean_castling_rights()
            b2.ep_square = None
            if b2.is_valid() and top in b2.legal_moves and b2.is_check() == b.is_check():
                out.append((kind, chess.square_name(int(sq)), p.symbol(), b2))
        if len(out) >= k:
            break
    return out


def mass_on(net, boards, mi):
    _, qs, probs = policy_eval(net, boards)
    out = []
    for b, p in zip(boards, probs):
        A = np.array([FEATS(b, m)[KEYS[mi]] for m in b.legal_moves], dtype=np.float32)
        out.append(float(p @ A))
    return np.array(out), qs


KEYS = None


def cmd_test():
    global KEYS
    K = int(sys.argv[2])
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    E, meta = load_eval()
    KEYS = meta["move_types"]
    mined = json.load(open(os.path.join(OUT, mined_name())))
    rule = mined["passing"][K]
    mi = rule["move_index"]
    C, cn, spec = all_conditions(E, meta)
    ci = rule["cond_index"]
    assert cn[ci] == rule["condition"]
    c = C[ci]
    _, items, played = load_sample()
    t1_top = meta["t1_top"]
    net = LeelaTransformer(os.path.join(KDATA, "chess", "t1-256x10.pb.gz"))
    rng = np.random.default_rng(1)
    res = {"rule": {k: rule[k] for k in ("condition", "move_type", "support", "t1_rate_in_C", "human_rate_in_C", "ratio",
                                          "interaction", "z")}}
    log(f"rule {K}: in [{rule['condition']}] play [{rule['move_type']}]")
    # ---- cause: edits that switch C off vs edits that keep it
    sel = rng.permutation(np.where(c & (E["T"][:, mi] > 0.5))[0])[:int(os.environ.get("N_CAUSE", "100"))]
    eb, meta_e = [], []
    for i in sel:
        b = chess.Board(items[i]["fen"])
        for kind, sq, sym, b2 in edits(b, chess.Move.from_uci(t1_top[i]), rng):
            eb.append(b2)
            meta_e.append((int(i), kind, sq, sym, bool(holds(spec[ci], b2))))
    log(f"cause: {len(sel)} positions, {len(eb)} edits ({sum(1 for m in meta_e if not m[4])} switch the situation off)")
    after, _ = mass_on(net, eb, mi)
    base, _ = mass_on(net, [chess.Board(items[i]["fen"]) for i in sel], mi)
    base = dict(zip([int(i) for i in sel], base))
    before = np.array([base[m[0]] for m in meta_e])
    drop = (before - after) / np.maximum(before, 1e-6)
    off = np.array([not m[4] for m in meta_e])
    boots = []
    for _ in range(1000):
        a = rng.choice(np.where(off)[0], off.sum())
        bb = rng.choice(np.where(~off)[0], (~off).sum())
        boots.append(drop[a].mean() - drop[bb].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res["cause"] = {"positions": int(len(sel)), "edits_switching_off": int(off.sum()), "edits_keeping_C": int((~off).sum()),
                    "mean_mass_before": float(before.mean()),
                    "relative_drop_when_switched_off": float(drop[off].mean()) if off.any() else None,
                    "relative_drop_when_kept": float(drop[~off].mean()) if (~off).any() else None,
                    "difference_ci95": [float(lo), float(hi)]}
    res["cause"]["pass"] = bool(off.sum() >= 20 and drop[off].mean() >= 0.30 and lo > 0)
    log("cause:", json.dumps(res["cause"]))
    # ---- worth: t1's value of the human's move, with and without M, where t1 plays M
    sel2 = np.where(c & (E["T"][:, mi] > 0.5))[0]
    ab, at, keep = [], [], []
    for i in sel2:
        b = chess.Board(items[i]["fen"])
        m = chess.Move.from_uci(played[i])
        if m not in b.legal_moves or played[i] == t1_top[i]:
            continue
        bh, bt = b.copy(stack=False), b.copy(stack=False)
        bh.push(m)
        bt.push(chess.Move.from_uci(t1_top[i]))
        if bh.is_game_over() or bt.is_game_over():
            continue
        ab.append(bh)
        at.append(bt)
        keep.append(int(i))
    keep = np.array(keep)
    _, q_after, _ = policy_eval(net, ab)
    _, q_t1, _ = policy_eval(net, at)
    loss = q_after - q_t1     # regret: the opponent's value after the human's move minus after t1's move (K11b fix (a))
    hm = E["H"][keep, mi] > 0.5
    boots = []
    for _ in range(2000):
        a = rng.choice(np.where(~hm)[0], (~hm).sum())
        bb = rng.choice(np.where(hm)[0], hm.sum())
        boots.append(loss[a].mean() - loss[bb].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res["worth"] = {"positions": int(len(keep)), "human_played_M": int(hm.sum()),
                    "mean_loss_without_M": float(loss[~hm].mean()), "mean_loss_with_M": float(loss[hm].mean()) if hm.any() else None,
                    "difference": float(loss[~hm].mean() - loss[hm].mean()) if hm.any() else None, "ci95": [float(lo), float(hi)],
                    "units": "t1 expected score (win minus loss, -1..1) the human's move gives up against t1's own move; "
                             "positions where the human played t1's move are left out"}
    res["worth"]["pass"] = bool(hm.sum() >= 20 and lo > 0)
    log("worth:", json.dumps(res["worth"]))
    # ---- examples: C holds, t1 plays M, the human did not, largest loss
    order = np.argsort(-loss * (~hm))
    res["examples"] = [{"fen": items[keep[j]]["fen"], "t1_move": t1_top[keep[j]], "human_move": played[keep[j]],
                        "human_elo": int(E["elo"][keep[j]]), "loss": float(loss[j])} for j in order[:6]]
    os.makedirs(OUT, exist_ok=True)
    tag = "c" if os.environ.get("MOVES") == "c" else os.environ.get("MODE", "a")
    json.dump(res, open(os.path.join(OUT, f"test_{tag}{K}.json"), "w"), indent=1)


def wdl_eval(net, boards, batch=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(boards), batch):
            _, w = net(torch.tensor(np.stack([encode(b) for b in boards[i:i + batch]])))
            out.append(w.numpy())
    return np.concatenate(out) if out else np.zeros((0, 3), dtype=np.float32)


def cmd_freshq():
    """K11e: t1's WDL on the fresh positions; confirmation of e1/e2; draw probability and regret where the rule fires."""
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    Fz = dict(np.load(os.path.join(WORK, "fresh.npz")))
    fm = json.load(open(os.path.join(WORK, "fresh_meta.json")))
    keys, items, tops, elo = fm["keys"], fm["items"], fm["tops"], Fz["elo"]
    boards = [chess.Board(it["fen"]) for it in items]
    net = LeelaTransformer(os.path.join(KDATA, "chess", "t1-256x10.pb.gz"))
    path = os.path.join(WORK, "fresh_wdl.npy")
    if os.path.exists(path):
        W = np.load(path)
    else:
        W = wdl_eval(net, boards)
        np.save(path, W)
    q = W[:, 0] - W[:, 2]
    mi = keys.index("queen trade (captures or offers the queen)")
    T, H, L, R = (Fz[k][:, mi] for k in ("T", "H", "L", "R"))
    res = {}
    for rid, c, direction in (("e1", q <= -0.5, +1), ("e2", q >= 0.5, -1)):
        d = T - H
        n, nn = int(c.sum()), int((~c).sum())
        z = (d[c].mean() - d[~c].mean()) / np.sqrt(d[c].var() / n + d[~c].var() / nn)
        bands = {bn: (float(T[c & f(elo)].mean()), float(H[c & f(elo)].mean()), int((c & f(elo)).sum())) for bn, f in BANDS}
        r = {"situation": "t1 q <= -0.5 (lost)" if rid == "e1" else "t1 q >= +0.5 (winning)", "move_type": keys[mi], "support": n,
             "t1": float(T[c].mean()), "human": float(H[c].mean()), "LD2": float(L[c].mean()), "744706": float(R[c].mean()),
             "ratio": float(T[c].mean() / max(H[c].mean(), 1e-9)), "interaction_z": float(z), "bands": bands}
        if rid == "e1":
            r["confirmed"] = bool(r["ratio"] >= 1.5 and z > 3 and r["LD2"] > r["human"] and r["744706"] > r["human"]
                                  and all(t > h for t, h, _ in bands.values()))
        else:
            r["confirmed"] = bool(r["ratio"] <= 0.85 and all(t < h for t, h, _ in bands.values()) and r["744706"] < r["human"])
        res[rid] = r
        log(f"{rid} {'CONFIRMED' if r['confirmed'] else 'not confirmed'}: {r['situation']} -> queen trade: t1 {r['t1']:.4f} human "
            f"{r['human']:.4f} (x{r['ratio']:.2f}) LD2 {r['LD2']:.4f} 744706 {r['744706']:.4f} z {z:.1f} n={n} "
            + " ".join(f"{bn}: {t:.4f}/{h:.4f}" for bn, (t, h, k) in bands.items()))
    # mechanism and worth where e1 fires: t1 trades queens in a lost position and the human did something else
    fire = np.where((q <= -0.5) & (T > 0.5) & (np.array(tops["T"]) != np.array(tops["H"])))[0]
    bt, bh, kept = [], [], []
    for i in fire:
        b1, b2 = boards[i].copy(stack=False), boards[i].copy(stack=False)
        b1.push_uci(tops["T"][i])
        b2.push_uci(tops["H"][i])
        if b1.is_game_over() or b2.is_game_over():
            continue
        bt.append(b1); bh.append(b2); kept.append(int(i))
    Wt, Wh = wdl_eval(net, bt), wdl_eval(net, bh)
    # after a move the opponent is to move: the mover's win = opponent's loss (index 2), draw stays index 1
    mover_t = np.stack([Wt[:, 2], Wt[:, 1], Wt[:, 0]], 1)
    mover_h = np.stack([Wh[:, 2], Wh[:, 1], Wh[:, 0]], 1)
    rng = np.random.default_rng(3)
    def ci(x):
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    dd = mover_t[:, 1] - mover_h[:, 1]
    dq = (mover_t[:, 0] - mover_t[:, 2]) - (mover_h[:, 0] - mover_h[:, 2])
    dl = mover_t[:, 2] - mover_h[:, 2]
    res["mechanism"] = {"positions": len(kept), "draw_prob_after_t1_trade": float(mover_t[:, 1].mean()),
                        "draw_prob_after_human_move": float(mover_h[:, 1].mean()), "draw_gain": float(dd.mean()), "draw_gain_ci95": ci(dd),
                        "loss_prob_change": float(dl.mean()), "loss_prob_change_ci95": ci(dl),
                        "expected_score_gain": float(dq.mean()), "expected_score_gain_ci95": ci(dq),
                        "mover_wdl_before": W[kept].mean(0).tolist() if kept else None}
    log("mechanism:", json.dumps(res["mechanism"]))
    order = np.argsort(-dd)
    res["examples"] = [{"fen": items[kept[j]]["fen"], "t1_move": tops["T"][kept[j]], "human_move": tops["H"][kept[j]],
                        "human_elo": int(elo[kept[j]]), "t1_q_before": float(q[kept[j]]),
                        "draw_after_t1": float(mover_t[j, 1]), "draw_after_human": float(mover_h[j, 1]), "site": items[kept[j]]["site"]}
                       for j in order[:8]]
    json.dump(res, open(os.path.join(OUT, "queen_trade_fresh.json"), "w"), indent=1)


CONFIRM = [  # K11d, fixed before the fresh data existed (CHANGES.md)
    ("c0", "in_check", "eq", 0, "c", "pawn lever (push attacks an enemy pawn)"),
    ("c1", "material", "le", -1, "c", "queen trade (captures or offers the queen)"),
    ("c2", "in_check", "eq", 0, "c", "pawn sacrifice (pushed pawn en prise)"),
    ("c3", "enemy king in the centre (d/e file)", "king", 1, "c", "pawn lever (push attacks an enemy pawn)"),
    ("c4", "n_pieces", "le", 20, "c", "pawn sacrifice (pushed pawn en prise)"),
    ("b1", "material", "le", -1, "b", "offers a like-for-like trade"),
    ("b2", "own king off its home ranks", "king", 1, "b", "offers material (sacrifice-like)"),
]


def fresh_positions(n=16000, skip=12000, seed=11):
    import random
    import chess.pgn
    from kepler.targets.chess.data import _text_stream
    rng = random.Random(seed)
    stream = _text_stream(os.path.join(KDATA, "chess", "lichess_2013-01.pgn.zst"))
    for _ in range(skip):
        chess.pgn.skip_game(stream)
    out = []
    while len(out) < n:
        g = chess.pgn.read_game(stream)
        if g is None:
            break
        moves = list(g.mainline_moves())
        if len(moves) < 14:
            continue
        plies = sorted(rng.sample(range(10, len(moves) - 1), min(2, len(moves) - 11)))
        b = g.board()
        k = 0
        for i, m in enumerate(moves):
            if k < len(plies) and i == plies[k]:
                if not b.is_game_over():
                    out.append({"fen": b.fen(), "played": m.uci(), "white_elo": g.headers.get("WhiteElo", "?"),
                                "black_elo": g.headers.get("BlackElo", "?"), "site": g.headers.get("Site", "")})
                k += 1
            b.push(m)
    return out[:n]


def cmd_fresh():
    from kepler.targets.chess.concepts import concepts
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    items = fresh_positions()
    log(f"{len(items)} fresh positions from games after the first 12,000")
    boards = [chess.Board(it["fen"]) for it in items]
    tops = {}
    for nm, net in (("T", LeelaTransformer(os.path.join(KDATA, "chess", "t1-256x10.pb.gz"))),
                    ("L", LeelaNet(os.path.join(KDATA, "chess", "LD2.pb.gz"))),
                    ("R", LeelaNet(os.path.join(KDATA, "chess", "744706.pb.gz")))):
        tops[nm], _, _ = policy_eval(net, boards)
        log(nm, "done")
    tops["H"] = [it["played"] for it in items]
    feats = {}
    for nm, mv in tops.items():
        fb = [move_feats(b, chess.Move.from_uci(m)) for b, m in zip(boards, mv)]
        fc = [move_feats_c(b, chess.Move.from_uci(m)) for b, m in zip(boards, mv)]
        feats[nm] = [{**a, **c} for a, c in zip(fb, fc)]
    keys = list(feats["T"][0])
    cond = []
    for b in boards:
        c = concepts(b)
        k = king_conditions(b)
        cond.append({"material": c["material"], "in_check": c["in_check"], "n_pieces": c["n_pieces"], **k})
    elo = [int(e) if str(e).isdigit() else 0 for e in [it["white_elo"] if b.turn else it["black_elo"] for it, b in zip(items, boards)]]
    np.savez(os.path.join(WORK, "fresh.npz"), **{nm: np.array([[f[k] for k in keys] for f in feats[nm]], dtype=np.float32) for nm in feats},
             elo=np.array(elo))
    json.dump({"keys": keys, "cond": cond, "items": items, "tops": tops}, open(os.path.join(WORK, "fresh_meta.json"), "w"))
    log("saved")


def cmd_confirm():
    Fz = dict(np.load(os.path.join(WORK, "fresh.npz")))
    fm = json.load(open(os.path.join(WORK, "fresh_meta.json")))
    keys, cond, elo = fm["keys"], fm["cond"], Fz["elo"]
    res = {}
    for rid, name, kind, v, vocab, mk in CONFIRM:
        x = np.array([c[name] for c in cond], dtype=float)
        c = {"eq": x == v, "le": x <= v, "king": x == v}[kind]
        mi = keys.index(mk)
        T, H, L, R = (Fz[k][:, mi] for k in ("T", "H", "L", "R"))
        d = T - H
        n, nn = int(c.sum()), int((~c).sum())
        inter = d[c].mean() - d[~c].mean()
        z = inter / np.sqrt(d[c].var() / n + d[~c].var() / nn)
        bands = {bn: (float(T[c & f(elo)].mean()), float(H[c & f(elo)].mean()), int((c & f(elo)).sum())) for bn, f in BANDS}
        r = {"situation": f"{name} {kind} {v}", "move_type": mk, "support": n, "t1": float(T[c].mean()), "human": float(H[c].mean()),
             "LD2": float(L[c].mean()), "744706": float(R[c].mean()), "ratio": float(T[c].mean() / max(H[c].mean(), 1e-9)),
             "outside": {"t1": float(T[~c].mean()), "human": float(H[~c].mean())}, "interaction_z": float(z), "bands": bands}
        r["confirmed"] = bool(z > 3 and r["ratio"] >= 1.3 and r["LD2"] > r["human"] and r["744706"] > r["human"]
                              and all(t > h for t, h, _ in bands.values()))
        res[rid] = r
        log(f"{rid} {'CONFIRMED' if r['confirmed'] else 'not confirmed'}: [{r['situation']}] -> [{mk}] t1 {r['t1']:.3f} human "
            f"{r['human']:.3f} (x{r['ratio']:.2f}) LD2 {r['LD2']:.3f} 744706 {r['744706']:.3f} z {z:.1f} n={n} bands "
            + " ".join(f"{bn}: {t:.3f}/{h:.3f} (n={k})" for bn, (t, h, k) in bands.items()))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "confirm_fresh.json"), "w"), indent=1)


if __name__ == "__main__":
    {"eval": cmd_eval, "evalc": cmd_evalc, "mine": cmd_mine, "test": cmd_test, "fresh": cmd_fresh,
     "confirm": cmd_confirm, "freshq": cmd_freshq}[sys.argv[1]]()
