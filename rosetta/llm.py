"""Rosetta's cheap-model roles, live through Luminal (DeepSeek V4.1 Flash):
- propose(): the volume proposer. Shown a unit's evidence (its most and least active squares in real positions, and
  the exhaustive search's best attempts), it writes candidate definitions in the restricted expression language.
  Every candidate is scored by code; the model's own opinion of a candidate counts for nothing.
- teach(): the simulation test. A fresh call gets only a plain-English definition and new positions, and must name
  the squares where the unit is active."""
import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

import chess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_env():
    p = os.path.join(ROOT, "lab", ".env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
from lab import luminal  # noqa: E402

MEANING = {
    "empty": "no piece", "own_P": "our pawn (side to move = 'our')", "opp_P": "their pawn", "rel_rank": "rank from our side, 0 = our back rank",
    "file": "file 0=a..7=h", "edge_dist": "distance to nearest edge file", "center_dist": "Chebyshev distance to the centre",
    "att_us": "number of our pieces attacking the square", "att_them": "number of their pieces attacking it",
    "min_att_us": "value of our cheapest attacker (0 none, 1 pawn, 3 minor, 5 rook, 9 queen, 100 king)", "min_att_them": "same for them",
    "pawn_att_us": "attacked by our pawn", "pawn_att_them": "attacked by their pawn", "dist_own_king": "king distance to our king",
    "dist_opp_king": "king distance to their king", "own_king_zone": "adjacent to our king", "opp_king_zone": "adjacent to their king",
    "piece_reach": "squares the piece standing here attacks", "pinned": "piece here is pinned", "defended": "piece here is defended by its own side",
    "attacked": "piece here is attacked by the other side", "passed_pawn": "pawn here is passed", "isolated_pawn": "pawn here is isolated",
    "doubled_pawn": "pawn here is doubled", "safe_from_opp_pawns": "no enemy pawn can ever attack this square",
    "safe_from_own_pawns": "none of our pawns can ever attack it", "open_file": "no pawns on the file", "semiopen_us": "no own pawn on the file",
    "semiopen_them": "no enemy pawn on the file", "light_square": "light square", "behind_own_pawn": "an own pawn is ahead on this file",
    "in_front_of_opp_pawn": "an enemy pawn is behind it on this file (from our view, ahead of us)", "hanging_opp": "their piece here is attacked by us and undefended",
    "hanging_own": "our piece here is attacked and undefended", "opp_att_by_lower": "their piece here is attacked by a cheaper piece of ours",
    "own_att_by_lower": "our piece here is attacked by a cheaper piece of theirs", "mob_us": "number of our legal moves landing here",
    "mob_them": "number of their moves landing here", "check_from_us": "one of our moves to this square gives check",
    "check_from_them": "one of their moves to this square would give check", "control": "att_us - att_them",
    "outnumbered_own": "our piece here has more attackers than defenders", "outnumbered_opp": "their piece here has more attackers than defenders",
    "la_m1_from": "from-square of the network's own best move", "la_m1_to": "to-square of the network's best move",
    "la_m2_from": "from-square of their best reply (network's own line)", "la_m2_to": "to-square of their best reply",
    "la_m3_from": "from-square of the network's follow-up move", "la_m3_to": "to-square of its follow-up",
    "la_m1_attacks": "squares attacked by our piece after the best move", "la_m3_attacks": "squares attacked by our piece after the follow-up",
}
for p in "NBRQK":
    MEANING[f"own_{p}"] = f"our {'knight bishop rook queen king'.split()['NBRQK'.index(p)]}"
    MEANING[f"opp_{p}"] = f"their {'knight bishop rook queen king'.split()['NBRQK'.index(p)]}"


def real_name(k, fen):
    flip = fen.split()[1] == "b"
    return chess.square_name(chess.square_mirror(k) if flip else k)


def evidence(fens, act, n_pos=8):
    """act: [n, 64] centered unit activity for the positions in fens."""
    order = np.argsort(-act.max(1))[:n_pos]
    lines = []
    for i in order:
        a = act[i]; top = np.argsort(-a)[:5]; low = np.argsort(a)[:3]
        lines.append(f"FEN {fens[i]} | most active: " + ", ".join(f"{real_name(k, fens[i])} ({a[k]:+.2f})" for k in top) +
                     " | least active: " + ", ".join(f"{real_name(k, fens[i])} ({a[k]:+.2f})" for k in low))
    return "\n".join(lines)


def propose(names, fens, act, attempts, n=15, effort="high"):
    prims = "\n".join(f"- {nm}: {MEANING.get(nm, '')}" for nm in names)
    tried = "\n".join(f"- {t}  (held-out correlation {f:+.3f})" for t, f in attempts)
    prompt = f"""A unit inside a chess-playing neural network has one activity value per square. Below are positions
where it is most active, with its most and least active squares (real square names; "our" = the side to move).

{evidence(fens, act)}

Write {n} candidate definitions of when this unit is active on a square, as expressions over these per-square
primitives (each is a number or 0/1 for that square):
{prims}

Allowed: the primitive names, numbers, and, or, not, + - *, comparisons. At most 6 primitive names per expression.
An exhaustive search over conjunctions of up to three conditions already tried these, so go beyond them (sums,
counts, disjunctions, look-ahead squares):
{tried}

Answer with a JSON list of strings only, no commentary."""
    r = luminal.complete({"messages": [{"role": "user", "content": prompt}], "max_tokens": 6000}, effort=effort)
    txt = r["choices"][0]["message"]["content"] or ""
    m = re.search(r"\[.*\]", txt, re.S)
    try:
        return [e for e in json.loads(m.group(0)) if isinstance(e, str)] if m else []
    except Exception:
        return []


def propose_many(jobs, workers=6):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda j: propose(*j), jobs))


def board_text(fen):
    """The position drawn as an 8 x 8 grid with coordinates (uppercase = White), plus who is to move."""
    b = chess.Board(fen); rows = []
    for r in range(7, -1, -1):
        rows.append(f"{r + 1} " + " ".join((b.piece_at(chess.square(f, r)).symbol() if b.piece_at(chess.square(f, r)) else ".") for f in range(8)))
    rows.append("  a b c d e f g h")
    return "\n".join(rows) + f"\n{'White' if b.turn else 'Black'} to move ('us')."


def teach(sentence, fens, k=5, effort="high", boards=False):
    """Fresh call: from the sentence alone, name the k squares where the unit is most active in each position."""
    pos = "\n\n".join(f"Position {i + 1}:\n{board_text(f)}" if boards else f"{i + 1}. {f}" for i, f in enumerate(fens))
    prompt = f"""A unit in a chess network is active on squares described by this rule (the side to move is "us"):
"{sentence}"
For each position below, list the {k} squares where the rule most applies (real square names like e4), best first.
{pos}
Answer with a JSON list of lists of square names, one list per position, nothing else."""
    r = luminal.complete({"messages": [{"role": "user", "content": prompt}], "max_tokens": 8000}, effort=effort)
    txt = r["choices"][0]["message"]["content"] or ""
    m = re.search(r"\[\s*\[.*\]\s*\]", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else []
    except Exception:
        return []
