"""The five domains of the starting vocabulary (bench/accept/VOCABULARY.md): base features, operations, root types and
the maximum length L of each. Every value carries the batch of inputs as its first axis (numpy arrays), except token
sequences and network weights, whose items differ in size and are held as lists.
"""
import numpy as np

from .core import Domain, CONST as C

EPS = 1e-9


def _f(x):
    return np.asarray(x, dtype=np.float64)


# ====================================================================================================== chess
def chess_domain():
    """ctx: {"P": [N, 64, 55] per-square features in the mover's frame, "G": [N, 5] position features} (featurize)."""
    from .chess_board import DESC_NAMES, EXTRA_NAMES, SEE_NAMES
    d = Domain("chess", {"Sq": "a value on each of the 64 squares, mover's frame", "Pos": "one value per position"},
               roots=("Sq", "Pos"), L=6)
    for i, n in enumerate(list(DESC_NAMES) + list(EXTRA_NAMES) + list(SEE_NAMES)):
        d.atom(n, "Sq", lambda ctx, i=i: _f(ctx["P"][:, :, i]))
    for j, n in enumerate(POS_NAMES):
        d.atom(n, "Pos", lambda ctx, j=j: _f(ctx["G"][:, j]))
    for T in ("Sq", "Pos"):
        d.op("not", [T], T, lambda a: 1.0 - (a > 0), "1 where the value is 0 or less, else 0")
        d.op("and", [T, T], T, lambda a, b: np.minimum(a, b), "minimum")
        d.op("or", [T, T], T, lambda a, b: np.maximum(a, b), "maximum")
        d.op("add", [T, T], T, lambda a, b: a + b)
        d.op("sub", [T, T], T, lambda a, b: a - b)
        d.op("mul", [T, T], T, lambda a, b: a * b)
        d.op("gt", [T, T], T, lambda a, b: (a > b).astype(float))
        d.op("add", [T, C], T, lambda a, c: a + c)
        d.op("mul", [T, C], T, lambda a, c: a * c)
        d.op("ge", [T, C], T, lambda a, c: (a >= c).astype(float))
        d.op("le", [T, C], T, lambda a, c: (a <= c).astype(float))
        d.op("eq", [T, C], T, lambda a, c: (np.abs(a - c) < 1e-6).astype(float))
    # spatial: the value at the neighbouring square (0 off the board); "north" is toward the opponent
    for name, (dr, df) in {"north": (1, 0), "south": (-1, 0), "east": (0, 1), "west": (0, -1), "ne": (1, 1),
                           "nw": (1, -1), "se": (-1, 1), "sw": (-1, -1)}.items():
        d.op(name, ["Sq"], "Sq", lambda a, dr=dr, df=df: _shift(a, dr, df), f"value at the square ({dr:+d} rank, {df:+d} file)")
    d.op("king_any", ["Sq"], "Sq", lambda a: _any(a, KING_STEPS), "maximum over the 8 king-step neighbours")
    d.op("knight_any", ["Sq"], "Sq", lambda a: _any(a, KNIGHT_STEPS), "maximum over the knight-jump neighbours")
    d.op("file_any", ["Sq"], "Sq", lambda a: np.repeat(a.reshape(-1, 8, 8).max(1, keepdims=True), 8, 1).reshape(-1, 64),
         "maximum over the same file")
    d.op("rank_any", ["Sq"], "Sq", lambda a: np.repeat(a.reshape(-1, 8, 8).max(2, keepdims=True), 8, 2).reshape(-1, 64),
         "maximum over the same rank")
    d.op("argmax_sq", ["Sq"], "Sq", lambda a: _onehot(np.argmax(a, 1)), "1 on the square with the largest value (first by index)")
    d.op("argmin_sq", ["Sq"], "Sq", lambda a: _onehot(np.argmin(a, 1)), "1 on the square with the smallest value")
    d.op("after", ["Sq", "Sq", "Pos"], "Pos", None,
         "(after FROM TO EXPR): play the legal move from the square where FROM is largest to the square where TO is "
         "largest (promotion to a queen), then evaluate EXPR, an expression of base features only, in the new position, "
         "from the view of the side then to move; 0 where there is no such legal move. A special form: EXPR is "
         "evaluated in the child position, not the current one")
    d.op("count", ["Sq"], "Pos", lambda a: (a > 0).sum(1).astype(float), "number of squares where the value is above 0")
    d.op("sum", ["Sq"], "Pos", lambda a: a.sum(1))
    d.op("max", ["Sq"], "Pos", lambda a: a.max(1))
    d.op("min", ["Sq"], "Pos", lambda a: a.min(1))
    d.op("bcast", ["Pos"], "Sq", lambda p: np.repeat(np.asarray(p, float)[:, None], 64, 1), "the position value on every square")
    return d


POS_NAMES = ["in_check", "n_legal", "castle_us", "castle_them", "halfmove"]
KING_STEPS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
KNIGHT_STEPS = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]


def _onehot(idx):
    out = np.zeros((len(idx), 64))
    out[np.arange(len(idx)), idx] = 1.0
    return out


def _shift(a, dr, df):
    """out[r, f] = a[r + dr, f + df] (0 off the board)."""
    a = np.asarray(a, float).reshape(-1, 8, 8)
    out = np.zeros_like(a)
    r0, r1 = max(0, -dr), min(8, 8 - dr)
    f0, f1 = max(0, -df), min(8, 8 - df)
    out[:, r0:r1, f0:f1] = a[:, r0 + dr:r1 + dr, f0 + df:f1 + df]
    return out.reshape(-1, 64)


def _any(a, steps):
    return np.max(np.stack([_shift(a, dr, df) for dr, df in steps]), 0)


def featurize(fens, procs=1):
    """FENs -> ctx for the chess domain (per-square features and position features)."""
    import chess
    from .chess_board import descriptors, static_extra, see_features
    P, G = [], []
    for f in fens:
        b = chess.Board(f)
        P.append(np.concatenate([descriptors(b), static_extra(b), see_features(b)], 1))
        us = b.turn
        G.append([float(b.is_check()), float(b.legal_moves.count()), float(b.has_castling_rights(us)),
                  float(b.has_castling_rights(not us)), float(b.halfmove_clock)])
    return {"P": np.asarray(P, np.float32), "G": np.asarray(G, np.float32), "fens": list(fens)}


def child_fens(fens, frm, to):
    """The positions after the legal move from argmax(frm) to argmax(to) (mover's frame), or None."""
    import chess
    out = []
    for f, a, b in zip(fens, np.argmax(frm, 1), np.argmax(to, 1)):
        board = chess.Board(f)
        flip = board.turn == chess.BLACK
        sq_from = chess.square_mirror(int(a)) if flip else int(a)
        sq_to = chess.square_mirror(int(b)) if flip else int(b)
        mv = next((m for m in board.legal_moves if m.from_square == sq_from and m.to_square == sq_to and
                   (m.promotion in (None, chess.QUEEN))), None)
        if mv is None:
            out.append(None)
        else:
            board.push(mv)
            out.append(board.fen())
    return out


# ====================================================================================================== numeric sequences
def seq_domain():
    """ctx: {"x": [N, n] array of numbers (integers or reals)}."""
    d = Domain("seq", {"Seq": "a value at each position of the sequence", "Num": "one value per sequence"},
               roots=("Seq", "Num"), L=7)
    d.atom("x", "Seq", lambda ctx: _f(ctx["x"]), "the input sequence")
    d.atom("idx", "Seq", lambda ctx: np.tile(np.arange(ctx["x"].shape[1], dtype=float), (ctx["x"].shape[0], 1)), "positions 0..n-1")
    d.atom("n", "Num", lambda ctx: np.full(ctx["x"].shape[0], float(ctx["x"].shape[1])), "the length")
    for T in ("Seq", "Num"):
        d.op("add", [T, T], T, lambda a, b: a + b)
        d.op("sub", [T, T], T, lambda a, b: a - b)
        d.op("mul", [T, T], T, lambda a, b: a * b)
        d.op("max2", [T, T], T, np.maximum)
        d.op("min2", [T, T], T, np.minimum)
        d.op("gt", [T, T], T, lambda a, b: (a > b).astype(float))
        d.op("eq", [T, T], T, lambda a, b: (np.abs(a - b) < 1e-9).astype(float))
        d.op("neg", [T], T, lambda a: -a)
        d.op("abs", [T], T, np.abs)
        d.op("add", [T, C], T, lambda a, c: a + c)
        d.op("mul", [T, C], T, lambda a, c: a * c)
        d.op("ge", [T, C], T, lambda a, c: (a >= c).astype(float))
        d.op("le", [T, C], T, lambda a, c: (a <= c).astype(float))
        d.op("relu", [T], T, lambda a: np.maximum(a, 0.0), "max(a, 0)")
        d.op("max2", [T, C], T, lambda a, c: np.maximum(a, c))
        d.op("min2", [T, C], T, lambda a, c: np.minimum(a, c))
        d.op("mod", [T, C], T, lambda a, c: np.mod(a, c) if c else a * np.nan, "remainder (integers)")
        d.op("floordiv", [T, C], T, lambda a, c: np.floor_divide(a, c) if c else a * np.nan)
    d.op("where", ["Seq", "Seq", "Seq"], "Seq", lambda c, a, b: np.where(c > 0, a, b))
    d.op("rank", ["Seq"], "Seq", lambda a: np.argsort(np.argsort(a, 1, kind="stable"), 1, kind="stable").astype(float),
         "0-based ascending rank, ties by position")
    d.op("sort", ["Seq"], "Seq", lambda a: np.sort(a, 1), "ascending")
    d.op("rev", ["Seq"], "Seq", lambda a: a[:, ::-1].copy())
    d.op("cumsum", ["Seq"], "Seq", lambda a: np.cumsum(a, 1))
    d.op("cummax", ["Seq"], "Seq", lambda a: np.maximum.accumulate(a, 1))
    d.op("cummin", ["Seq"], "Seq", lambda a: np.minimum.accumulate(a, 1))
    d.op("lag", ["Seq", C], "Seq", lambda a, c: _lag(a, int(c)), "the value c positions earlier (0 before the start)")
    d.op("bcast", ["Num"], "Seq", lambda v: np.asarray(v, float)[:, None], "the value at every position (broadcast)")
    d.op("sum", ["Seq"], "Num", lambda a: a.sum(1))
    d.op("max", ["Seq"], "Num", lambda a: a.max(1))
    d.op("min", ["Seq"], "Num", lambda a: a.min(1))
    d.op("mean", ["Seq"], "Num", lambda a: a.mean(1))
    d.op("argmax", ["Seq"], "Num", lambda a: a.argmax(1).astype(float), "first position of the maximum")
    d.op("argmin", ["Seq"], "Num", lambda a: a.argmin(1).astype(float))
    d.op("count", ["Seq"], "Num", lambda a: (a > 0).sum(1).astype(float), "positions where the value is above 0")
    d.op("at", ["Seq", "Num"], "Num", lambda a, i: a[np.arange(len(a)), np.clip(np.asarray(i, int), 0, a.shape[1] - 1)],
         "the value at a position (clipped to the sequence)")
    d.op("at", ["Seq", C], "Num", lambda a, c: a[:, int(np.clip(c, 0, a.shape[1] - 1))])
    return d


def _lag(a, c):
    out = np.zeros_like(a)
    if c <= 0:
        return a.copy()
    out[:, c:] = a[:, :-c] if c < a.shape[1] else 0
    return out


# ====================================================================================================== token sequences (RASP)
class _Fn:
    def __init__(self, f, name):
        self.f, self.name = f, name

    def __call__(self, *a):
        try:
            return self.f(*a)
        except Exception:  # noqa: BLE001 - undefined value, as in RASP
            return None


def tok_domain():
    """RASP (Weiss et al. 2021) as Tracr implements it: s-ops (a value per token), selectors (query x key booleans),
    select, aggregate, selector_width, map and sequence_map. ctx: {"tokens": list of token lists}.
    aggregate is the numerical mean of the selected values (0 if none); aggregate_cat returns the selected value
    when every selected key holds the same value, else None (Tracr's categorical semantics)."""
    d = Domain("tok", {"SOp": "a value per token", "Sel": "a selector: query x key booleans", "Pred": "a comparison",
                       "Fn1": "a one-argument function", "Fn2": "a two-argument function"},
               roots=("SOp",), L=12, const_values="token")
    d.atom("tokens", "SOp", lambda ctx: [list(t) for t in ctx["tokens"]])
    d.atom("indices", "SOp", lambda ctx: [list(range(len(t))) for t in ctx["tokens"]])
    preds = {"EQ": lambda k, q: k == q, "NEQ": lambda k, q: k != q, "LT": lambda k, q: k < q, "LEQ": lambda k, q: k <= q,
             "GT": lambda k, q: k > q, "GEQ": lambda k, q: k >= q, "TRUE": lambda k, q: True, "FALSE": lambda k, q: False}
    for n, f in preds.items():
        d.atom(n, "Pred", lambda ctx, f=f: f, f"predicate(key, query): {n}")
    f1 = {"id": lambda x: x, "inc": lambda x: x + 1, "dec": lambda x: x - 1, "neg": lambda x: -x,
          "not1": lambda x: not x, "abs1": abs}
    for n, f in f1.items():
        d.atom(n, "Fn1", lambda ctx, f=f, n=n: _Fn(f, n))
    f2 = {"plus": lambda a, b: a + b, "minus": lambda a, b: a - b, "times": lambda a, b: a * b,
          "equal": lambda a, b: a == b, "less": lambda a, b: a < b, "greater": lambda a, b: a > b,
          "larger": max, "smaller": min, "both": lambda a, b: bool(a and b), "either": lambda a, b: bool(a or b)}
    for n, f in f2.items():
        d.atom(n, "Fn2", lambda ctx, f=f, n=n: _Fn(f, n))
    d.op("eqc", [C], "Fn1", lambda c: _Fn(lambda x: x == c, f"=={c}"), "x == c")
    d.op("ltc", [C], "Fn1", lambda c: _Fn(lambda x: x < c, f"<{c}"), "x < c")
    d.op("gtc", [C], "Fn1", lambda c: _Fn(lambda x: x > c, f">{c}"), "x > c")
    d.op("addc", [C], "Fn1", lambda c: _Fn(lambda x: x + c, f"+{c}"), "x + c")
    d.op("mulc", [C], "Fn1", lambda c: _Fn(lambda x: x * c, f"*{c}"), "x * c")
    d.op("modc", [C], "Fn1", lambda c: _Fn(lambda x: x % c, f"%{c}"), "x mod c")
    d.op("subfrom", [C], "Fn1", lambda c: _Fn(lambda x: c - x, f"{c}-"), "c - x")
    d.op("select", ["SOp", "SOp", "Pred"], "Sel", _select, "select(keys, queries, predicate)")
    d.op("sel_and", ["Sel", "Sel"], "Sel", lambda a, b: [x & y for x, y in zip(a, b)])
    d.op("sel_or", ["Sel", "Sel"], "Sel", lambda a, b: [x | y for x, y in zip(a, b)])
    d.op("sel_not", ["Sel"], "Sel", lambda a: [~x for x in a])
    d.op("aggregate", ["Sel", "SOp"], "SOp", _aggregate_num, "numerical: the mean of the selected values (0 if none)")
    d.op("aggregate_cat", ["Sel", "SOp"], "SOp", _aggregate_cat, "categorical: the selected value (None if none or mixed)")
    d.op("selector_width", ["Sel"], "SOp", lambda s: [list(m.sum(1).astype(int)) for m in s], "number selected per query")
    d.op("map", ["Fn1", "SOp"], "SOp", lambda f, s: [[None if v is None else f(v) for v in row] for row in s])
    d.op("sequence_map", ["Fn2", "SOp", "SOp"], "SOp",
         lambda f, a, b: [[None if x is None or y is None else f(x, y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)])
    return d


def _select(keys, queries, pred):
    out = []
    for k, q in zip(keys, queries):
        m = np.zeros((len(q), len(k)), bool)
        for i, qv in enumerate(q):
            for j, kv in enumerate(k):
                try:
                    m[i, j] = kv is not None and qv is not None and bool(pred(kv, qv))
                except TypeError:
                    m[i, j] = False
        out.append(m)
    return out


def _aggregate_num(sel, vals):
    out = []
    for m, v in zip(sel, vals):
        row = []
        for i in range(m.shape[0]):
            js = np.nonzero(m[i])[0]
            xs = [float(v[j]) for j in js if v[j] is not None]
            row.append(sum(xs) / len(js) if len(js) else 0.0)
        out.append(row)
    return out


def _aggregate_cat(sel, vals):
    out = []
    for m, v in zip(sel, vals):
        row = []
        for i in range(m.shape[0]):
            xs = {v[j] for j in np.nonzero(m[i])[0]}
            row.append(next(iter(xs)) if len(xs) == 1 else None)
        out.append(row)
    return out


# ====================================================================================================== fluid fields
FLUID_FIELDS = ["u", "v", "p", "q", "u_prev", "v_prev", "q_prev"]


def fluid_domain():
    """ctx: {name: [N, H, W] arrays} for the fields the adapter declares (velocity u, v; pressure p; a transported
    scalar q; the previous frame's u, v, q). Unit grid spacing, periodic boundaries."""
    d = Domain("fluid", {"Field": "a value at each grid cell", "Num": "one value per frame"}, roots=("Field", "Num"), L=6)
    for n in FLUID_FIELDS:
        d.atom(n, "Field", lambda ctx, n=n: _f(ctx[n]))
    dx = lambda a: (np.roll(a, -1, 2) - np.roll(a, 1, 2)) / 2.0  # noqa: E731
    dy = lambda a: (np.roll(a, -1, 1) - np.roll(a, 1, 1)) / 2.0  # noqa: E731
    lap = lambda a: np.roll(a, 1, 1) + np.roll(a, -1, 1) + np.roll(a, 1, 2) + np.roll(a, -1, 2) - 4 * a  # noqa: E731
    for T in ("Field", "Num"):
        d.op("add", [T, T], T, lambda a, b: a + b)
        d.op("sub", [T, T], T, lambda a, b: a - b)
        d.op("mul", [T, T], T, lambda a, b: a * b)
        d.op("neg", [T], T, lambda a: -a)
        d.op("abs", [T], T, np.abs)
        d.op("sq", [T], T, lambda a: a * a)
        d.op("add", [T, C], T, lambda a, c: a + c)
        d.op("mul", [T, C], T, lambda a, c: a * c)
    d.op("dx", ["Field"], "Field", dx, "central difference along x")
    d.op("dy", ["Field"], "Field", dy, "central difference along y")
    d.op("lap", ["Field"], "Field", lap, "five-point Laplacian")
    d.op("div", ["Field", "Field"], "Field", lambda a, b: dx(a) + dy(b), "divergence of (a, b)")
    d.op("curl", ["Field", "Field"], "Field", lambda a, b: dx(b) - dy(a), "vorticity of (a, b)")
    d.op("adv", ["Field", "Field", "Field"], "Field", lambda a, b, q: a * dx(q) + b * dy(q), "(a, b) . grad q")
    d.op("smooth", ["Field"], "Field", lambda a: sum(np.roll(np.roll(a, i, 1), j, 2) for i in (-1, 0, 1) for j in (-1, 0, 1)) / 9.0,
         "3 x 3 mean")
    d.op("shx", ["Field"], "Field", lambda a: np.roll(a, -1, 2), "value one cell along +x")
    d.op("shy", ["Field"], "Field", lambda a: np.roll(a, -1, 1), "value one cell along +y")
    d.op("relu", ["Field"], "Field", lambda a: np.maximum(a, 0))
    d.op("gt", ["Field", "Field"], "Field", lambda a, b: (a > b).astype(float))
    d.op("ge", ["Field", C], "Field", lambda a, c: (a >= c).astype(float))
    d.op("bcast", ["Num"], "Field", lambda v: np.asarray(v, float)[:, None, None])
    d.op("mean", ["Field"], "Num", lambda a: a.mean((1, 2)))
    d.op("max", ["Field"], "Num", lambda a: a.max((1, 2)))
    d.op("min", ["Field"], "Num", lambda a: a.min((1, 2)))
    d.op("rms", ["Field"], "Num", lambda a: np.sqrt((a * a).mean((1, 2))))
    return d


# ====================================================================================================== network weights
WEIGHT_ATOMS = ["W0", "W1", "W2", "W3", "Wlast", "b0", "b1", "b2", "b3", "blast"]


def weights_domain():
    """The input is another network's weights. ctx: {"nets": list of {"W": [matrices, first layer first],
    "b": [vectors]}}. W0..W3 are the first four layers (missing layers are zero 1 x 1), Wlast the last; Wprod is the
    product of all layers (the network's end-to-end linear part)."""
    d = Domain("weights", {"Mat": "a matrix per network", "Vec": "a vector per network", "Num": "one value per network"},
               roots=("Num", "Vec", "Mat"), L=6)

    def layer(ctx, k, kind):
        out = []
        for net in ctx["nets"]:
            xs = net[kind]
            out.append(np.asarray(xs[k], float) if -len(xs) <= k < len(xs) else np.zeros((1, 1) if kind == "W" else 1))
        return out
    for k in range(4):
        d.atom(f"W{k}", "Mat", lambda ctx, k=k: layer(ctx, k, "W"))
        d.atom(f"b{k}", "Vec", lambda ctx, k=k: layer(ctx, k, "b"))
    d.atom("Wlast", "Mat", lambda ctx: layer(ctx, -1, "W"))
    d.atom("blast", "Vec", lambda ctx: layer(ctx, -1, "b"))
    d.atom("Wprod", "Mat", lambda ctx: [_prod(n["W"]) for n in ctx["nets"]], "W_last ... W_1 W_0")
    d.atom("nlayers", "Num", lambda ctx: np.array([len(n["W"]) for n in ctx["nets"]], float))
    d.atom("nparams", "Num", lambda ctx: np.array([sum(np.size(w) for w in n["W"]) + sum(np.size(b) for b in n["b"])
                                                   for n in ctx["nets"]], float))
    each = lambda f: (lambda *xs: [f(*z) for z in zip(*xs)])  # noqa: E731
    num = lambda f: (lambda *xs: np.array([f(*z) for z in zip(*xs)], float))  # noqa: E731
    d.op("matmul", ["Mat", "Mat"], "Mat", each(lambda a, b: a @ b if a.shape[1] == b.shape[0] else np.full((1, 1), np.nan)))
    d.op("T", ["Mat"], "Mat", each(lambda a: a.T))
    for T in ("Mat", "Vec"):
        d.op("add", [T, T], T, each(lambda a, b: a + b if a.shape == b.shape else np.full(a.shape, np.nan)))
        d.op("sub", [T, T], T, each(lambda a, b: a - b if a.shape == b.shape else np.full(a.shape, np.nan)))
        d.op("mul", [T, T], T, each(lambda a, b: a * b if a.shape == b.shape else np.full(a.shape, np.nan)))
        d.op("abs", [T], T, each(np.abs))
        d.op("relu", [T], T, each(lambda a: np.maximum(a, 0)))
        d.op("scale", [T, C], T, lambda xs, c: [x * c for x in xs])
        d.op("sum", [T], "Num", num(np.sum))
        d.op("mean", [T], "Num", num(np.mean))
        d.op("std", [T], "Num", num(np.std))
        d.op("max", [T], "Num", num(np.max))
        d.op("min", [T], "Num", num(np.min))
        d.op("count_gt", [T, C], "Num", lambda xs, c: np.array([float((x > c).sum()) for x in xs]))
        d.op("frac_pos", [T], "Num", num(lambda a: float((a > 0).mean())))
    d.op("mv", ["Mat", "Vec"], "Vec", each(lambda a, v: a @ v if a.shape[1] == v.shape[0] else np.full(a.shape[0], np.nan)))
    d.op("fro", ["Mat"], "Num", num(lambda a: float(np.linalg.norm(a))))
    d.op("spec", ["Mat"], "Num", num(lambda a: float(np.linalg.svd(a, compute_uv=False)[0])))
    d.op("trace", ["Mat"], "Num", num(lambda a: float(np.trace(a))))
    d.op("cond", ["Mat"], "Num", num(lambda a: float(_cond(a))))
    d.op("eig_max", ["Mat"], "Num", num(lambda a: float(np.max(np.linalg.eigvals(a).real)) if a.shape[0] == a.shape[1] else np.nan))
    d.op("rank", ["Mat", C], "Num", lambda xs, c: np.array([float((np.linalg.svd(x, compute_uv=False) > c).sum()) for x in xs]),
         "singular values above c")
    d.op("sv", ["Mat", C], "Num", lambda xs, c: np.array([_sv(x, int(c)) for x in xs]), "the c-th singular value (0 = largest)")
    d.op("rowsum", ["Mat"], "Vec", each(lambda a: a.sum(1)))
    d.op("colsum", ["Mat"], "Vec", each(lambda a: a.sum(0)))
    d.op("rownorm", ["Mat"], "Vec", each(lambda a: np.linalg.norm(a, axis=1)))
    d.op("diag", ["Mat"], "Vec", each(np.diag))
    d.op("svals", ["Mat"], "Vec", each(lambda a: np.linalg.svd(a, compute_uv=False)))
    d.op("norm", ["Vec"], "Num", num(lambda v: float(np.linalg.norm(v))))
    d.op("argmax", ["Vec"], "Num", num(lambda v: float(np.argmax(v))))
    for op, f in (("add", lambda a, b: a + b), ("sub", lambda a, b: a - b), ("mul", lambda a, b: a * b),
                  ("div", lambda a, b: a / (b + EPS * (b == 0)))):
        d.op(op, ["Num", "Num"], "Num", f)
    d.op("neg", ["Num"], "Num", lambda a: -a)
    d.op("abs", ["Num"], "Num", np.abs)
    d.op("log", ["Num"], "Num", lambda a: np.log(np.abs(a) + EPS))
    d.op("add", ["Num", C], "Num", lambda a, c: a + c)
    d.op("mul", ["Num", C], "Num", lambda a, c: a * c)
    d.op("ge", ["Num", C], "Num", lambda a, c: (a >= c).astype(float))
    d.op("le", ["Num", C], "Num", lambda a, c: (a <= c).astype(float))
    return d


def _prod(Ws):
    out = np.asarray(Ws[0], float)
    for w in Ws[1:]:
        w = np.asarray(w, float)
        if w.shape[1] != out.shape[0]:
            return np.full((1, 1), np.nan)
        out = w @ out
    return out


def _cond(a):
    s = np.linalg.svd(a, compute_uv=False)
    return s[0] / max(s[-1], EPS)


def _sv(a, k):
    s = np.linalg.svd(a, compute_uv=False)
    return float(s[k]) if 0 <= k < len(s) else 0.0


DOMAINS = {"chess": chess_domain, "seq": seq_domain, "tok": tok_domain, "fluid": fluid_domain, "weights": weights_domain}


def get(name):
    return DOMAINS[name]()
