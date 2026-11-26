"""Rosetta core (domain-agnostic): define a network unit's per-site activity with a short program over a library of
primitives, score it on held-out data, and set pass lines from controls.

A program is a conjunction of up to three literals; a literal is a primitive compared with a threshold
("att_them>=2", "not own_P"). Programs contain no weights and no fitted sub-models, so a program cannot be a
disguised copy of the network. Fidelity = Pearson correlation between the program's map and the unit's map over
(positions x sites), on held-out positions.
"""
import ast
import numpy as np


# ------------------------------------------------------------------ literals
def literal_specs(P, names, max_thr=8):
    """Literal specs (k, op, t, name) from the primitive table P [n, S, K] (thresholds from the data, not the unit)."""
    X = P.reshape(-1, P.shape[-1])
    specs = []
    for k, nm in enumerate(names):
        v = X[:, k]
        u = np.unique(v)
        if len(u) <= 1:
            continue
        if len(u) == 2 and set(np.round(u, 6)) <= {0.0, 1.0}:
            specs += [(k, ">=", 0.5, nm), (k, "<", 0.5, f"not {nm}")]
            continue
        if len(u) <= 9:
            ge, le = u[1:], u[:-1]
        else:
            q = np.unique(np.quantile(v, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))
            ge, le = q[1:] if len(q) > 1 else q, q[:-1] if len(q) > 1 else q
        for t in ge[:max_thr]:
            specs.append((k, ">=", float(t), f"{nm}>={t:g}"))
        for t in le[:max_thr]:
            specs.append((k, "<=", float(t), f"{nm}<={t:g}"))
    return specs


def literal_vec(P, spec):
    k, op, t, _ = spec
    v = P.reshape(-1, P.shape[-1])[:, k]
    return (v >= t) if op == ">=" else (v < t) if op == "<" else (v <= t)


def literal_matrix(P, specs):
    return np.stack([literal_vec(P, s) for s in specs]).astype(np.float32)


# ------------------------------------------------------------------ search
def _corr_from_sums(sy, cnt, n, sd):
    p = cnt / n
    den = n * sd * np.sqrt(np.clip(p * (1 - p), 1e-12, None))
    c = sy / den
    c[(cnt < 20) | (cnt > n - 20)] = 0.0          # too rare or too common to define anything
    return c


def search(y, L, specs, beam=48, depth=3, keep=10):
    """y: unit activity [n*S]; L: literal matrix [M, n*S] on the SAME (training) rows.
    Returns up to `keep` programs (tuples of literal indices) with training correlation, best first."""
    y = y.astype(np.float32)
    yc = y - y.mean()
    sd = yc.std() + 1e-9
    n = len(y)
    s1 = L @ yc
    c1 = L.sum(1)
    cor1 = _corr_from_sums(s1, c1, n, sd)
    cand = {(i,): cor1[i] for i in range(len(specs))}
    frontier = [(i,) for i in np.argsort(-np.abs(cor1))[:beam]]
    for d in range(2, depth + 1):
        new = {}
        for prog in frontier:
            b = np.prod(L[list(prog)], axis=0)
            sy = L @ (b * yc)
            cnt = L @ b
            cr = _corr_from_sums(sy, cnt, n, sd)
            for j in np.argsort(-np.abs(cr))[:beam]:
                if j in prog:
                    continue
                key = tuple(sorted(prog + (int(j),)))
                new[key] = cr[j]
        cand.update(new)
        frontier = [k for k, _ in sorted(new.items(), key=lambda kv: -abs(kv[1]))[:beam]]
    ranked = sorted(cand.items(), key=lambda kv: -abs(kv[1]))
    # prefer the shortest program within 0.01 of a longer one
    out = []
    for prog, c in ranked:
        if any(set(q).issubset(prog) and abs(cq) >= abs(c) - 0.01 for q, cq in out):
            continue
        out.append((prog, float(c)))
        if len(out) >= keep:
            break
    return out


def search_pos(yt, L3, specs, beam=48, depth=3, keep=10, min_count=5):
    """Position-level search: the program's count per position (sum over sites) against the unit's total per
    position (yt [n]). L3: literal matrix reshaped [M, n, S]."""
    yc = (yt - yt.mean()).astype(np.float32)
    sd = yc.std() + 1e-9

    def corr(Cn):                       # Cn: [M, n] counts per position
        Cc = Cn - Cn.mean(1, keepdims=True)
        den = np.sqrt((Cc ** 2).sum(1)) * sd * np.sqrt(len(yc))
        c = (Cc @ yc) / np.maximum(den, 1e-9)
        c[(Cn > 0).sum(1) < min_count] = 0.0
        return c
    c1 = corr(L3.sum(2))
    cand = {(i,): c1[i] for i in range(len(specs))}
    frontier = [(i,) for i in np.argsort(-np.abs(c1))[:beam]]
    for d in range(2, depth + 1):
        new = {}
        for prog in frontier:
            b = np.prod(L3[list(prog)], axis=0)            # [n, S]
            cr = corr(np.einsum("mns,ns->mn", L3, b))
            for j in np.argsort(-np.abs(cr))[:beam]:
                if j in prog:
                    continue
                new[tuple(sorted(prog + (int(j),)))] = cr[j]
        cand.update(new)
        frontier = [k for k, _ in sorted(new.items(), key=lambda kv: -abs(kv[1]))[:beam]]
    ranked = sorted(cand.items(), key=lambda kv: -abs(kv[1]))
    out = []
    for prog, c in ranked:
        if any(set(q).issubset(prog) and abs(cq) >= abs(c) - 0.01 for q, cq in out):
            continue
        out.append((prog, float(c)))
        if len(out) >= keep:
            break
    return out


def program_map(P, specs, prog):
    b = np.ones(P.shape[0] * P.shape[1], dtype=np.float32)
    for i in prog:
        b *= literal_vec(P, specs[i])
    return b


def fidelity(y, m):
    if m.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(y.astype(np.float64), m.astype(np.float64))[0, 1])


def describe(specs, prog):
    return " and ".join(specs[i][3] for i in prog)


# ------------------------------------------------------------------ expressions (cheap-model proposals)
ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.Name, ast.Load, ast.Constant,
           ast.Add, ast.Sub, ast.Mult, ast.Div, ast.BitAnd, ast.BitOr, ast.Invert, ast.Not, ast.USub,
           ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.And, ast.Or)


def expr_map(P, names, expr, budget=6):
    """Evaluate a restricted expression over primitive names (no calls, no attributes, at most `budget` names)."""
    tree = ast.parse(expr, mode="eval")
    used = 0
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED):
            raise ValueError(f"not allowed: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(f"unknown primitive {node.id}")
            used += 1
    if used > budget:
        raise ValueError("over the term budget")
    env = {nm: P.reshape(-1, P.shape[-1])[:, k].astype(np.float32) for k, nm in enumerate(names)}

    class T(ast.NodeTransformer):           # boolean logic -> arithmetic on 0/1 maps
        def visit_BoolOp(self, node):
            self.generic_visit(node)
            v = node.values[0]
            for w in node.values[1:]:
                v = (ast.BinOp(left=v, op=ast.Mult(), right=w) if isinstance(node.op, ast.And)
                     else ast.Call(func=ast.Name("_or", ast.Load()), args=[v, w], keywords=[]))
            return v

        def visit_BinOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, ast.BitAnd):
                return ast.BinOp(left=node.left, op=ast.Mult(), right=node.right)
            if isinstance(node.op, ast.BitOr):
                return ast.Call(func=ast.Name("_or", ast.Load()), args=[node.left, node.right], keywords=[])
            return node

        def visit_UnaryOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, (ast.Not, ast.Invert)):
                return ast.BinOp(left=ast.Constant(1.0), op=ast.Sub(), right=node.operand)
            return node

        def visit_Compare(self, node):
            self.generic_visit(node)
            return ast.Call(func=ast.Name("_f", ast.Load()), args=[node], keywords=[])
    t2 = ast.fix_missing_locations(T().visit(tree))
    env["_f"] = lambda a: np.asarray(a, dtype=np.float32)
    env["_or"] = lambda a, b: np.maximum(np.asarray(a, np.float32), np.asarray(b, np.float32))
    code = compile(t2, "<rosetta-expr>", "eval")
    with np.errstate(all="ignore"):
        try:
            r = eval(code, {"__builtins__": {}}, env)
        except (TypeError, ValueError) as e:
            raise ValueError(f"evaluation error: {e}")
    r = np.asarray(r, dtype=np.float32)
    if r.ndim == 0:
        r = np.full(P.shape[0] * P.shape[1], float(r), dtype=np.float32)
    return np.nan_to_num(r)
