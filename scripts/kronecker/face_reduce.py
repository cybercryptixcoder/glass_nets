"""P2: known math applied recursively. B1 (Dvir bounds + conjugate forms + length bounds), and on every tight
Dvir face the exact reduction g(A,B,t) = sum c^A_{a,r} c^B_{a,s} g(r, s, tbar), a = A ∩ B (Dvir 1993;
checked here against exact tables for n <= 10, scripts/kronecker/face_check.py), with P2 applied to the pieces.
Predicts 'zero' only when a necessary condition fails or every piece of a tight face is predicted zero."""
import sys
from functools import lru_cache
from fractions import Fraction
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from skew import chi_skew, partitions, z, meet, conj

def inter(a, b): return sum(min(p, q) for p, q in zip(a, b))

@lru_cache(maxsize=None)
def lr_support(lam, alpha):
    """Partitions r with c^lam_{alpha, r} > 0."""
    k = sum(lam) - sum(alpha)
    if k == 0: return ((),) if lam == alpha else ()
    cl = list(partitions(k))
    vals = {c: Fraction(chi_skew(lam, alpha, c), z(c)) for c in cl}
    out = []
    for r in cl:
        s = sum(vals[c] * chi_skew(r, (), c) for c in cl)
        if s != 0: out.append(r)
    return tuple(out)

def faces(A, B, t):
    """Tight faces with target t: list of (A*, B*, t*) such that g(A,B,t) = g(A*,B*,t*) and t*_1 = |A* ∩ B*|."""
    Ac, Bc, tc = conj(A), conj(B), conj(t)
    out = []
    for (x, y, w) in ((A, B, t), (Ac, Bc, t), (A, Bc, tc), (Ac, B, tc)):
        if w and w[0] == inter(x, y): out.append((x, y, w))
    return out

def b1(A, B, t):
    Ac, Bc, tc = conj(A), conj(B), conj(t)
    for (x, y, w) in ((A, B, t), (B, t, A), (t, A, B)):
        xc, yc, wc = conj(x), conj(y), conj(w)
        if w[0] > min(inter(x, y), inter(xc, yc)): return False
        if len(w) > min(inter(x, yc), inter(xc, y)): return False
        if len(w) > min(len(x) * len(y), x[0] * y[0]): return False
        if w[0] > min(x[0] * len(y), len(x) * y[0]): return False
    return True

def canon(A, B, t):
    forms = []
    for (x, y, w) in ((A, B, t), (A, t, B), (B, A, t), (B, t, A), (t, A, B), (t, B, A)):
        for (u, v, s) in ((x, y, w), (conj(x), conj(y), w), (conj(x), y, conj(w)), (x, conj(y), conj(w))):
            forms.append((u, v, s))
    return min(forms)

@lru_cache(maxsize=None)
def p2(A, B, t, depth=0):
    if sum(A) <= 1: return True
    if not b1(A, B, t): return False
    if depth >= 6: return True
    for (x, y, w) in ((A, B, t), (B, t, A), (t, A, B)):            # each slot as the target
        for (xs, ys, ws) in faces(x, y, w):
            a = meet(xs, ys); tb = ws[1:]
            if not tb:                                               # nothing left: g = [xs == ys]
                if xs != ys: return False
                continue
            ok = False
            for r in lr_support(xs, a):
                for s in lr_support(ys, a):
                    if p2(*canon(r, s, tb), depth + 1):
                        ok = True; break
                if ok: break
            if not ok: return False
    return True
