"""D1 stage 1b: recover a finite abelian group and each symbol's element in it, from circle planes alone.

Each circle plane i gives every symbol t a unit complex number z_i(t). If the symbols are elements g(t) of a finite
abelian group G and plane i shows a character chi_i of G, then z_i(t) = c_i * chi_i(g(t)) plus noise. Characters
multiply, so integer vectors v with prod_i z_i(t)^v_i constant over t ("relations") are exactly the relations among
the characters. The relation lattice L gives the group the planes see: Z^P / L, decomposed by the Smith normal form
into Z_d1 x Z_d2 x ... Each symbol is then decoded to the element of that group that best explains all planes at once
(a single plane is too noisy to place a symbol; all of them together are not).

Nothing about the task is assumed: not the group, not its size, not the order of the symbols.
"""
import itertools
import numpy as np
from sympy import Matrix
from sympy.matrices.normalforms import smith_normal_decomp


def plane_angles(X, planes, center=None):
    mu = X.mean(0) if center is None else center
    TH, RAD = [], []
    for pl in planes:
        Y = (X - mu) @ pl["Q"]
        TH.append(np.arctan2(Y[:, 1], Y[:, 0])); RAD.append(np.hypot(Y[:, 0], Y[:, 1]))
    return np.array(TH), np.array(RAD)


def find_relations(TH, on, B=None, thresh=0.6, max_combos=300_000):
    """Integer vectors v (|v_i| <= B) with |mean_t exp(i v.theta(t))| > thresh, over symbols `on`."""
    P = TH.shape[0]
    if B is None:
        B = 1
        while (2 * (B + 1) + 1) ** P <= max_combos and B < 60:
            B += 1
    rng = np.arange(-B, B + 1)
    T = TH[:, on]
    rels = []
    combos = itertools.product(rng, repeat=P)
    while True:
        chunk = np.array(list(itertools.islice(combos, 20000)))
        if len(chunk) == 0:
            break
        c = np.abs(np.exp(1j * (chunk @ T)).mean(1))
        for v, cv in zip(chunk[c > thresh], c[c > thresh]):
            if v.any():
                rels.append((float(cv), tuple(int(x) for x in v)))
    return rels, B


def group_from_relations(rels, P):
    """Smith normal form of the relation lattice. Returns invariant factors d (>1 only), and the matrix A (P x r) with
    plane i = character a_i in Z_d1 x ... (A[i, k] mod d[k])."""
    if not rels:
        return None
    R = Matrix([list(v) for _, v in rels])
    D, U, V = smith_normal_decomp(R)       # D = U R V, so the lattice L (rows of R) satisfies L V = rows of D
    diag = [abs(int(D[i, i])) if i < D.rows else 0 for i in range(P)]
    keep = [k for k in range(P) if diag[k] != 1]
    if any(diag[k] == 0 for k in keep):
        return {"d": [diag[k] for k in keep], "A": None, "finite": False}
    A = np.array([[int(V[i, k]) % diag[k] for k in keep] for i in range(P)], dtype=np.int64)   # e_i -> row i of V
    return {"d": [diag[k] for k in keep], "A": A, "finite": True}


def decode(TH, A, d, iters=8, weights=None, ref=None):
    """Element g(t) in Z_d1 x ... for every symbol, maximising sum_i w_i cos(theta_i(t) - phi_i - 2pi <a_i, g>/d)."""
    P, n = TH.shape
    d = np.array(d)
    els = np.array(list(itertools.product(*[range(x) for x in d]))) if len(d) else np.zeros((1, 0), int)
    phase_el = 2 * np.pi * ((els[None, :, :] * A[:, None, :]) / d[None, None, :]).sum(-1)   # P x |G|
    w = np.ones(P) if weights is None else np.asarray(weights, float)
    ref = 0 if ref is None else ref
    phi = TH[:, ref].copy()                       # symbol `ref` = identity to start
    for _ in range(iters):
        score = np.zeros((n, len(els)))
        for i in range(P):
            score += w[i] * np.cos(TH[i][:, None] - phi[i] - phase_el[i][None, :])
        g = score.argmax(1)
        for i in range(P):
            phi[i] = np.angle(np.exp(1j * (TH[i] - phase_el[i][g])).mean())
    srt = np.sort(score, 1)
    fit = srt[:, -1] / w.sum()                      # 1 = every plane agrees perfectly
    margin = (srt[:, -1] - srt[:, -2]) / w.sum() if srt.shape[1] > 1 else np.zeros(n)
    return {"g": els[g], "g_index": g, "fit": fit, "margin": margin, "phi": phi, "elements": els}


def _in_lattice(v, basis_rows):
    if not basis_rows:
        return False
    M = Matrix(basis_rows)
    D, U, V = smith_normal_decomp(M)
    y = Matrix([list(v)]) * V
    for k in range(min(D.rows, D.cols)):
        dk = int(D[k, k])
        if dk == 0:
            if y[0, k] != 0:
                return False
        elif int(y[0, k]) % dk:
            return False
    return all(y[0, k] == 0 for k in range(min(D.rows, D.cols), D.cols))


def recover(X, planes, center=None, thresh=0.6, min_fit=0.85):
    """Full stage: relations -> group -> per-symbol elements. X = rows are symbols.

    Relations are taken in order of coherence and kept only if the decode still explains the symbols (mean fit >= min_fit
    once the group is finite): a relation that is not really there makes the decode fall apart, a real one does not."""
    TH, RAD = plane_angles(X, planes, center)
    on = (RAD > 0.5 * np.median(RAD, axis=1, keepdims=True)).all(0)
    rels, B = find_relations(TH, on, thresh=thresh)
    rels.sort(reverse=True)
    w = [pl.get("energy", 1.0) for pl in planes]
    kept, G, dec = [], None, None
    for c, v in rels:
        if _in_lattice(v, [list(x) for _, x in kept]):
            continue
        G2 = group_from_relations(kept + [(c, v)], len(planes))
        if G2["finite"]:
            d2 = decode(TH, G2["A"], G2["d"], weights=w)
            if float(np.mean(d2["fit"][on])) < min_fit:
                continue
            G, dec = G2, d2
        kept.append((c, v))
    out = {"n_relations": len(rels), "kept_relations": kept, "B": B, "on": on}
    if G is None:
        out.update({"group": None, "finite": False})
        return out
    out.update({"group": G["d"], "finite": True, "A": G["A"], **dec,
                "order": int(np.prod(G["d"])) if G["d"] else 1,
                "distinct": int(len({tuple(x) for x in dec["g"][on]}))})
    return out


def recover_auto(X, planes, center=None, min_fit=0.85, max_planes=6, seed_from=5):
    """Choose planes: seed with the best triple among the strongest planes (three planes have short relations; two
    often do not), then add the remaining planes by energy, keeping a plane only if the recovered group still explains
    every symbol (mean fit >= min_fit) and places at least as many symbols. Deflation leaves partial copies and junk
    planes behind; this drops them."""
    if len(planes) < 3:
        return {"finite": False, "group": None, "planes_used": [], "reason": f"only {len(planes)} circle planes"}
    order = sorted(range(len(planes)), key=lambda i: -planes[i]["energy"])
    def score(r):
        return (r["distinct"], float(np.mean(r["fit"][r["on"]])))
    best, used = None, []
    for trial in itertools.combinations(order[:seed_from], min(3, len(order))):
        r = recover(X, [planes[j] for j in trial], center, min_fit=min_fit)
        if r["finite"] and float(np.mean(r["fit"][r["on"]])) >= min_fit and (best is None or score(r) > score(best)):
            best, used = r, list(trial)
    if best is None:
        return {"finite": False, "group": None}
    for i in order:
        if i in used or len(used) >= max_planes:
            continue
        r = recover(X, [planes[j] for j in used + [i]], center, min_fit=min_fit)
        if r["finite"] and float(np.mean(r["fit"][r["on"]])) >= min_fit and r["distinct"] >= best["distinct"]:
            best, used = r, used + [i]
    best["planes_used"] = used
    return best
