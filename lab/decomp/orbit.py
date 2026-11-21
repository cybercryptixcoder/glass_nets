"""D1 stage 1, non-abelian: the symbols as an orbit of an orthogonal group action.

If a network embeds group element g as rho(g) u for an orthogonal representation rho (any group, abelian or not),
then after whitening the symbols' points lie on a sphere, and the group acts on that point set by rotations: an
orthogonal map R with R y_g = y_(h g) for every g. Finding all orthogonal maps that permute the point set gives a
permutation group on the symbols; a subgroup of the right size acting regularly (one element per symbol) is the
group itself, and its permutations give the Cayley table. Nothing about the group is assumed.

  find_sphere(X)            smallest whitened rank at which the symbols sit on a sphere
  orbit_symmetries(Y, tol)  all orthogonal maps permuting the points (as permutations)
  regular_subgroups(P, n)   subgroups of order n acting simply transitively
  describe(H)               order statistics (which group it is, up to isomorphism, for small groups)
"""
import itertools
import numpy as np


def find_sphere(X, kmax=8, cv_max=0.03):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    for k in range(2, min(kmax, len(S)) + 1):
        if S[k - 1] / S[0] < 0.05:
            break
        Y = U[:, :k]
        r = np.linalg.norm(Y, axis=1)
        cv = float(r.std() / r.mean())
        if cv < cv_max:
            return k, Y / r[:, None], Vt[:k], cv
    return None


def orbit_symmetries(Y, tol=0.1, max_found=200):
    n, k = Y.shape
    Gm = Y @ Y.T
    refs = [0]
    while len(refs) < k:
        best = None
        for c in range(n):
            if c in refs:
                continue
            s = np.linalg.svd(Y[refs + [c]], compute_uv=False)[-1]
            if best is None or s > best[0]:
                best = (s, c)
        refs.append(best[1])
    perms = set()

    def rec(img):
        if len(perms) >= max_found:
            return
        d = len(img)
        if d == k:
            U, S, Vt = np.linalg.svd(Y[list(img)].T @ Y[refs]); R = U @ Vt
            M = (Y @ R.T) @ Y.T; p = M.argmax(1)
            if np.all(M.max(1) > 1 - tol) and len(set(p.tolist())) == n:
                perms.add(tuple(int(x) for x in p))
            return
        for c in range(n):
            if c in img:
                continue
            if abs(Gm[refs[d], refs[d]] - Gm[c, c]) < tol and all(abs(Gm[refs[d], refs[e]] - Gm[c, img[e]]) < tol for e in range(d)):
                rec(img + [c])
    rec([])
    return sorted(perms)


def compose(p, q):
    return tuple(p[i] for i in q)            # (p o q)(i) = p(q(i))


def closure(gens, n, cap):
    idp = tuple(range(n)); H = {idp}; frontier = [idp]
    while frontier:
        new = []
        for h in frontier:
            for g in gens:
                c = compose(g, h)
                if c not in H:
                    H.add(c); new.append(c)
        frontier = new
        if len(H) > cap:
            return None
    return H


def regular_subgroups(P, n):
    """Subgroups of the permutation group P of order n acting regularly (each symbol hit by exactly one element)."""
    Ps = set(P)
    if not all(compose(a, b) in Ps for a in P[:20] for b in P[:20]):
        return []
    out = set()
    for a, b in itertools.combinations_with_replacement(P, 2):
        H = closure([a, b], n, n)
        if H is not None and len(H) == n and len({h[0] for h in H}) == n:
            out.add(frozenset(H))
    return [sorted(H) for H in out]


def element_orders(H, n):
    idp = tuple(range(n)); cnt = {}
    for h in H:
        k, x = 1, h
        while x != idp:
            x = compose(h, x); k += 1
        cnt[k] = cnt.get(k, 0) + 1
    return dict(sorted(cnt.items()))


KNOWN = {  # element-order profiles that pin down a group of that order (small cases)
    (24, ((1, 1), (2, 9), (3, 8), (4, 6))): "S4 (the symmetric group on 4 letters; rotations and reflections of a tetrahedron)",
    (24, ((1, 1), (2, 3), (3, 8), (6, 12))): "A4 x Z2",
    (6, ((1, 1), (2, 3), (3, 2))): "S3",
    (8, ((1, 1), (2, 5), (4, 2))): "D4 (symmetries of a square)",
    (8, ((1, 1), (2, 1), (4, 6))): "Q8 (quaternions)",
    (12, ((1, 1), (2, 3), (3, 8))): "A4",
    (60, ((1, 1), (2, 15), (3, 20), (5, 24))): "A5",
}


def describe(H, n):
    eo = element_orders(H, n)
    return KNOWN.get((n, tuple(eo.items())), None), eo


def tables(H, n):
    """Cayley tables T[a][b] for every choice of identity symbol e: with sigma_a the element taking e to a, a*b = sigma_a(b).
    For the left-regular action this is the group law when e is the identity; the 2-4 candidates per regular subgroup
    differ only in which symbol is the identity and in left/right."""
    out = []
    for e in range(n):
        sig = {h[e]: h for h in H}
        out.append((e, [[sig[a][b] for b in range(n)] for a in range(n)]))
    return out
