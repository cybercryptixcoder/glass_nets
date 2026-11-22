"""D1 factor mode: does the network secretly split its computation into independent sub-computations? Blind.

Idea. If each input is represented as a sum of independent parts, x(a) = u_1(a_1) + u_2(a_2) + ..., where a_i is a
coarse label of a (a partition of the symbols), then differences between symbols that agree on every part but one are
pure differences of that one part, and the same difference vector recurs once for every value of the other parts
("parallelograms"). Repeated difference vectors therefore expose the parts' subspaces; projecting the symbols onto a
part's subspace collapses them into that part's few blocks; and the blocks of different parts form a full product grid
(every combination occurs, once). Nothing about the task is assumed: not the number of parts, not their sizes.

Then the network's own logits are tested for the same split: logits(a, b) = sum_i L_i[a_i, b_i] (one table per part,
indexed by the pair of blocks), fitted by alternating averages. If that sum reproduces the network's answer on every
input, the network is (functionally) that many independent small computations, and each table L_i is one of them.

  split_side(X)                one binary split of a representation into two parts (recursion in factor_side)
  factor_side(X)               all parts: list of partitions of the symbols
  additive_tables(L, pa, pb)   the per-part logit tables, how much of the logits they explain, argmax agreement
  part_table(tab, ka, kb)      a part's composition table (pair of blocks -> output value) by clustering its pair vectors
  name_table(T)                what the table is: an absorbing element plus a Latin square, and the group it is isotopic to
"""
import itertools
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import cKDTree
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score


def _grid(PA, PB, n):
    cells = {}
    for a, b in zip(PA, PB):
        cells[(a, b)] = cells.get((a, b), 0) + 1
    single = sum(1 for v in cells.values() if v == 1)
    ka, kb = PA.max() + 1, PB.max() + 1
    return single / max(ka * kb, n)


def split_candidates(Z, kmax_blocks=60, tols=(0.1, 0.15, 0.22, 0.3), seed=0, top=6, min_grid=0.6):
    """Several two-part splits (different tolerances, subspace dimensions and block counts), best grid score first,
    one per pair of block counts. The arbiter between them is the logit test, not the grid."""
    n, k = Z.shape
    if k < 2 or n < 6:
        return []
    A, B = np.triu_indices(n, 1)
    D = Z[A] - Z[B]
    scale = np.median(np.linalg.norm(D, axis=1)) + 1e-12
    tree = cKDTree(np.r_[D, -D])
    cands = {}
    for tol in tols:
        cnt = np.array([len(x) for x in tree.query_ball_point(D, tol * scale)])
        for r in _split_at(Z, D, cnt >= max(4, 3 * np.median(cnt)), kmax_blocks, seed, collect=True) or []:
            key = tuple(sorted((r[1].max() + 1, r[2].max() + 1)))
            if r[0] >= min_grid and (key not in cands or r[0] > cands[key][0]):
                cands[key] = r
    return sorted(cands.values(), key=lambda r: -r[0])[:top]


def split_side(Z, kmax_blocks=60, tols=(0.1, 0.15, 0.22, 0.3), seed=0):
    """Z: symbols x k (centred). Returns (score, PA, PB, QA, QB) for the best two-part split over several
    tolerances for "the same difference vector", or None. A difference counts as repeated when it recurs at least
    3 times as often as a typical difference does at that tolerance."""
    n, k = Z.shape
    if k < 2 or n < 6:
        return None
    A, B = np.triu_indices(n, 1)
    D = Z[A] - Z[B]
    scale = np.median(np.linalg.norm(D, axis=1)) + 1e-12
    tree = cKDTree(np.r_[D, -D])
    best = None
    for tol in tols:
        cnt = np.array([len(x) for x in tree.query_ball_point(D, tol * scale)])
        r = _split_at(Z, D, cnt >= max(4, 3 * np.median(cnt)), kmax_blocks, seed)
        if r is not None and (best is None or r[0] > best[0]):
            best = r
    return best


def _split_at(Z, D, sel, kmax_blocks, seed, collect=False):
    n, k = Z.shape
    Dp = D[sel]
    if len(Dp) < 10:
        return None
    Dn = Dp / np.linalg.norm(Dp, axis=1, keepdims=True)
    lab = SpectralClustering(2, affinity="precomputed", random_state=seed).fit_predict(np.abs(Dn @ Dn.T) ** 2)
    best = None; allc = {}
    for c0 in range(2):
        if (lab == c0).sum() < 3:
            continue
        v = np.linalg.svd(Dp[lab == c0], full_matrices=False)[2]
        for d0 in range(1, k):
            QA = v[:d0]
            QB = np.linalg.svd(Z - Z @ QA.T @ QA, full_matrices=False)[2][:k - d0]
            LA = linkage(Z @ QA.T, "average"); LB = linkage(Z @ QB.T, "average")
            for ka in range(2, min(kmax_blocks, n) + 1):
                PA = fcluster(LA, ka, "maxclust") - 1
                if PA.max() + 1 < ka:
                    continue
                for kb in range(2, min(kmax_blocks, n) + 1):
                    if ka * kb < 0.8 * n:
                        continue
                    PB = fcluster(LB, kb, "maxclust") - 1
                    s = _grid(PA, PB, n)
                    if best is None or s > best[0]:
                        best = (s, PA, PB, QA, QB)
                    if collect and ((ka, kb) not in allc or s > allc[(ka, kb)][0]):
                        allc[(ka, kb)] = (s, PA, PB, QA, QB)
    if collect:
        return sorted(allc.values(), key=lambda r: -r[0])[:12]
    return best


def _coords(X, whiten):
    """Centred coordinates. Rank: the largest drop between consecutive singular values (above 1e-3 of the top one),
    so that weak but structured directions are kept and the noise floor is not. whiten=True equalises the kept
    directions (a part carried by small directions then counts as much as one carried by large ones)."""
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    m = int((S / S[0] > 1e-3).sum())
    if m <= 2:
        k = m
    else:
        r = S[:m - 1] / np.maximum(S[1:m], 1e-12)
        k = int(np.argmax(r) + 1)
        k = max(k, int((S / S[0] > 0.05).sum()))
    return U[:, :k] * (np.sqrt(len(X)) if whiten else S[:k])


def factor_side(X, min_score=0.8, depth=0, max_depth=3, log=None, whiten=None):
    """Recursive binary splits. Returns a list of partitions (one per part) of the symbols. Both unwhitened and
    whitened coordinates are tried at the top level; the better product grid wins."""
    best = None
    for wh in ([False, True] if whiten is None else [whiten]):
        Z = _coords(X, wh)
        sp = split_side(Z)
        if sp is not None and (best is None or sp[0] > best[0][0]):
            best = (sp, Z, wh)
    if best is None:
        sp, Z, wh = None, None, False
    else:
        sp, Z, wh = best
    if sp is None or sp[0] < min_score:
        if log:
            log(f"{'  ' * depth}no further split (grid score {0 if sp is None else sp[0]:.2f})")
        return [np.zeros(len(X), int)] if depth else []
    s, PA, PB, QA, QB = sp
    if log:
        log(f"{'  ' * depth}split into parts of {PA.max() + 1} and {PB.max() + 1} blocks (grid score {s:.3f}, dims {QA.shape[0]}+{QB.shape[0]})")
    parts = []
    for P, Q in ((PA, QA), (PB, QB)):
        if depth + 1 < max_depth and P.max() + 1 >= 6:
            C = np.array([(Z @ Q.T)[P == b].mean(0) for b in range(P.max() + 1)])   # the part's block centres
            sub = factor_side(C, max(min_score, 0.85), depth + 1, max_depth, log, whiten=wh)   # stricter for block centres; the logit test is the arbiter
            if len(sub) >= 2:
                parts += [s_[P] for s_ in sub]
                continue
        parts.append(P)
    return parts


def additive_tables(L, pa, pb, iters=40):
    """L: logits (n_a x n_b x n_out), centred over classes. pa, pb: lists of partitions (same order, paired parts).
    Fit L ~ sum_i T_i[pair index of part i]. Returns tables, explained share, argmax agreement."""
    na, nb = L.shape[:2]
    A, B = np.meshgrid(np.arange(na), np.arange(nb), indexing="ij")
    idx = [P[A] * (Q.max() + 1) + Q[B] for P, Q in zip(pa, pb)]
    sizes = [(P.max() + 1) * (Q.max() + 1) for P, Q in zip(pa, pb)]
    T = [np.zeros((s, L.shape[2])) for s in sizes]
    for _ in range(iters):
        for i in range(len(T)):
            R = L - sum(T[j][idx[j]] for j in range(len(T)) if j != i)
            T[i] = np.array([R[idx[i] == u].mean(0) if (idx[i] == u).any() else np.zeros(L.shape[2]) for u in range(sizes[i])])
    Lh = sum(T[i][idx[i]] for i in range(len(T)))
    return T, float(1 - ((L - Lh) ** 2).sum() / (L ** 2).sum()), float((Lh.argmax(-1) == L.argmax(-1)).mean()), Lh


def part_table(tab, ka, kb, kmax=60):
    """A part's composition table: pairs whose logit contributions are near-identical produce the same output value.
    Number of values chosen by silhouette on correlation distance. Returns (table ka x kb, values per class, K)."""
    C = np.corrcoef(tab); Dm = np.clip(1 - C, 0, None); iu = np.triu_indices(len(tab), 1)
    Lk = linkage(Dm[iu], "average"); best = None
    for K in range(2, min(len(tab) - 1, kmax)):
        lb = fcluster(Lk, K, "maxclust") - 1
        if lb.max() + 1 < 2:
            continue
        s = silhouette_score(Dm, lb, metric="precomputed")
        if best is None or s > best[0]:
            best = (s, lb)
    lab = best[1]; K = lab.max() + 1
    M = np.array([tab[lab == v].mean(0) for v in range(K)])
    return lab.reshape(ka, kb), M.argmax(0), K, float(best[0])


def _loop_from_latin(Lsq):
    """Principal isotope of a Latin square: x*y = L[R^-1(x), C^-1(y)] with R(i) = L[i, 0], C(j) = L[0, j]."""
    m = len(Lsq)
    R = {Lsq[i][0]: i for i in range(m)}; Cc = {Lsq[0][j]: j for j in range(m)}
    syms = sorted(set(Lsq[0]))
    return {(x, y): Lsq[R[x]][Cc[y]] for x in syms for y in syms}, syms


def name_table(T):
    """Describe a composition table: absorbing rows/columns (an element that sends everything to one value, like 0 in a
    multiplication), and whether the rest is a Latin square isotopic to a group, which group (order statistics)."""
    T = np.asarray(T)
    rows_const = [i for i in range(T.shape[0]) if len(set(T[i])) == 1]
    cols_const = [j for j in range(T.shape[1]) if len(set(T[:, j])) == 1]
    R = [i for i in range(T.shape[0]) if i not in rows_const]
    Cn = [j for j in range(T.shape[1]) if j not in cols_const]
    sub = T[np.ix_(R, Cn)]
    out = {"absorbing_rows": rows_const, "absorbing_cols": cols_const, "core_shape": list(sub.shape)}
    if sub.size == 0:
        return out
    latin = sub.shape[0] == sub.shape[1] and all(len(set(r)) == sub.shape[1] for r in sub) and all(len(set(c)) == sub.shape[0] for c in sub.T) \
        and len(set(sub.ravel())) == sub.shape[0]
    out["latin_square"] = bool(latin)
    if not latin:
        return out
    op, syms = _loop_from_latin(sub.tolist())
    assoc = all(op[(op[(x, y)], z)] == op[(x, op[(y, z)])] for x in syms for y in syms for z in syms)
    comm = all(op[(x, y)] == op[(y, x)] for x in syms for y in syms)
    e = next(x for x in syms if all(op[(x, y)] == y for y in syms))
    orders = {}
    for x in syms:
        k, y = 1, x
        while y != e:
            y = op[(y, x)]; k += 1
        orders[k] = orders.get(k, 0) + 1
    m = len(syms)
    cyclic = m in orders
    out.update({"isotopic_to_group": bool(assoc), "commutative": bool(comm), "element_orders": dict(sorted(orders.items())),
                "group": (f"cyclic group Z_{m}" if cyclic else f"non-cyclic group of order {m}") if assoc else None})
    if assoc and cyclic and len(rows_const) == 1 and len(cols_const) == 1:
        q = m + 1
        prime = q > 1 and all(q % d for d in range(2, int(q ** 0.5) + 1))
        out["reads_as"] = (f"the multiplication table of the field with {q} elements (an absorbing zero plus the cyclic group of its {m} non-zero elements)"
                           if prime else f"an absorbing zero plus a cyclic group of order {m}")
    return out


def part_tables_from_outputs(net, pa, pb):
    """Each part's composition table read from the network's own answers, given the parts found blind: two block pairs
    of part i produce the same output value of that part iff, with the other parts held fixed, they always give the same
    answer. Returns per part: (table, number of values, share of pairs whose signature is consistent)."""
    na, nb = net.shape
    reps = {}
    for a in range(na):
        for b in range(nb):
            key = tuple(int(P[a]) for P in pa) + tuple(int(Q[b]) for Q in pb)
            reps.setdefault(key, (a, b))
    K = len(pa); out = []
    for i in range(K):
        ka, kb = pa[i].max() + 1, pb[i].max() + 1
        others = [j for j in range(K) if j != i]
        ctx = sorted({tuple((k[j], k[K + j]) for j in others) for k in reps})
        sigs = {}
        for u in range(ka):
            for v in range(kb):
                s = []
                for c in ctx:
                    key = [None] * (2 * K); key[i] = u; key[K + i] = v
                    for j, (x, y) in zip(others, c):
                        key[j] = x; key[K + j] = y
                    ab = reps.get(tuple(key))
                    s.append(-1 if ab is None else int(net[ab]))
                sigs[(u, v)] = tuple(s)
        # merge pairs whose signatures agree wherever both are defined (a missing combination gives -1)
        keys = list(sigs); S = np.array([sigs[k] for k in keys])
        lab = -np.ones(len(keys), int); reps_s = []
        order = np.argsort((S < 0).sum(1))                  # most complete signatures first
        for idx in order:
            sv = S[idx]
            for g, rv in enumerate(reps_s):
                both = (sv >= 0) & (rv >= 0)
                if both.sum() and np.all(sv[both] == rv[both]):
                    lab[idx] = g; reps_s[g] = np.where(rv >= 0, rv, sv); break
            else:
                lab[idx] = len(reps_s); reps_s.append(sv.copy())
        T = np.zeros((ka, kb), int)
        for k_, l_ in zip(keys, lab):
            T[k_] = l_
        out.append((T, len(reps_s)))
    return out


def ring_fingerprint(T):
    """Row/column profiles (distinct values per row) and the invertible core, compared with multiplication and addition
    tables of Z_m (m = number of values) and with products of smaller cyclic rings."""
    T = np.asarray(T); m = len(set(T.ravel()))
    prof = lambda M: sorted(len(set(r)) for r in M)
    fp = (prof(T), prof(T.T))
    found = []
    def mul(q): return np.array([[(i * j) % q for j in range(q)] for i in range(q)])
    def add(q): return np.array([[(i + j) % q for j in range(q)] for i in range(q)])
    if fp == (prof(mul(m)), prof(mul(m).T)):
        found.append(f"multiplication in Z_{m}")
    if fp == (prof(add(m)), prof(add(m).T)):
        found.append(f"addition in Z_{m} (a Latin square)")
    units_rows = [i for i in range(T.shape[0]) if len(set(T[i])) == T.shape[1]]
    units_cols = [j for j in range(T.shape[1]) if len(set(T[:, j])) == T.shape[0]]
    core = T[np.ix_(units_rows, units_cols)]
    core_info = None
    if len(units_rows) and len(set(core.ravel())) == len(units_rows) == len(units_cols):
        core_info = name_table(core)
    return {"values": m, "row_profile": fp[0], "col_profile": fp[1], "matches": found,
            "invertible_core": {"size": len(units_rows), **(core_info or {})}}


def transfer_partition(L, pb, side="a", kmax=60):
    """One side's parts are known (pb, partitions of the b symbols); find the other side's parts from the network's
    logits: fit L(a, b) ~ sum_i T_i[a, b_i] with a free, then symbols a whose rows T_i[a, :] agree share part i's
    block. (side='b' does the mirror image: pass L transposed.) Returns a list of partitions of the a symbols."""
    na, nb, C = L.shape
    sizes = [P.max() + 1 for P in pb]
    T = [np.zeros((na, s, C)) for s in sizes]
    for _ in range(25):
        for i in range(len(T)):
            R = L - sum(T[j][:, pb[j], :] for j in range(len(T)) if j != i)
            T[i] = np.stack([R[:, pb[i] == u, :].mean(1) for u in range(sizes[i])], 1)
    # cluster a's rows per part; choose the block counts jointly so that a's blocks form a product grid
    links = []
    for i in range(len(T)):
        V = T[i].reshape(na, -1)
        Cm = np.corrcoef(V); Dm = np.clip(1 - Cm, 0, None); iu = np.triu_indices(na, 1)
        links.append(linkage(Dm[iu], "average"))
    best = None
    ranges = [range(2, min(na, kmax) + 1) for _ in links]
    if len(links) == 2:
        for k1 in ranges[0]:
            P1 = fcluster(links[0], k1, "maxclust") - 1
            for k2 in ranges[1]:
                if k1 * k2 < 0.8 * na:
                    continue
                P2 = fcluster(links[1], k2, "maxclust") - 1
                sc = _grid(P1, P2, na)
                if best is None or sc > best[0]:
                    best = (sc, [P1, P2])
        return best[1]
    return [fcluster(Lk, max(2, int(round(na ** (1 / len(links))))), "maxclust") - 1 for Lk in links]


def merge_equivalent_blocks(T, P, axis=0):
    """Blocks whose rows (axis=0) or columns (axis=1) of a part's composition table are identical are treated
    identically by the network: merge them. Returns the new partition and table."""
    T = np.asarray(T); M = T if axis == 0 else T.T
    keys = {}
    new_of_old = np.zeros(M.shape[0], int)
    for u in range(M.shape[0]):
        new_of_old[u] = keys.setdefault(tuple(M[u].tolist()), len(keys))
    rows = [next(u for u in range(M.shape[0]) if new_of_old[u] == g) for g in range(len(keys))]
    M2 = M[rows]
    return new_of_old[P], (M2 if axis == 0 else M2.T)
