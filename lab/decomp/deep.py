"""D1 deep mode: networks whose composition is not a group law (first case: a^b mod p).

Blind, like stage 1: no task, no meaning of the symbols, no candidate features. Internal activations are used; the
network's outputs are not used to fit anything.

  side_representations(W)        x1 = F(a) + G(b) + const at the MLP's input; how additive it is
  components(X)                   split one side into statistically independent parts (independent subspace analysis:
                                  whiten, ICA, group the independent components by dependence)
  blocks(X, Q)                    the symbols' points in one part collapse to a few blocks: a partition of the symbols
  block_group(...)                the blocks' own structure: stage-1 group recovery on the block centres, with symbols
                                  sitting at the centre treated as an absorbing zero
"""
import itertools, warnings
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import FastICA
from sklearn.metrics import mutual_info_score

from decomp import program as pg


def side_representations(W, n):
    I = pg.internals(W, pg.all_pairs(n))
    x1 = I["x1"][:, -1].reshape(n, n, -1)
    mu = x1.mean((0, 1)); F = x1.mean(1) - mu; G = x1.mean(0) - mu
    R = x1 - F[:, None] - G[None] - mu
    share = float(1 - (R ** 2).sum() / ((x1 - mu) ** 2).sum())
    return F, G, share, I


def _levels(f, rel_tol=0.04):
    """Quantise a 1-D function of the symbols into levels (gaps larger than rel_tol of its range start a new level)."""
    o = np.argsort(f); v = f[o]
    cut = np.r_[0, np.cumsum(np.diff(v) > rel_tol * (v[-1] - v[0] + 1e-12))]
    lab = np.empty(len(f), int); lab[o] = cut
    return lab


def components(X, seeds=8, rank_ratio=0.05, dep_thresh=0.25, log=None):
    """Independent subspaces of X (rows = symbols). Returns a list of (n x k) orthonormal-ish bases in X's space and
    the chosen ICA sources. Choice among ICA seeds: the one whose sources have the fewest distinct levels in total
    (parts that are functions of a coarse label are low-alphabet; mixtures are not)."""
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = int((S / S[0] > rank_ratio).sum())
    Z = U[:, :k] * np.sqrt(len(X))
    best = None
    for seed in range(seeds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ica = FastICA(n_components=k, whiten=False, random_state=seed, max_iter=3000, fun="cube")
            Ssrc = ica.fit_transform(Z)
        lev = sum(len(set(_levels(Ssrc[:, i]).tolist())) for i in range(k))
        if best is None or lev < best[0]:
            best = (lev, Ssrc, ica.mixing_)
    lev, Ssrc, A = best
    labs = [_levels(Ssrc[:, i]) for i in range(k)]
    H = np.array([mutual_info_score(l, l) for l in labs])
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            M[i, j] = mutual_info_score(labs[i], labs[j]) / max(1e-9, min(H[i], H[j]))
    # group sources: connected components of the dependence graph
    groups, seen = [], set()
    for i in range(k):
        if i in seen:
            continue
        stack, g = [i], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); g.append(u)
            stack += [v for v in range(k) if v not in seen and M[u, v] > dep_thresh]
        groups.append(sorted(g))
    if log:
        log(f"    ICA: {k} sources, total levels {lev}; dependence groups {groups}")
    return groups, Ssrc, M


def blocks(P, rel=0.15):
    """Cluster points P (n x m) into blocks; tolerance relative to the median distance from the centre."""
    sc = np.median(np.linalg.norm(P - P.mean(0), axis=1)) + 1e-12
    lab = fcluster(linkage(P, "average"), rel * sc, "distance") - 1
    cent = np.array([P[lab == b].mean(0) for b in range(lab.max() + 1)])
    return lab, cent


def product_structure(labels):
    """How the parts' partitions combine: are they independent, and is their product injective on the symbols?"""
    n = len(labels[0])
    joint = np.unique(np.stack(labels, 1), axis=0)
    sizes = [int(l.max() + 1) for l in labels]
    mi = []
    for i, j in itertools.combinations(range(len(labels)), 2):
        mi.append(mutual_info_score(labels[i], labels[j]) / max(1e-9, min(mutual_info_score(labels[i], labels[i]), mutual_info_score(labels[j], labels[j]))))
    return {"block_counts": sizes, "joint_distinct": int(len(joint)), "symbols": n, "pairwise_dependence": [round(x, 3) for x in mi]}
