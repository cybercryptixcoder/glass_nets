"""orbit ids under argument permutations and even conjugation (shared with b2_lgbm.py)."""
import numpy as np
from kron_feats import conj

def orbit_ids(parts):
    """Canonical id of each triple under permutations of the three arguments and conjugation of an even number."""
    P = len(parts); C = conj(parts)
    key = {tuple(p): i for i, p in enumerate(parts.tolist())}
    cj = np.array([key[tuple(c)] for c in C.tolist()])
    a, b, c = np.meshgrid(np.arange(P), np.arange(P), np.arange(P), indexing="ij")
    best = None
    for (x, y, z) in [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]:
        for (u, v, w) in [(x, y, z), (cj[x], cj[y], z), (cj[x], y, cj[z]), (x, cj[y], cj[z])]:
            idx = (u * P + v) * P + w
            best = idx if best is None else np.minimum(best, idx)
    return best

