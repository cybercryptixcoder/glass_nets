"""Shared helpers: load exact tables, per-partition invariants, and the known necessary conditions (B1)."""
import numpy as np, os

KDATA = os.environ.get("KDATA", "/home/claude/kdata")


def load(n):
    d = np.load(os.path.join(KDATA, f"kron_n{n}.npz"))
    return d["parts"].astype(np.int64), d["g"]


def conj(parts):
    """Conjugate partitions, same padding (P x n)."""
    P, n = parts.shape
    out = np.zeros_like(parts)
    for j in range(n):
        out[:, j] = (parts > j).sum(1)
    return out


def inter(A, B):
    """|a ∩ b| for every pair: sum_i min(a_i, b_i). Returns P x P."""
    return np.minimum(A[:, None, :], B[None, :, :]).sum(-1)


def b1_masks(parts):
    """Known necessary conditions for g(l, m, v) != 0 (Dvir 1993 in all orderings, with conjugates; the
    length bound l(v) <= l(l) l(m) in its conjugate forms). Returns a boolean P x P x P array: True where all
    conditions hold (B1 predicts nonzero)."""
    C = conj(parts)
    row1, ln = parts[:, 0], C[:, 0]
    I, Ic, Imc = inter(parts, parts), inter(C, C), inter(parts, C)   # |a∩b|, |a'∩b'|, |a∩b'|
    P = len(parts)
    ok = np.ones((P, P, P), dtype=bool)
    # target in third slot: v_1 <= |l∩m|, v_1 <= |l'∩m'|, l(v) <= |l∩m'|, l(v) <= |l'∩m|
    t1 = row1[None, None, :] <= np.minimum(I, Ic)[:, :, None]
    t2 = ln[None, None, :] <= np.minimum(Imc, Imc.T)[:, :, None]
    base = t1 & t2                                               # condition with the target in slot 3
    ok &= base
    ok &= base.transpose(2, 0, 1)                                # target in slot 1 (g symmetric in its arguments)
    ok &= base.transpose(1, 2, 0)                                # target in slot 2
    # length bounds: l(v) <= l(l) l(m) and conjugate forms, all slots
    L = (ln[:, None] * ln[None, :]); R = (row1[:, None] * row1[None, :]); M = (row1[:, None] * ln[None, :])
    lb = (ln[None, None, :] <= np.minimum(L, R)[:, :, None]) & (row1[None, None, :] <= np.minimum(M, M.T)[:, :, None])
    ok &= lb & lb.transpose(2, 0, 1) & lb.transpose(1, 2, 0)
    return ok
