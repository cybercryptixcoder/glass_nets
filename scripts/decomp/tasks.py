"""Binary-operation tasks for the decompiler's discovery set (D1).

Each task is a table f(a, b) over inputs a, b in 0..n-1 with outputs in 0..n_out-1. The decompiler never imports this
file: it only sees weights. Answer keys used for scoring live in each model's train.json, written by train_table.py.

Some tasks hide their structure on purpose:
  secret_add   a+b mod p under a secret random relabelling of every token and output class
  secret_z8z14 the group Z_8 x Z_14 (112 elements) under a secret random labelling
  secret_s4    the symmetric group S4 (24 elements, non-abelian) under a secret random labelling
  xyxy         a + b + ab mod p  (isomorphic to multiplication through t -> t+1, not told to anyone)
Others have no published mechanism we know of (checked separately before any claim):
  pow          a^b mod p
  sqsum        a^2 + b^2 mod p
"""
import itertools
import numpy as np


def _perm(n, seed):
    return np.random.default_rng(seed).permutation(n)


def _s4():
    els = list(itertools.permutations(range(4)))
    idx = {e: i for i, e in enumerate(els)}
    T = np.zeros((24, 24), int)
    for i, x in enumerate(els):
        for j, y in enumerate(els):
            T[i, j] = idx[tuple(x[y[k]] for k in range(4))]   # (x o y)(k) = x(y(k))
    return T


def table(task, p=113, secret_seed=12345):
    """Return (T, n, n_out, info). T[a, b] is the label. info holds the answer key (secrets included)."""
    a = np.arange(p)[:, None]; b = np.arange(p)[None, :]
    if task == "add":
        return (a + b) % p, p, p, {"rule": f"(a+b) mod {p}"}
    if task == "mul":
        return (a * b) % p, p, p, {"rule": f"(a*b) mod {p}"}
    if task == "xyxy":
        return (a + b + a * b) % p, p, p, {"rule": f"(a+b+ab) mod {p}", "hidden": "(a+1)(b+1)-1: multiplication after t -> t+1"}
    if task == "pow":
        T = np.array([[pow(int(x), int(y), p) for y in range(p)] for x in range(p)])
        return T, p, p, {"rule": f"a^b mod {p}"}
    if task == "sqsum":
        return (a * a + b * b) % p, p, p, {"rule": f"(a^2+b^2) mod {p}"}
    if task == "secret_add":
        pi = _perm(p, secret_seed)                     # label of the true element t is pi[t]
        inv = np.argsort(pi)
        A, B = inv[a], inv[b]
        return pi[(A + B) % p], p, p, {"rule": f"pi((pi^-1 a + pi^-1 b) mod {p})", "secret_pi": pi.tolist()}
    if task == "secret_z8z14":
        n = 112
        els = [(i, j) for i in range(8) for j in range(14)]
        pi = _perm(n, secret_seed); inv = np.argsort(pi)
        T = np.zeros((n, n), int)
        for x in range(n):
            for y in range(n):
                (i1, j1), (i2, j2) = els[inv[x]], els[inv[y]]
                T[x, y] = pi[els.index(((i1 + i2) % 8, (j1 + j2) % 14))]
        return T, n, n, {"rule": "Z_8 x Z_14 under a secret labelling", "secret_pi": pi.tolist(), "elements": els}
    if task == "secret_s4":
        G = _s4(); n = 24
        pi = _perm(n, secret_seed); inv = np.argsort(pi)
        T = pi[G[inv[:, None], inv[None, :]]]
        return T, n, n, {"rule": "S4 composition under a secret labelling", "secret_pi": pi.tolist()}
    if task.startswith("randab"):              # randab<seed>: a random finite abelian group under a random labelling
        rng = np.random.default_rng(int(task[6:]) + 777)
        specs = [[2, 2, 30], [3, 36], [4, 28], [2, 6, 6], [2, 2, 2, 12], [5, 20], [2, 50], [3, 3, 12], [6, 18], [2, 4, 12], [7, 14], [2, 2, 26]]
        d = specs[rng.integers(len(specs))]
        els = list(itertools.product(*[range(x) for x in d])); m = len(els); idx = {e: i for i, e in enumerate(els)}
        pi = rng.permutation(m); inv = np.argsort(pi)
        T = np.zeros((m, m), int)
        for x in range(m):
            for y in range(m):
                ex, ey = els[inv[x]], els[inv[y]]
                T[x, y] = pi[idx[tuple((u + v) % k for u, v, k in zip(ex, ey, d))]]
        return T, m, m, {"rule": f"Z_{' x Z_'.join(map(str, d))} under a secret labelling", "group": d, "secret_pi": pi.tolist()}
    if task.startswith("mulmod"):              # a*b mod n for composite n, e.g. mulmod120 (zero divisors, non-cyclic units)
        m = int(task[6:]); a2 = np.arange(m)[:, None]; b2 = np.arange(m)[None, :]
        return (a2 * b2) % m, m, m, {"rule": f"(a*b) mod {m}"}
    if task.startswith("addmod"):
        m = int(task[6:]); a2 = np.arange(m)[:, None]; b2 = np.arange(m)[None, :]
        return (a2 + b2) % m, m, m, {"rule": f"(a+b) mod {m}"}
    if task == "cubesum":
        return (a ** 3 + b ** 3) % p, p, p, {"rule": f"(a^3+b^3) mod {p}"}
    raise ValueError(task)


TASKS = ["add", "mul", "xyxy", "pow", "sqsum", "secret_add", "secret_z8z14", "secret_s4"]
