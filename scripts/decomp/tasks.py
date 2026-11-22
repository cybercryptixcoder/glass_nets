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


def _closure(gens):
    idp = tuple(range(len(gens[0]))); H = {idp}; fr = [idp]
    while fr:
        nx = []
        for h in fr:
            for g in gens:
                c = tuple(g[i] for i in h)
                if c not in H:
                    H.add(c); nx.append(c)
        fr = nx
    els = sorted(H); idx = {e: i for i, e in enumerate(els)}
    return np.array([[idx[tuple(a[i] for i in b)] for b in els] for a in els])


def _cyc(*cs, n):
    p = list(range(n))
    for c in cs:
        for i, x in enumerate(c):
            p[x] = c[(i + 1) % len(c)]
    return tuple(p)


def _named_group(name):
    """Non-abelian groups for the blind challenge (Nov 22): permutation groups from generators."""
    if name.startswith("D"):                     # dihedral group of the m-gon, order 2m
        m = int(name[1:]); return _closure([tuple((i + 1) % m for i in range(m)), tuple((-i) % m for i in range(m))]), f"D{m} (dihedral, order {2 * m})"
    if name == "F20":
        return _closure([tuple((i + 1) % 5 for i in range(5)), tuple((2 * i) % 5 for i in range(5))]), "Z5 x| Z4 (Frobenius, order 20)"
    if name == "F21":
        return _closure([tuple((i + 1) % 7 for i in range(7)), tuple((2 * i) % 7 for i in range(7))]), "Z7 x| Z3 (Frobenius, order 21)"
    if name == "SL23":                           # SL(2,3) acting on the 8 nonzero vectors of F3^2
        vecs = [(x, y) for x in range(3) for y in range(3) if (x, y) != (0, 0)]; vi = {v: i for i, v in enumerate(vecs)}
        def act(M): return tuple(vi[((M[0] * x + M[1] * y) % 3, (M[2] * x + M[3] * y) % 3)] for x, y in vecs)
        return _closure([act((1, 1, 0, 1)), act((0, 2, 1, 0))]), "SL(2,3) (order 24)"
    if name == "A4Z3":
        return _closure([_cyc((0, 1, 2), n=7), _cyc((0, 1), (2, 3), n=7), _cyc((4, 5, 6), n=7)]), "A4 x Z3 (order 36)"
    if name == "S3S3":
        return _closure([_cyc((0, 1, 2), n=6), _cyc((0, 1), n=6), _cyc((3, 4, 5), n=6), _cyc((3, 4), n=6)]), "S3 x S3 (order 36)"
    if name == "A5":
        return _closure([_cyc((0, 1, 2, 3, 4), n=5), _cyc((0, 1, 2), n=5)]), "A5 (order 60)"
    raise ValueError(name)


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
    if task.startswith("secretg_"):             # secretg_<name>: a non-abelian group (built from generators) under a secret labelling
        G, gname = _named_group(task[8:]); n = len(G)
        pi = _perm(n, secret_seed + sum(map(ord, task))); inv = np.argsort(pi)
        T = pi[G[inv[:, None], inv[None, :]]]
        return T, n, n, {"rule": f"{gname} composition under a secret labelling", "secret_pi": pi.tolist(), "group_name": gname}
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
