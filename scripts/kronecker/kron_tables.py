"""Exact Kronecker coefficients g(lam, mu, nu) of S_n for every triple of partitions of n.

g = sum over conjugacy classes c of chi_lam(c) chi_mu(c) chi_nu(c) / z_c, with characters from the
Murnaghan-Nakayama rule (beta-numbers). Exact: computed mod three primes below 2^20 (float64 matrix
products stay exact: each term < 2^40, sums < 2^49), reconstructed from two primes by CRT and checked
against the third. g < sqrt(n!) < 2^40 for n <= 20, so two primes determine it.
Output: <out>/kron_n<n>.npz with parts (P x n, zero-padded), g (P x P x P, int64 for n<=16, else
int32), dims, classes. Usage: python3 kron_tables.py OUTDIR n1 n2 ...
"""
import sys, os, time, math
import numpy as np
from functools import lru_cache

sys.setrecursionlimit(10000)


def partitions(n, maxpart=None):
    maxpart = n if maxpart is None else maxpart
    if n == 0:
        yield ()
        return
    for k in range(min(n, maxpart), 0, -1):
        for rest in partitions(n - k, k):
            yield (k,) + rest


@lru_cache(maxsize=None)
def chi(lam, mu):
    """Character of the irreducible lam at the class of cycle type mu (Murnaghan-Nakayama)."""
    if not mu:
        return 1 if not lam else 0
    r, rest = mu[0], mu[1:]
    k = len(lam)
    beta = [lam[i] + (k - 1 - i) for i in range(k)]
    bset = set(beta)
    total = 0
    for b in beta:
        t = b - r
        if t >= 0 and t not in bset:
            sign = sum(1 for c in beta if t < c < b)
            nb = sorted([c for c in beta if c != b] + [t], reverse=True)
            kk = len(nb)
            newlam = tuple(x for x in (nb[i] - (kk - 1 - i) for i in range(kk)) if x > 0)
            total += (-1) ** sign * chi(newlam, rest)
    return total


def z(mu):
    out = 1
    for i in set(mu):
        m = mu.count(i)
        out *= i ** m * math.factorial(m)
    return out


PRIMES = [1048573, 1048571, 1048559]   # three primes below 2^20


def kron_mod(X, zs, p):
    """g mod p for all triples: X (P x C) characters, zs class z-values."""
    P, C = len(X), len(X[0])
    Xp = np.array([[int(v) % p for v in row] for row in X], dtype=np.float64)
    w = np.array([pow(int(zz) % p, p - 2, p) for zz in zs], dtype=np.float64)
    G = np.empty((P, P, P), dtype=np.int64)
    for a in range(P):
        A = np.fmod(Xp[a] * w, p)
        B = np.fmod(Xp * A[None, :], p)
        G[a] = np.fmod(B @ Xp.T, p).astype(np.int64)
    return G


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for n in map(int, sys.argv[2:]):
        t0 = time.time()
        parts = list(partitions(n))
        P = len(parts)
        X = [[chi(l, m) for m in parts] for l in parts]          # classes indexed by the same partitions
        zs = [z(m) for m in parts]
        dims = np.array([X[i][P - 1] for i in range(P)], dtype=object)   # class (1^n) is the last partition
        t1 = time.time()
        Gs = [kron_mod(X, zs, p) for p in PRIMES]
        p1, p2, p3 = PRIMES
        inv = pow(p1, p2 - 2, p2)
        g = Gs[0] + p1 * (((Gs[1] - Gs[0]) % p2) * inv % p2)        # CRT, < p1*p2 ~ 1.1e12, exact in int64
        assert ((g % p3) == Gs[2]).all(), "third prime disagrees"
        # identity check: sum_nu g dim(nu) = dim(lam) dim(mu)
        dimf = np.array([int(d) for d in dims], dtype=object)
        for a in range(0, P, max(1, P // 7)):
            for b in range(0, P, max(1, P // 7)):
                s = sum(int(g[a, b, c]) * int(dimf[c]) for c in range(P))
                assert s == int(dimf[a]) * int(dimf[b]), (n, a, b)
        arr = np.zeros((P, n), dtype=np.int8)
        for i, l in enumerate(parts):
            arr[i, :len(l)] = l
        gd = g.astype(np.int64 if n <= 16 else np.int32)
        np.savez_compressed(os.path.join(out, f"kron_n{n}.npz"), parts=arr, g=gd,
                            dims=np.array([str(d) for d in dims]), zs=np.array([str(v) for v in zs]))
        nz = (g != 0).mean()
        print(f"n={n} partitions={P} chars {t1 - t0:.1f}s total {time.time() - t0:.1f}s nonzero_frac={nz:.4f} max_g={int(g.max())}", flush=True)


if __name__ == "__main__":
    main()
