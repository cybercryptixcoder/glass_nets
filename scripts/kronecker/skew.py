"""Skew characters chi^{lam/alpha} by the Murnaghan-Nakayama rule, and partition helpers."""
from functools import lru_cache
import math


def contains(lam, alpha):
    return len(alpha) <= len(lam) and all(a <= l for a, l in zip(alpha, lam))


@lru_cache(maxsize=None)
def chi_skew(lam, alpha, mu):
    """Value of the skew character lam/alpha at cycle type mu (|mu| = |lam| - |alpha|)."""
    if not contains(lam, alpha):
        return 0
    if not mu:
        return 1 if lam == alpha else 0
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
            if contains(newlam, alpha):
                total += (-1) ** sign * chi_skew(newlam, alpha, rest)
    return total


def partitions(n, maxpart=None):
    maxpart = n if maxpart is None else maxpart
    if n == 0:
        yield ()
        return
    for k in range(min(n, maxpart), 0, -1):
        for rest in partitions(n - k, k):
            yield (k,) + rest


def z(mu):
    out = 1
    for i in set(mu):
        m = mu.count(i)
        out *= i ** m * math.factorial(m)
    return out


def meet(a, b):
    return tuple(x for x in (min(p, q) for p, q in zip(a, b)) if x > 0)


def conj(p):
    return tuple(sum(1 for x in p if x > j) for j in range(p[0])) if p else ()
