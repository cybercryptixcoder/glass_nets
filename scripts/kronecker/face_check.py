"""Check: for nu_1 = |lam ∩ mu|, g(lam,mu,nu) = < chi^{lam/a} chi^{mu/a}, chi^{nubar} >, a = lam ∩ mu."""
import sys, numpy as np
from fractions import Fraction
sys.path.insert(0, "scripts/kronecker")
from skew import chi_skew, partitions, z, meet
from kron_feats import load
for n in range(4, 11):
    parts, g = load(n); P = [tuple(int(x) for x in p if x) for p in parts]; idx = {p: i for i, p in enumerate(P)}
    bad = tested = 0
    for l in P:
        for m in P:
            a = meet(l, m); k = sum(a); rest = n - k
            for v in P:
                if v[0] != k: continue
                vb = v[1:]
                if rest == 0:
                    val = 1 if (l == m) else 0   # a = l = m
                else:
                    val = sum(Fraction(chi_skew(l, a, c) * chi_skew(m, a, c) * chi_skew(vb, (), c), z(c)) for c in partitions(rest))
                tested += 1
                if val != int(g[idx[l], idx[m], idx[v]]): bad += 1
    print(f"n={n}: tested {tested} face triples (nu_1 = |lam∩mu|), mismatches {bad}", flush=True)
