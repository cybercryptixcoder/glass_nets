"""D1: test the nested-discrete-log mechanism in a network trained on a^b mod p, for any prime p; and the same
ring-structure tests for a network trained on a*b mod a composite n.

  python3 scripts/decomp/pow_mechanism.py <model_id> [--p P]
  python3 scripts/decomp/pow_mechanism.py <model_id> --mulmod N

Hypothesis (found on dc_pow_s1; docs/decomp/POW_MECHANISM.md). With g a generator mod p and x = log_g(a) in
Z_{p-1}, the network computes x*b in Z_{p-1} by splitting it by CRT into prime-power parts Z_q, and inside each part
stores a residue by its multiplicative structure (valuation and unit-group characters, a second logarithm).

Tests (internal activations; logits only to score the causal patches):
  T1  additivity: x1 = F(a) + G(b) at the MLP's input
  T2  CRT split: F as a sum of functions of (x mod q) over the prime powers q of p-1, versus other splits
  T3  inside each part: spectrum over additive vs multiplicative characters of Z_q
  T4  causal, per CRT part and side: swap only that part's features with another input's; predicted output is the
      CRT recombination with that part taken from the other input
  T5  causal, second level: for each part, swap one cyclic factor of the unit group of Z_q (Legendre-type characters)
Writes results/decomp/<id>_pow.json.
"""
import argparse, itertools, json, os, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lab"))
from decomp import program as pg


def factor(n):
    f, d = {}, 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1; n //= d
        d += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def generator(p):
    fs = factor(p - 1)
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in fs):
            return g


def unit_factors(q):
    """Cyclic factors of the unit group of Z_q as (generator, order) pairs: Z_q* = prod <g_i>."""
    ell = min(factor(q))
    if ell == 2:
        if q == 2:
            return []
        if q == 4:
            return [(q - 1, 2)]
        return [(q - 1, 2), (5, q // 4)]
    # odd prime power: cyclic; find a generator
    phi = q // ell * (ell - 1)
    for g in range(2, q):
        if np.gcd(g, q) == 1 and all(pow(g, phi // r, q) != 1 for r in factor(phi)):
            return [(g, phi)]


def unit_coords(q):
    """Map each unit r of Z_q to its exponents along unit_factors(q)."""
    fac = unit_factors(q)
    table = {}
    for es in itertools.product(*[range(o) for _, o in fac]):
        r = 1
        for (g, _), e in zip(fac, es):
            r = r * pow(g, e, q) % q
        table[r] = es
    return fac, table


def run(model_id, p, n_patch=3000, seed=0):
    """a^b mod p: x = log_g(a) in Z_{p-1} (a = 0 left out), y = b, the ring is Z_{p-1}, the answer is g^(x*y)."""
    g = generator(p); n1 = p - 1
    dl = np.full(p, -1); x = 1
    for e in range(n1):
        dl[x] = e; x = x * g % p
    ex = np.array([pow(g, e, p) for e in range(n1)])
    truth = np.array([[pow(int(a), int(b), p) for b in range(p)] for a in range(p)])
    rep = run_ring(model_id, p, n1, dl, np.arange(p), lambda z: ex[z], truth, a_min=1, n_patch=n_patch, seed=seed)
    rep.update({"p": p, "generator": g, "p_minus_1": n1})
    return rep


def run_ring(model_id, n_sym, n1, dl, ycoord, out_of, truth, a_min=0, n_patch=3000, seed=0):
    """Generic core. The network's inputs are symbols 0..n_sym-1; symbol a carries x = dl[a] in the ring Z_n1 (-1 = no
    coordinate), symbol b carries y = ycoord[b]; the answer for (a, b) is out_of(x * y mod n1)."""
    W = dict(np.load(os.path.join(ROOT, "lab", "models", model_id, "weights.npz")))
    rng = np.random.default_rng(seed)
    p = n_sym; t = np.asarray(ycoord); ex = None
    qs = [l ** k for l, k in sorted(factor(n1).items())]
    I = pg.internals(W, pg.all_pairs(p)); x1 = I["x1"][:, -1].reshape(p, p, -1); net = I["logits"].argmax(-1).reshape(p, p)
    mu = x1.mean((0, 1)); F = x1.mean(1) - mu; G = x1.mean(0) - mu
    R = x1 - F[:, None] - G[None] - mu
    rep = {"model": model_id, "ring": n1, "crt_parts": qs, "accuracy": float((net == truth).mean()),
           "T1_additive_share": float(1 - (R ** 2).sum() / ((x1 - mu) ** 2).sum())}

    def onehot(v, m):
        return [(v % m == r).astype(float) for r in range(m)]

    def explained(X, feats):
        Xc = X - X.mean(0); Fm = np.column_stack([np.ones(len(X))] + feats)
        B, *_ = np.linalg.lstsq(Fm, Xc, rcond=None); Rr = Xc - Fm @ B
        return float(1 - (Rr ** 2).sum() / (Xc ** 2).sum())

    okA = dl >= 0
    sides = {"a": (F[okA], dl[okA]), "b": (G, t)}
    # T2: CRT split vs alternatives (each single part, the whole ring, and every coarser split into two factors)
    T2 = {}
    for s, (X, c) in sides.items():
        res = {"crt(" + "+".join(map(str, qs)) + ")": explained(X, sum((onehot(c, q) for q in qs), []))}
        for q in qs:
            res[f"only mod {q}"] = explained(X, onehot(c, q))
        divs = [d for d in range(2, n1) if n1 % d == 0 and np.gcd(d, n1 // d) == 1 and d < n1 // d]
        for d in divs:
            if d in qs and n1 // d in qs:
                continue
            res[f"split {d}+{n1 // d}"] = explained(X, onehot(c, d) + onehot(c, n1 // d))
        T2[s] = res
    rep["T2_split_r2"] = T2

    def parts(X, c):
        n = len(X); cols = [np.ones(n)]; idx = {}
        for q in qs:
            idx[q] = (len(cols), len(cols) + q); cols += onehot(c, q)
        B, *_ = np.linalg.lstsq(np.column_stack(cols), X, rcond=None)
        return {q: B[a:b] for q, (a, b) in idx.items()}

    PA, PB = parts(*sides["a"]), parts(*sides["b"])
    # T3: additive vs multiplicative spectra inside each part
    T3 = {}
    for s, P in (("a", PA), ("b", PB)):
        T3[s] = {}
        for q in qs:
            C = P[q] - P[q].mean(0); r = np.arange(q)
            add = np.array([np.abs(np.exp(-2j * np.pi * k * r / q) @ C).__pow__(2).sum() for k in range(1, q)])
            fac, tab = unit_coords(q)
            us = sorted(tab); E = np.array([tab[u] for u in us]) if fac else np.zeros((len(us), 0))
            Cu = C[us] - C[us].mean(0)
            mul = []
            for js in itertools.product(*[range(o) for _, o in fac]):
                ch = np.exp(2j * np.pi * sum(E[:, i] * js[i] / fac[i][1] for i in range(len(fac)))) if fac else np.ones(len(us))
                mul.append((np.abs(ch.conj() @ Cu) ** 2).sum())
            mul = np.array(mul)
            def top_share(v, k=3):
                v = np.sort(v)[::-1]; return float(v[:k].sum() / max(v.sum(), 1e-12))
            T3[s][q] = {"additive_top3_share": top_share(add), "multiplicative_top3_share_on_units": top_share(mul),
                        "units_share_of_part": float((Cu ** 2).sum() / max((C ** 2).sum(), 1e-12))}
    rep["T3_spectra"] = T3

    def crt(res):
        z = 0
        for q, r in res.items():
            m = n1 // q; z += r * m * pow(m, -1, q)
        return z % n1

    N = n_patch
    A = rng.integers(a_min, p, N); A2 = rng.integers(a_min, p, N); B = rng.integers(0, p, N); B2 = rng.integers(0, p, N)
    xa, xa2 = dl[A], dl[A2]
    toks = np.stack([A, B, np.full(N, p)], 1)

    def patched(D):
        return pg.internals(W, toks, patch1=lambda v: v + D)["logits"].argmax(-1)

    T4 = {}
    for side in ("a", "b"):
        for q in qs:
            D = np.zeros((N, F.shape[1])); want = np.zeros(N, int)
            for i in range(N):
                xs, bs = (xa2[i] if side == "a" else xa[i]), (B2[i] if side == "b" else B[i])
                if side == "a":
                    D[i] = PA[q][xa2[i] % q] - PA[q][xa[i] % q]
                else:
                    D[i] = PB[q][B2[i] % q] - PB[q][B[i] % q]
                res = {qq: (xa[i] * B[i]) % qq for qq in qs}
                res[q] = (xs * bs) % q
                want[i] = out_of(crt(res))
            out = patched(D)
            sel = want != net[A, B]
            # control: same norms, random directions inside the span of all CRT features
            Q = np.linalg.qr(rng.standard_normal((F.shape[1], q - 1)))[0]
            Dr = rng.standard_normal((N, q - 1)) @ Q.T
            Dr *= (np.linalg.norm(D, axis=1) / np.maximum(np.linalg.norm(Dr, axis=1), 1e-12))[:, None]
            outr = patched(Dr)
            T4[f"{side} mod {q}"] = {"success": float((out[sel] == want[sel]).mean()), "n": int(sel.sum()),
                                     "random_control": float((outr[sel] == want[sel]).mean())}
    rep["T4_crt_swaps"] = T4

    # T5: one cyclic factor of the unit group of Z_q, for b (both b and b' units mod q)
    T5 = {}
    for q in qs:
        fac, tab = unit_coords(q)
        if len(tab) < 2:
            continue
        us = sorted(tab); E = np.array([tab[u] for u in us])
        js_all = list(itertools.product(*[range(o) for _, o in fac]))
        Bm = np.array([np.exp(2j * np.pi * sum(E[:, i] * js[i] / fac[i][1] for i in range(len(fac)))) for js in js_all]).T
        beta = np.linalg.solve(Bm, PB[q][us].astype(complex))
        def feat(es):
            ch = np.array([np.exp(2j * np.pi * sum(es[i] * js[i] / fac[i][1] for i in range(len(fac)))) for js in js_all])
            return (ch @ beta).real
        inv = {v: k for k, v in tab.items()}
        for fi, (gen, order) in enumerate(fac):
            # split this cyclic factor further into its prime-order pieces when it is not of prime-power order
            for ell in sorted(factor(order)):
                sub = ell ** factor(order)[ell]
                D = np.zeros((N, F.shape[1])); want = np.full(N, -1)
                for i in range(N):
                    rb, rb2 = B[i] % q, B2[i] % q
                    if rb not in tab or rb2 not in tab:
                        continue
                    e, e2 = list(tab[rb]), tab[rb2]
                    new = list(e)
                    # replace the order-`sub` component of factor fi: CRT inside Z_order
                    comp = order // sub
                    cand = [z for z in range(order) if z % sub == e2[fi] % sub and z % comp == e[fi] % comp]
                    new[fi] = cand[0]
                    if tuple(new) == tuple(e):
                        continue
                    D[i] = feat(new) - feat(e)
                    res = {qq: (xa[i] * B[i]) % qq for qq in qs}
                    res[q] = (xa[i] * inv[tuple(new)]) % q
                    want[i] = out_of(crt(res))
                sel = want >= 0
                if sel.sum() < 20:
                    continue
                out = patched(D)
                T5[f"b mod {q}: unit factor <{gen}> of order {order}, piece of order {sub}"] = {
                    "success": float((out[sel] == want[sel]).mean()), "n": int(sel.sum()),
                    "unpatched_kept": float((out[sel] == net[A, B][sel]).mean())}
    rep["T5_unit_factor_swaps"] = T5
    return rep


def run_mulmod(model_id, n):
    """a*b mod n (composite n): the ring is Z_n itself, x = a, y = b, the answer is x*y."""
    truth = (np.arange(n)[:, None] * np.arange(n)[None, :]) % n
    return run_ring(model_id, n, n, np.arange(n), np.arange(n), lambda z: z, truth, a_min=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--p", type=int, default=None)
    ap.add_argument("--mulmod", type=int, default=None, help="the network computes a*b mod this n (composite) instead of a^b mod p")
    a = ap.parse_args()
    if a.mulmod:
        rep = run_mulmod(a.model, a.mulmod); rep.setdefault("p_minus_1", rep["ring"])
        out = a.model + "_ring.json"
    else:
        p = a.p or json.load(open(os.path.join(ROOT, "lab", "models", a.model, "train.json"))).get("p", 113)
        rep = run(a.model, p); out = a.model + "_pow.json"
    os.makedirs(os.path.join(ROOT, "results", "decomp"), exist_ok=True)
    json.dump(rep, open(os.path.join(ROOT, "results", "decomp", out), "w"), indent=1)
    print(f"{a.model}: ring Z_{rep['ring']} = {' x '.join(map(str, rep['crt_parts']))}, accuracy {rep['accuracy']:.4f}, additive {rep['T1_additive_share']:.4f}")
    for s, r in rep["T2_split_r2"].items():
        print(f"  T2 side {s}: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()))
    for s, r in rep["T3_spectra"].items():
        for q, v in r.items():
            print(f"  T3 side {s} mod {q}: additive top-3 {v['additive_top3_share']:.2f}, multiplicative top-3 (units) {v['multiplicative_top3_share_on_units']:.2f}")
    for k, v in rep["T4_crt_swaps"].items():
        print(f"  T4 {k:10s}: success {v['success']:.3f} (n={v['n']}), random control {v['random_control']:.3f}")
    for k, v in rep["T5_unit_factor_swaps"].items():
        print(f"  T5 {k}: success {v['success']:.3f} (n={v['n']}), unpatched kept {v['unpatched_kept']:.3f}")


if __name__ == "__main__":
    main()
