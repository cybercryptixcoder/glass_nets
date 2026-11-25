"""KR1b (B) rule search, method as declared in ledger row KR1e. Dev n = 8..14 only.
Usage: rule_search.py [shuffle]   writes results/kronecker/rule_search[_shuffled].json"""
import sys, os, json, itertools, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import face_reduce as fr
from gwxz import gwxz_zero
from fathook import fathook_zero
from skew import conj
_b1 = fr.b1
fr.b1 = lambda A, B, t: _b1(A, B, t) and not gwxz_zero(A, B, t) and not fathook_zero(A, B, t)
fr.p2.cache_clear()
from kron_feats import load, b1_masks
from b2_lgbm_lib import orbit_ids

def quot_sizes(p):
    L = len(p) + (len(p) % 2)
    beta = [p[i] + (L - 1 - i) if i < len(p) else (L - 1 - i) for i in range(L)]
    sizes = []
    for r in (0, 1):
        run = sorted([b // 2 for b in beta if b % 2 == r], reverse=True); m = len(run)
        sizes.append(sum(run[i] - (m - 1 - i) for i in range(m)))
    core_empty = sum(1 for b in beta if b % 2 == 0) == sum(1 for b in beta if b % 2 == 1)
    return core_empty, sizes

def inter(a, b): return sum(min(x, y) for x, y in zip(a, b))

def dvir_slack(T):
    best = 99
    for (x, y, w) in (T, (T[1], T[2], T[0]), (T[2], T[0], T[1])):
        for (u, v, s) in ((x, y, w), (conj(x), conj(y), w), (conj(x), y, conj(w)), (x, conj(y), conj(w))):
            best = min(best, inter(u, v) - s[0])
    return best

def fh_slack(T):
    best = 99
    for (A, B, t) in (T, (T[1], T[2], T[0]), (T[2], T[0], T[1])):
        for k1 in range(len(A) + 1):
            l1 = A[k1] if k1 < len(A) else 0
            for k2 in range(len(B) + 1):
                l2 = B[k2] if k2 < len(B) else 0
                K, L = k1 * k2 + l1 * l2, k1 * l2 + k2 * l1
                if len(t) > K: best = min(best, L - t[K])
    return best

FEATS = ["n_even", "c_cat", "c_tw", "c_hook", "c_nbhook", "c_rect", "c_fh", "c_sc", "c_core0", "c_qbal", "n_eqc", "dvir_slack", "fh_slack"]

def features(T, n):
    k = n // 2
    def cat(p): return n % 2 == 0 and (p == (k, k) or p == (2,) * k)
    def tw(p): return len(p) <= 2 or p[0] <= 2
    def hook(p): return len(p) >= 2 and p[1] == 1
    def nbhook(p): return hook(p) and abs((p[0] - 1) - (len(p) - 1)) <= 1
    qs = [quot_sizes(p) for p in T]
    f = [n % 2 == 0, sum(map(cat, T)), sum(map(tw, T)), sum(map(hook, T)), sum(map(nbhook, T)),
         sum(len(set(p)) == 1 for p in T), sum(len(set(p)) <= 2 for p in T), sum(p == conj(p) for p in T),
         sum(q[0] for q in qs), sum(abs(q[1][0] - q[1][1]) <= 1 for q in qs),
         sum(x == y or x == conj(y) for x, y in ((T[0], T[1]), (T[0], T[2]), (T[1], T[2]))),
         min(dvir_slack(T), 3), min(max(fh_slack(T), 0), 3)]
    return [int(x) for x in f]

def build(ns):
    X, W, Y, N = [], [], [], []
    for n in ns:
        t0 = time.time()
        parts, g = load(n); P = len(parts); Pp = [tuple(int(x) for x in p if x) for p in parts]
        nz = (g != 0).ravel(); ok = b1_masks(parts).ravel(); orb = orbit_ids(parts).ravel()
        uo, inv = np.unique(orb, return_inverse=True); cnt = np.bincount(inv)
        u, rem = np.divmod(uo, P * P); v, w = np.divmod(rem, P)
        for i in np.flatnonzero(ok[uo]):
            T = (Pp[u[i]], Pp[v[i]], Pp[w[i]])
            if not fr.p2(*fr.canon(*T)): continue
            X.append(features(T, n)); W.append(int(cnt[i])); Y.append(int(not nz[uo[i]])); N.append(n)
        print("built", n, len(X), f"{time.time() - t0:.0f}s", flush=True)
    return np.array(X), np.array(W), np.array(Y), np.array(N)

def atoms(X):
    out = []
    for j, name in enumerate(FEATS):
        vals = sorted(set(X[:, j].tolist()))
        for c in vals:
            out.append((f"{name}=={c}", X[:, j] == c))
            if c > min(vals): out.append((f"{name}>={c}", X[:, j] >= c))
    return out

def evaluate(mask, W, Y, N, ns):
    cov, fz = {}, {}
    for n in ns:
        sel = N == n
        res = (Y == 1) & sel; non = (Y == 0) & sel
        cov[n] = float(W[mask & res].sum() / max(1, W[res].sum()))
        fz[n] = float(W[mask & non].sum())
    return cov, fz

def main():
    shuffle = len(sys.argv) > 1 and sys.argv[1] == "shuffle"
    ns = list(range(8, 15)); score_ns = [10, 11, 12, 13, 14]
    X, W, Y, N = build(ns)
    if shuffle:
        rng = np.random.default_rng(0)
        for n in ns:
            idx = np.flatnonzero(N == n); Y[idx] = Y[rng.permutation(idx)]
    nonzero_total = {}
    for n in ns:                                   # all nonzero ordered triples at n (the budget's base)
        parts, g = load(n); nonzero_total[n] = int((g != 0).sum())
    budget = np.array([0.001 * nonzero_total[n] for n in ns])
    # aggregate orbits with identical feature vectors (the search only sees features), per n and label
    U, inv = np.unique(X, axis=0, return_inverse=True); inv = inv.ravel()
    ni = np.searchsorted(ns, N)
    RES = np.zeros((len(U), len(ns))); NON = np.zeros((len(U), len(ns)))
    np.add.at(RES, (inv[Y == 1], ni[Y == 1]), W[Y == 1]); np.add.at(NON, (inv[Y == 0], ni[Y == 0]), W[Y == 0])
    res_tot = RES.sum(0); sc = np.array([ns.index(n) for n in score_ns])
    print("distinct feature vectors", len(U), "residue per n", dict(zip(ns, res_tot.tolist())), flush=True)
    A = atoms(U); print("atoms", len(A), flush=True)
    def gen():
        for a1, m1 in A: yield (a1,), m1
        for (a1, m1), (a2, m2) in itertools.combinations(A, 2): yield (a1, a2), m1 & m2
        for (a1, m1), (a2, m2), (a3, m3) in itertools.combinations(A, 3): yield (a1, a2, a3), m1 & m2 & m3
    chosen, union = [], np.zeros(len(U), dtype=bool)
    for step in range(3):
        best = None
        for names, m in gen():
            u = union | m
            fz = NON[u].sum(0)
            if (fz > budget).any(): continue
            cov = RES[u].sum(0) / np.maximum(1, res_tot)
            s_ = (cov[sc].min(), cov[sc].mean())        # smallest coverage; ties broken by the mean (KR1e amendment)
            if best is None or s_ > best[0]: best = (s_, names, m, cov, fz)
        if best is None or (chosen and best[0] <= (chosen[-1]["min_cov"], chosen[-1]["mean_cov"])): break
        union = union | best[2]
        chosen.append(dict(conj=list(best[1]), min_cov=float(best[0][0]), mean_cov=float(best[0][1]), cov={str(n): float(c) for n, c in zip(ns, best[3])},
                           false_zero_triples={str(n): float(f) for n, f in zip(ns, best[4])}))
        print("step", step, chosen[-1], flush=True)
    out = dict(shuffled=shuffle, features=FEATS, dev_ns=ns, score_ns=score_ns, budget={str(n): float(b) for n, b in zip(ns, budget)},
               distinct_feature_vectors=int(len(U)), rule=chosen, atoms_used=sum(len(c["conj"]) for c in chosen))
    os.makedirs("results/kronecker", exist_ok=True)
    json.dump(out, open(f"results/kronecker/rule_search{'_shuffled' if shuffle else ''}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
