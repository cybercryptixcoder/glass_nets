"""The definer's first search (GAP.md C4; Gate 2 groundwork, not a gate component yet): bottom-up enumeration of the
frozen starting vocabulary by length, with observational-equivalence pruning (an expression whose values on the probe
inputs equal an earlier one's is dropped), scoring candidates against a target by correlation.

Constants are taken from a small fixed grid (the slot semantics of VOCABULARY.md §5 fits them; a grid is the simplest
fitter). Special forms (`after`) are skipped in this first version.
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench", "accept"))
import vocab  # noqa: E402
from vocab.core import CONST  # noqa: E402

GRID = (0, 1, 2, 3)


def _key(v):
    a = np.asarray(v, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        return None
    return hash(np.round(a, 6).tobytes())


def enumerate_exprs(dom, ctx, max_len, skip=("after",), grid=GRID, cap=200000):
    """Distinct expressions by type and length: {type: [(text, value, length)]}."""
    pool = {t: [] for t in list(dom.types) + [CONST]}
    seen = set()
    for c in grid:
        pool[CONST].append((repr(c), c, 1))
    for a in dom.atoms.values():
        v = a.fn(ctx)
        k = _key(v)
        if k is not None and (a.type, k) not in seen:
            seen.add((a.type, k)); pool[a.type].append((a.name, v, 1))
    ops = [o for ov in dom.ops.values() for o in ov if o.name not in skip and o.fn is not None]
    for n in range(2, max_len + 1):
        new = {t: [] for t in pool}
        for o in ops:
            m = len(o.args)
            if m > n - 1:
                continue
            for sizes in itertools.product(range(1, n), repeat=m):
                if sum(sizes) != n - 1:
                    continue
                cands = [[e for e in pool[ta] if e[2] == s] for ta, s in zip(o.args, sizes)]
                if any(not c for c in cands):
                    continue
                for combo in itertools.product(*cands):
                    try:
                        with np.errstate(all="ignore"):
                            v = o.fn(*[c[1] for c in combo])
                    except Exception:  # noqa: BLE001
                        continue
                    k = _key(v)
                    if k is None or (o.ret, k) in seen:
                        continue
                    seen.add((o.ret, k))
                    new[o.ret].append(("(" + " ".join([o.name] + [c[0] for c in combo]) + ")", v, n))
                    if sum(len(x) for x in new.values()) > cap:
                        break
        for t in pool:
            pool[t] += new[t]
    return pool


def best_definitions(dom, ctx_fit, target, max_len, root, k=5, ctx_test=None, target_test=None, cap=60000):
    """Top-k expressions of the root type by |correlation| with the target on the probe inputs (ctx_fit). Expressions
    up to max_len - 1 are enumerated and stored; those of length max_len are streamed and scored without storing
    (memory). The top k are re-scored on held-out inputs when given."""
    pool = enumerate_exprs(dom, ctx_fit, max_len - 1, cap=cap)
    y = np.asarray(target, float).ravel()
    yc = (y - y.mean()) / (y.std() + 1e-12)
    top = []

    def consider(text, v, n):
        x = np.asarray(v, float).ravel()
        if x.shape != y.shape:
            return
        sd = x.std()
        if sd == 0 or not np.isfinite(sd):
            return
        r = abs(float(((x - x.mean()) / sd) @ yc) / len(y))
        top.append((r, text, n))
        if len(top) > 4 * k:
            top.sort(reverse=True); del top[k:]
    for text, v, n in pool[root]:
        consider(text, v, n)
    ops = [o for ov in dom.ops.values() for o in ov if o.ret == root and o.fn is not None and o.name != "after"]
    for o in ops:
        m = len(o.args)
        for sizes in itertools.product(range(1, max_len), repeat=m):
            if sum(sizes) != max_len - 1:
                continue
            cands = [[e for e in pool[ta] if e[2] == s_] for ta, s_ in zip(o.args, sizes)]
            if any(not c for c in cands):
                continue
            for combo in itertools.product(*cands):
                try:
                    with np.errstate(all="ignore"):
                        v = o.fn(*[c[1] for c in combo])
                except Exception:  # noqa: BLE001
                    continue
                consider("(" + " ".join([o.name] + [c[0] for c in combo]) + ")", v, max_len)
    top.sort(reverse=True)
    out = []
    for r, text, n in top[:k]:
        row = {"definition": text, "length": n, "probe_corr": round(r, 4)}
        xp = np.asarray(vocab.evaluate(text, dom, ctx_fit), float).ravel()
        slope, offset = np.polyfit(xp, y, 1) if xp.std() > 0 else (0.0, float(y.mean()))
        row["scale"], row["offset"] = round(float(slope), 6), round(float(offset), 6)   # target = scale * value + offset
        if ctx_test is not None:
            xv = np.asarray(vocab.evaluate(text, dom, ctx_test), float).ravel()
            yt = np.asarray(target_test, float).ravel()
            row["heldout_corr"] = round(float(abs(np.corrcoef(xv, yt)[0, 1])), 4) if xv.std() > 0 else 0.0
            row["heldout_r2_affine"] = round(float(1 - np.mean((slope * xv + offset - yt) ** 2) / np.var(yt)), 4) if np.var(yt) > 0 else None
        out.append(row)
    return out, {t: len(v) for t, v in pool.items()}
