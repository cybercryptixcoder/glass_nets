"""Fuzz test before the freeze: random well-typed expressions of every length up to L in every domain, evaluated on
small inputs. A base operation that raises on well-typed arguments is a bug. Run: python3 bench/accept/vocab/fuzz_vocab.py"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vocab  # noqa: E402
from vocab.core import CONST  # noqa: E402

warnings.filterwarnings("ignore")


def random_expr(dom, typ, size, rng, depth=0):
    """A random expression of the given type with at most `size` symbols (None if none found)."""
    for _ in range(30):
        if typ == CONST:
            return round(float(rng.choice([-2, -1, 0, 0.5, 1, 2, 3])), 2) if dom.const_values == "number" else \
                str(rng.choice(["a", "b", "x", "1", "2"]))
        atoms = [a.name for a in dom.atoms.values() if a.type == typ]
        if size <= 1 or (atoms and rng.random() < 0.3):
            return str(rng.choice(atoms)) if atoms else None
        ops = [o for ov in dom.ops.values() for o in ov if o.ret == typ and len(o.args) < size]
        if not ops:
            return str(rng.choice(atoms)) if atoms else None
        o = ops[rng.integers(len(ops))]
        budget = size - 1
        args, ok = [], True
        for i, ta in enumerate(o.args):
            share = max(1, budget - (len(o.args) - i - 1))
            k = int(rng.integers(1, share + 1))
            a = random_expr(dom, ta, k, rng, depth + 1)
            if a is None:
                ok = False; break
            args.append(a); budget -= vocab.length(a if not isinstance(a, str) else a)
        if ok and budget >= 0:
            return [o.name] + args
    return None


def ctx_for(name, rng):
    if name == "chess":
        return vocab.featurize(["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                                "4k3/8/4p3/3q4/8/8/3R4/4K3 b - - 0 1"])
    if name == "seq":
        return {"x": rng.normal(size=(4, 5))}
    if name == "tok":
        return {"tokens": [["a", "x", "b"], ["x", "x"], ["b", "a", "x", "a"]]}
    if name == "fluid":
        return {k: rng.normal(size=(2, 8, 8)) for k in vocab.FLUID_FIELDS}
    return {"nets": [{"W": [rng.normal(size=(4, 3)), rng.normal(size=(2, 4))], "b": [np.zeros(4), np.zeros(2)]},
                     {"W": [rng.normal(size=(3, 3))], "b": [np.zeros(3)]}]}


def tolist(t):
    if isinstance(t, list):
        return "(" + " ".join(tolist(x) for x in t) + ")"
    if isinstance(t, str) and not t.replace(".", "").replace("-", "").isdigit() and t not in vocab.get("tok").atoms \
            and t not in {a for d in vocab.DOMAINS for a in vocab.get(d).atoms} and not t.startswith(("(", "@", "$")):
        return '"' + t + '"'
    return str(t)


def main(n=400):
    rng = np.random.default_rng(0)
    bad = 0
    for name in vocab.DOMAINS:
        dom = vocab.get(name)
        ctx = ctx_for(name, rng)
        done = errors = 0
        for i in range(n):
            root = dom.roots[i % len(dom.roots)]
            e = random_expr(dom, root, int(rng.integers(1, dom.L + 1)), rng)
            if e is None:
                continue
            text = tolist(e)
            ok, why, _t = vocab.admissible(text, dom)
            if not ok:
                continue
            try:
                vocab.evaluate(text, dom, ctx)
            except Exception as err:  # noqa: BLE001
                errors += 1
                if errors <= 3:
                    print(f"   {name}: {text} -> {type(err).__name__}: {err}")
            done += 1
        print(f"{name}: {done} admissible random expressions evaluated, {errors} raised")
        bad += errors
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
