"""Kill test, part 1: known necessary conditions (B1) against exact truth, per n."""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from kron_feats import load, b1_masks
res = {}
for n in range(6, 19):
    parts, g = load(n)
    nz = g != 0
    ok = b1_masks(parts)
    fn = int((nz & ~ok).sum())                      # a necessary condition violated by a nonzero triple: must be 0
    acc = float((ok == nz).mean())
    hard = int((ok & ~nz).sum())                   # zeros the known conditions miss
    res[n] = dict(triples=int(nz.size), nonzero=float(nz.mean()), b1_acc=acc, b1_false_neg=fn,
                  zeros=int((~nz).sum()), zeros_caught=int((~nz & ~ok).sum()), hard_zeros=hard,
                  b1_balanced=float(0.5 * ((ok & nz).sum() / nz.sum() + (~ok & ~nz).sum() / (~nz).sum())))
    r = res[n]
    print(f"n={n:2d} nonzero={r['nonzero']:.3f} B1 acc={acc:.4f} balanced={r['b1_balanced']:.4f} "
          f"zeros caught {r['zeros_caught']}/{r['zeros']} hard zeros {hard} false-neg {fn}", flush=True)
json.dump(res, open("results/kronecker/kill_b1.json", "w"), indent=1)
