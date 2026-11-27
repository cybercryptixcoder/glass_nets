"""The first search on planted targets (Gate 2 groundwork): a target computed from a known expression of the
vocabulary, on real chess positions (Lichess, via the LD2 data) and on random number sequences; the search must
return the planted expression, or one equal to it on held-out inputs (held-out correlation 1.0).
Run: python3 box/test_search.py"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box.search import best_definitions, vocab  # noqa: E402

fails = 0
items = json.load(open(os.path.expanduser("~/kdata/chess/LD2/meta.json")))
items = items["items"] if isinstance(items, dict) else items
fens = [it["fen"] for it in items[:900]]
d = vocab.get("chess")
cf, ct = vocab.featurize(fens[:60]), vocab.featurize(fens[600:900])
for planted, L in [("(count (and own_P (not defended)))", 5), ("(and opp_N (not defended))", 4), ("(sum (mul see_us hanging_opp))", 4)]:
    t0 = time.time()
    root = vocab.typecheck(planted, d)
    yf, yt = vocab.evaluate(planted, d, cf), vocab.evaluate(planted, d, ct)
    out, sizes = best_definitions(d, cf, yf, L, root, k=3, ctx_test=ct, target_test=yt)
    ok = out and out[0]["heldout_corr"] > 0.9999
    fails += not ok
    print(("PASS " if ok else "FAIL ") + f"chess planted {planted}: best {out[0]} | distinct {sizes} | {time.time() - t0:.0f} s")
# through an ingredient part: the target is simple in terms of a part, and too long in terms of the board
ing = "(and (argmax_sq see_us) (or opp_Q opp_R))"                       # the best exchange, when it takes a queen or rook
tgt = "(count (and (argmax_sq see_us) (or opp_Q opp_R)))"
pf = {"ing": ("Sq", vocab.evaluate(ing, d, cf))}; pt = {"ing": ("Sq", vocab.evaluate(ing, d, ct))}
out, sizes = best_definitions(d, cf, vocab.evaluate(tgt, d, cf), 2, "Pos", k=3, ctx_test=ct,
                              target_test=vocab.evaluate(tgt, d, ct), parts_fit=pf, parts_test=pt)
ok = out and out[0]["heldout_corr"] > 0.9999 and "@ing" in out[0]["definition"]
fails += not ok
print(("PASS " if ok else "FAIL ") + f"chess through a part: target {tgt} (length 7 > L) found as {out[0]}")
ds = vocab.get("seq")
rng = np.random.default_rng(0)
xf, xt = {"x": rng.normal(size=(200, 5))}, {"x": rng.normal(size=(300, 5))}
for planted, L in [("(argmax (relu (sub x (bcast (mean x)))))", 6), ("(sum (relu x))", 3), ("(at (cummax x) 3)", 4)]:
    t0 = time.time()
    root = vocab.typecheck(planted, ds)
    out, sizes = best_definitions(ds, xf, vocab.evaluate(planted, ds, xf), L, root, k=3, ctx_test=xt,
                                  target_test=vocab.evaluate(planted, ds, xt))
    ok = out and out[0]["heldout_corr"] > 0.9999
    fails += not ok
    print(("PASS " if ok else "FAIL ") + f"seq planted {planted}: best {out[0]} | distinct {sizes} | {time.time() - t0:.0f} s")
sys.exit(1 if fails else 0)
