"""The box pointed at a real network: Leela's LD2, its evaluation at block 10 (the last block before the heads).

Stages, as on the other targets, with what differs here:
1. Map: the block-10 sparse dictionary (512 variables, Rosetta's), ranked by attribution to LD2's value; the 16 most
   important are the candidate parts of the top level (RESULTS.md of the Rosetta sprint, section 9).
2. Top level: LD2's value defined through those parts, by an invented operation, a linear readout of at most four
   parts' board-wide totals plus an offset (five fitted constants, the budget), the best four chosen on training
   positions and scored on held-out ones. Control line: the same search over 16 variables drawn from the rest.
3. Define each part of the top level from the board: the vocabulary search over position-level expressions (60 probe
   positions), held-out correlation on 600 test positions, against a line from controls (the same search on three
   variables outside the top 16, the best of their held-out correlations). Below the line: unresolved, in the ledger.
4. Certify the top level by interventions: for each of its parts, swap that variable's board map (its activation on
   all 64 squares, through the dictionary's decoder direction) from position b into position a, and compare the
   change in LD2's value with the change the top level predicts. Control: the same swap applied to another variable
   outside the top 16 (the prediction stays the part's). Score: correlation of predicted and actual changes; line:
   the 99th percentile of the best of K control correlations, K = the number of parts.
Run: python3 box/leela_box.py"""
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_feats as F  # noqa: E402
from rosetta.run_feats import L, CV, CI  # noqa: E402
from rosetta.fresh_where import totals  # noqa: E402
from box.search import best_definitions, vocab  # noqa: E402

t0 = time.time()
rng = np.random.default_rng(0)
att = np.array(json.load(open(f"{L.RES}/glass2_attr.json"))["attribution"]); order = np.argsort(-att)
top = [int(f) for f in order[:16]]
others = [int(f) for f in order[64:]]
T = totals(np.asarray(CV, dtype=np.float32), CI)                     # [20000, 512] board-wide totals
TR, TE = np.asarray(L.TR), np.asarray(L.TE)
q = np.asarray(L.q, float)
r2 = lambda p, y: float(1 - np.mean((p - y) ** 2) / np.var(y))  # noqa: E731


def best_readout(cands, k=4):
    best = (-9, None, None)
    for S in itertools.combinations(cands, k):
        A = np.column_stack([T[TR][:, list(S)], np.ones(len(TR))])
        w, *_ = np.linalg.lstsq(A, q[TR], rcond=None)
        s = r2(np.column_stack([T[TE][:, list(S)], np.ones(len(TE))]) @ w, q[TE])
        if s > best[0]:
            best = (s, S, w)
    return best


# ---------------------------------------------------------------- 2. the top level
s_top, S_top, w_top = best_readout(top)
ctrl = []
for d in range(3):
    pool = list(rng.choice(others, 16, replace=False))
    ctrl.append(best_readout(pool)[0])
line_top = max(ctrl)
print(f"top level: value = readout of {S_top}; held-out R2 {s_top:.3f}; control line (best of 3 random 16-sets) {line_top:.3f}"
      f"  [{round(time.time() - t0)} s]", flush=True)

# ---------------------------------------------------------------- 3. define the parts from the board
items = L.items
probe = list(TR[:60]); test = list(TE[:600])
fens_p = [items[i]["fen"] for i in probe]; fens_t = [items[i]["fen"] for i in test]
d = vocab.get("chess")
cp, ct = vocab.featurize(fens_p), vocab.featurize(fens_t)


def define_total(f):
    out, _ = best_definitions(d, cp, T[probe, f], 5, "Pos", k=1, ctx_test=ct, target_test=T[test, f])
    return out[0]


ctrl_def = [define_total(int(f))["heldout_corr"] for f in rng.choice(others, 3, replace=False)]
line_def = max(ctrl_def)
defs = {}
for f in S_top:
    defs[f] = define_total(f)
    print(f"part f{f}: best board definition {defs[f]['definition']} held-out corr {defs[f]['heldout_corr']} "
          f"(line {line_def:.3f}) -> {'defined' if defs[f]['heldout_corr'] > line_def else 'unresolved'}  [{round(time.time() - t0)} s]", flush=True)

# ---------------------------------------------------------------- 4. certify the top level by interventions
m = L.get_net()
pairs = rng.choice(TE, (240, 2))
pairs = pairs[pairs[:, 0] != pairs[:, 1]]
A_rows, B_rows = pairs[:, 0], pairs[:, 1]
v_a = L.value(m, A_rows)


def swap_change(f):
    za, zb = F.fmap(f, A_rows), F.fmap(f, B_rows)
    v = L.value(m, A_rows, edit=F.edit_feature(f, za, zb))
    return v - v_a


def predicted_change(f):
    i = list(S_top).index(f)
    return w_top[i] * (T[B_rows, f] - T[A_rows, f])


scores = {}
for f in S_top:
    pc, ac = predicted_change(f), swap_change(f)
    scores[f] = float(np.corrcoef(pc, ac)[0, 1]) if pc.std() > 0 and ac.std() > 0 else 0.0
best_ctrl = []
for _ in range(20):
    sc = []
    for f in S_top:
        g = int(rng.choice(others))
        ac = swap_change(g)
        pc = predicted_change(f)
        sc.append(float(np.corrcoef(pc, ac)[0, 1]) if pc.std() > 0 and ac.std() > 0 else 0.0)
    best_ctrl.append(max(sc))
line_iia = float(np.quantile(best_ctrl, 0.99))
for f in S_top:
    print(f"intervention f{f}: corr(predicted, actual value change) {scores[f]:.3f} against line {line_iia:.3f} -> "
          f"{'pass' if scores[f] > line_iia else 'fail'}", flush=True)

res = {"top_level": {"parts": [int(f) for f in S_top], "weights": [float(x) for x in w_top], "heldout_r2": s_top,
                     "control_line": line_top, "controls": ctrl},
       "parts": {int(f): {"definition": defs[f]["definition"], "heldout_corr": defs[f]["heldout_corr"],
                          "status": "defined" if defs[f]["heldout_corr"] > line_def else "unresolved",
                          "intervention_corr": scores[f], "intervention_pass": scores[f] > line_iia} for f in S_top},
       "definition_line": line_def, "definition_controls": ctrl_def, "intervention_line": line_iia,
       "seconds": round(time.time() - t0)}
json.dump(res, open("results/build/leela_box_session2.json", "w"), indent=1)
print("done", round(time.time() - t0), "s")
