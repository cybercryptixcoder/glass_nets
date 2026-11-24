"""K4: rules synthesized from the chess network's own units, in quiet positions (no human chess ideas as a start).

    python3 scripts/kepler/chess_units.py

1. Quiet game positions; residual of LD2's value after the classical library + calculation features (out of fold).
2. LD2 channels (last site) whose summed activity predicts that residual on held-out positions.
3. Per channel: a depth-3 tree from per-square descriptors to the channel's activation on each square; compiled into
   a rule (the squares where the tree's strongest leaf holds), counted per position.
4. Each rule judged by the frozen K3 criteria on the full evaluation set, plus its gain inside quiet positions.
Must fail: the same synthesis on a random noise map shaped like a channel.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import chess  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

from chess_calc import features as calc_features  # noqa: E402
from chess_dark import Evaluator  # noqa: E402
from chess_known_test import load_chess, replication_q  # noqa: E402
from kepler.core.residual import gain, r2, surrogate  # noqa: E402
from kepler.targets.chess.squares import NAMES as DN, descriptors  # noqa: E402

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))
OUT = "results/kepler/chess/units"


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def calc_all(items):
    path = os.path.join(KDATA, "chess", "LD2", "calc_all.npz")
    if os.path.exists(path):
        return np.load(path)["C"]
    C = np.array([[f["qs_gain"], f["threat"], f["qs_mated"]] for f in
                  (calc_features(chess.Board(it["fen"])) for it in items)], dtype=np.float64)
    np.savez(path, C=C)
    return C


def leaf_rule(tree, names):
    """The path to the leaf with the highest mean value, as (conditions, value, n)."""
    t = tree.tree_
    best = None

    def walk(node, conds):
        nonlocal best
        if t.children_left[node] == -1:
            v, n = float(t.value[node][0][0]), int(t.n_node_samples[node])
            if best is None or v > best[1]:
                best = (list(conds), v, n)
            return
        f, th = names[t.feature[node]], float(t.threshold[node])
        walk(t.children_left[node], conds + [(f, "<=", th)])
        walk(t.children_right[node], conds + [(f, ">", th)])
    walk(0, [])
    return best


def rule_text(conds):
    parts = []
    for f, op, th in conds:
        if th in (0.5,) and op == ">":
            parts.append(f"{f}")
        elif th in (0.5,) and op == "<=":
            parts.append(f"not {f}")
        else:
            parts.append(f"{f} {op} {th:g}")
    return " and ".join(parts)


def rule_values(conds, D):
    """Count of squares satisfying every condition, for a stack of descriptor arrays D [n, 64, d]."""
    m = np.ones(D.shape[:2], dtype=bool)
    for f, op, th in conds:
        col = D[:, :, DN.index(f)]
        m &= (col <= th) if op == "<=" else (col > th)
    return m.sum(1).astype(np.float64)


def main():
    os.makedirs(OUT, exist_ok=True)
    items, X, names, groups = load_chess()
    q = np.array([it["q"] for it in items])
    q2 = replication_q(items)
    C = calc_all(items)
    log("calculation features ready")
    src = np.array([it["source"] for it in items])
    quiet = np.where((src == "game") & (X[:, names.index("in_check")] == 0) & (C[:, 0] == 0) & (C[:, 1] == 0))[0]
    L = np.column_stack([X, C])
    pred = np.zeros(len(quiet))
    for tr, te in GroupKFold(5).split(quiet, groups=groups[quiet]):
        pred[te] = surrogate(L[quiet[tr]], q[quiet[tr]]).predict(L[quiet[te]])
    r = q[quiet] - pred
    log(f"{len(quiet)} quiet positions; library+calculation R2 there {1 - np.var(r) / np.var(q[quiet]):.3f}")
    A = np.load(os.path.join(KDATA, "chess", "LD2", "acts_site10.npy"), mmap_mode="r")
    acts = np.asarray(A[quiet]).astype(np.float32).reshape(len(quiet), 128, 64)
    agg = acts.sum(2)
    half = len(quiet) // 2
    corr_a = np.array([np.corrcoef(agg[:half, c], r[:half])[0, 1] if agg[:half, c].std() > 0 else 0 for c in range(128)])
    corr_b = np.array([np.corrcoef(agg[half:, c], r[half:])[0, 1] if agg[half:, c].std() > 0 else 0 for c in range(128)])
    order = [c for c in np.argsort(-np.abs(corr_a)) if np.sign(corr_a[c]) == np.sign(corr_b[c])][:8]
    log("residual-predictive channels: " + ", ".join(f"{c} ({corr_a[c]:+.2f}/{corr_b[c]:+.2f})" for c in order))
    rng = np.random.default_rng(0)
    samp = rng.choice(len(quiet), min(1500, len(quiet)), replace=False)
    Dq = np.stack([descriptors(chess.Board(items[quiet[i]]["fen"])) for i in samp])
    log("descriptors ready")
    ev = Evaluator(items, X, names, q, q2, groups)
    Dev = np.stack([descriptors(b) for b in ev.boards])
    log("evaluation-set descriptors ready")
    results = []
    targets = [(f"channel_{c}", acts[samp, c, :]) for c in order]
    targets.append(("noise_control", rng.normal(size=(len(samp), 64)).astype(np.float32)))
    ntr = int(len(samp) * 0.75)
    for name, Y in targets:
        Xd = Dq.reshape(-1, len(DN))
        y = Y.reshape(-1)
        cut = ntr * 64
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=300, random_state=0).fit(Xd[:cut], y[:cut])
        fid = r2(y[cut:], tree.predict(Xd[cut:]))
        conds, val, n = leaf_rule(tree, DN)
        vals = rule_values(conds, Dev)
        rec = {"unit": name, "tree_fidelity_r2": fid, "rule": rule_text(conds), "conditions": conds,
               "leaf_mean": val, "leaf_n": n, "frequency": float(np.mean(vals > 0))}
        if np.std(vals) > 0:
            rec["test"] = ev.test(vals)
            # gain inside quiet positions of the evaluation set
            qmask = np.isin(ev.idx, quiet)
            idxq = ev.idx[qmask]
            vq = vals[qmask]
            ntq = int(qmask[:ev.ntr].sum())
            Lq = L[idxq]
            gq, _ = gain(Lq[:ntq], q[idxq][:ntq], Lq[ntq:], q[idxq][ntq:], vq[:ntq], vq[ntq:])
            rec["gain_quiet_over_library_plus_calc"] = gq
        results.append(rec)
        t = rec.get("test")
        log(f"{name}: tree R2 {fid:.3f} | rule: {rec['rule']} | on {rec['frequency']:.0%} of positions" +
            (f" | gain {t['gain_LD2']['gain']:+.4f} {np.round(t['gain_LD2']['ci95'], 4)} rep {t['gain_744706']['gain']:+.4f} "
             f"dec {t['decodability']['trained']:.2f}/{t['decodability']['control']:.2f} maxcorr {abs(t['max_corr_library'][1]):.2f} "
             f"quiet-gain {rec['gain_quiet_over_library_plus_calc']['gain']:+.4f} -> {'ACCEPTED' if t['accepted'] else 'no'}" if t else ""))
        json.dump(results, open(os.path.join(OUT, "rules.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
