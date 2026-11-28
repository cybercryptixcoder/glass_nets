"""The box, Code Zero: explain a small recurrent network as a blueprint.

Strategy (deliberately simple; everything here is meant to be improved):
1. Run the network on sample inputs.
2. Take its output scores (the logits, read from the final hidden state along the columns of W_o) as the variables.
3. Define each logit by the best-correlated expression of the vocabulary (up to 5 symbols), and locate that
   definition at the final hidden state by least squares.
4. The decision is the position of the largest logit, written as readout parts (pairwise comparisons).
"""
import time

import numpy as np

import boxkit as bk


def argmax_readout(ids, positions):
    """Readout parts computing the position (positions[j] for part ids[j]) of the largest of the parts `ids` (ties
    need no rule: the values are continuous)."""
    parts, k = [], len(ids)
    if k == 1:
        return [{"id": "decision", "type": "rule", "site": "readout", "definition": "(mul @%s 0)" % ids[0],
                 "upstream": [ids[0]]}]
    wins = []
    for j in range(k):
        prev = None
        for i in range(k):
            if i == j:
                continue
            g = f"g{j}_{i}"
            parts.append({"id": g, "type": "rule", "site": "readout", "definition": f"(gt @{ids[j]} @{ids[i]})",
                          "upstream": [ids[j], ids[i]]})
            if prev is None:
                prev = g
            else:
                w = f"w{j}_{i}"
                parts.append({"id": w, "type": "rule", "site": "readout", "definition": f"(mul @{prev} @{g})",
                              "upstream": [prev, g]})
                prev = w
        wins.append(prev)
    acc = None
    for j in range(k):
        if positions[j] == 0:
            continue
        t = f"t{j}"
        if acc is None:
            parts.append({"id": t, "type": "rule", "site": "readout", "definition": f"(mul @{wins[j]} {positions[j]})",
                          "upstream": [wins[j]]})
        else:
            parts.append({"id": t, "type": "rule", "site": "readout",
                          "definition": f"(add @{acc} (mul @{wins[j]} {positions[j]}))", "upstream": [acc, wins[j]]})
        acc = t
    parts.append({"id": "decision", "type": "rule", "site": "readout", "definition": f"(add @{acc} 0)",
                  "upstream": [acc]})
    return parts


def explain(net, budget_seconds):
    t0 = time.process_time()
    X = net.inputs(2000, seed=1)
    out, S = net.run(X)
    fit, test = np.arange(0, 1000), np.arange(1000, 2000)
    site = f"h{net.T}"
    probes = X[:200]
    parts, ids, pos = [], [], []
    Wo = net.weights["W_o"]
    for j in range(net.n_out):
        if time.process_time() - t0 > 0.8 * budget_seconds:
            break
        logit = out[:, j]
        cands = bk.best_definitions(probes, logit[:200], 5, "Num", k=3, X_test=X[test], target_test=logit[test])
        cands = [c for c in cands if c.get("heldout_corr") is not None]
        if not cands:
            continue
        best = max(cands, key=lambda c: abs(c["heldout_corr"]))
        definition = best["definition"] if best.get("scale", 1.0) >= 0 else f"(neg {best['definition']})"
        if not bk.admissible(definition)[0]:
            definition = best["definition"]
        values = bk.evaluate(definition, X)
        u, s = bk.fit_location(S, site, values, fit)
        pid = f"l{j}"
        parts.append({"id": pid, "type": "rule", "site": site, "definition": definition, "upstream": [],
                      "basis": u.tolist(), "scale": s})
        ids.append(pid)
        pos.append(j)
    if not ids:          # nothing found: return the logits' locations with no definition (unresolved)
        return {"parts": [{"id": f"l{j}", "type": "unresolved", "site": site,
                           "basis": (Wo[:, j] / np.linalg.norm(Wo[:, j])).tolist(), "scale": float(np.linalg.norm(Wo[:, j]))}
                          for j in range(net.n_out)], "output": "l0"}
    parts += argmax_readout(ids, pos)
    return {"parts": parts, "output": "decision"}
