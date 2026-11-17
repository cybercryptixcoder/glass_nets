"""Is this LM task a good target for the constructor? Run after building a task:  python3 lab/lm_scout.py <target>

Checks (each one cost a session a detour before it existed; see docs/EXPERIMENT_LOG.md E13, E18, E19):
  behaviour   how the network's decisions are spread over the answers; a task where one answer wins almost
              everywhere has little to explain.
  product     whether the input rows are every combination of the features' values. If they are, the judge's
              interchange and partial tests (which combine the inputs of two prompts) always land on another prompt
              of the task, and the reader can look up what the network does there. If not, those tests ask about
              inputs the reader has never seen (this stalled the countdown runs).
  lookup      the best held-out accuracy of a linear read-out of the network's decision from ONE token's embedding.
              Near 100% means the answer is a feature of a single token: nothing is computed along the way, and a
              construction that knows nothing about the network would pass (the ordinal-suffix trap, E13).
  positions   which positions the decision depends on: the share of decisions kept when one position's embedding is
              replaced by its mean. A good target depends on several.
Writes lab/models/<target>/scout.json.
"""
import json, sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import lm  # noqa: E402


def scout(tid, seed=0):
    rng = np.random.default_rng(seed)
    t = lm.get_task(tid)
    out = t.run(np.arange(t.N))
    dec = out.argmax(1)
    k = out.shape[1]
    share = np.bincount(dec, minlength=k) / t.N
    rep = {"target": tid, "n_prompts": int(t.N), "n_answers": int(k), "behaviour": {"decision_shares": share.round(4).tolist(),
           "top_share": float(share.max()), "effective_answers": float(np.exp(-(share[share > 0] * np.log(share[share > 0])).sum()))}}
    uniq = [len(np.unique(t.X[:, j])) for j in range(t.X.shape[1])]
    rep["product"] = {"values_per_feature": dict(zip(t.meta["x_names"], uniq)), "combinations": int(np.prod(uniq, dtype=float)),
                      "coverage": float(t.N / np.prod(uniq, dtype=float))}
    perm = rng.permutation(t.N)
    fit, test = perm[: t.N // 2], perm[t.N // 2:]
    positions = sorted({int(v) for p in t.roles.values() for v in np.unique(p)} | set(range(1, t.T)))
    name = {}
    for r, p in t.roles.items():
        if len(set(p.tolist())) == 1:
            name.setdefault(int(p[0]), r)
    E = t.resid(0, np.arange(t.N))                       # [N, T, d]: embeddings
    Y = np.eye(k)[dec]
    look, keep = {}, {}
    for p in positions:
        A = E[:, p]
        c = A[fit].mean(0)
        _, sv, Vt = np.linalg.svd(A[fit] - c, full_matrices=False)
        r = max(1, min(int((sv > 1e-6 * max(sv[0], 1e-12)).sum()), 64))
        Z = (A - c) @ Vt[:r].T
        W, *_ = np.linalg.lstsq(np.c_[Z[fit], np.ones(len(fit))], Y[fit], rcond=None)
        look[name.get(p, str(p))] = float(((np.c_[Z[test], np.ones(len(test))] @ W).argmax(1) == dec[test]).mean())
        o = t.run(test, hook={f"resid0@{p}": np.broadcast_to(A[fit].mean(0), (len(test), A.shape[1]))})
        keep[name.get(p, str(p))] = float((o.argmax(1) == dec[test]).mean())
    rep["lookup"] = {"by_position": look, "best": max(look.values()), "best_position": max(look, key=look.get)}
    rep["positions"] = {"decisions_kept_when_mean_ablated": keep, "needed": [p for p, v in keep.items() if v < 0.9]}
    warn = []
    if share.max() > 0.9:
        warn.append(f"one answer wins on {share.max():.0%} of prompts: little to explain")
    if rep["product"]["coverage"] < 0.95:
        warn.append(f"the inputs cover {rep['product']['coverage']:.0%} of the combinations of the features' values: the judge's partial and "
                    "interchange tests will ask about inputs outside the task")
    if rep["lookup"]["best"] > 0.95:
        warn.append(f"the decision is read off the embedding of one token ({rep['lookup']['best_position']}: {rep['lookup']['best']:.0%}): "
                    "probably a lookup, nothing computed")
    if len(rep["positions"]["needed"]) < 2:
        warn.append("the decision depends on fewer than two positions")
    rep["warnings"] = warn
    rep["verdict"] = "good target" if not warn else "weak target: " + "; ".join(warn)
    (t.dir / "scout.json").write_text(json.dumps(rep, indent=1))
    return rep


def render(rep):
    b, p, l, q = rep["behaviour"], rep["product"], rep["lookup"], rep["positions"]
    return "\n".join([
        f"SCOUT {rep['target']}: {rep['verdict']}",
        f"  behaviour: decision shares {b['decision_shares']} (effective number of answers {b['effective_answers']:.1f})",
        f"  product: {p['values_per_feature']} -> {p['combinations']} combinations, the task has {rep['n_prompts']} ({p['coverage']:.0%})",
        "  lookup (decision read off one token's embedding, held out): " + ", ".join(f"{k} {v:.0%}" for k, v in l["by_position"].items()),
        "  decisions kept when one position's embedding is mean-ablated: " + ", ".join(f"{k} {v:.0%}" for k, v in q["decisions_kept_when_mean_ablated"].items())])


if __name__ == "__main__":
    print(render(scout(sys.argv[1])))
