"""Unit map (docs/glass/UNITS_PROGRAM.md §4b): label every RNN neuron at every step with its best unit from a fixed
library, with per-part confidence (held-out fit, single-unit sufficiency, ablation importance), coverage against a
shuffled-weights control, and a per-neuron summary label.

    python3 lab/unitmap.py TARGET [TARGET ...]      -> results/glass/units/maps/<target>.{json,md}
"""
import itertools, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

LAB = Path(__file__).resolve().parent; ROOT = LAB.parent
sys.path.insert(0, str(LAB))
import judge, nets  # noqa: E402

OUT = ROOT / "results" / "glass" / "units" / "maps"
RAW = {"x_t", "x_prev"}


def library(X, t):
    """The unit library's base quantities of the history x_1..x_t (t is 1-based)."""
    H = X[:, :t]; s = np.sort(H, 1)
    q = {"x_t": X[:, t - 1], "max": s[:, -1], "2nd": s[:, -2] if t > 1 else s[:, -1], "min": s[:, 0]}
    if t > 1:
        sp = np.sort(X[:, :t - 1], 1)
        q.update({"x_prev": X[:, t - 2], "max_prev": sp[:, -1], "2nd_prev": sp[:, -2] if t > 2 else sp[:, -1]})
    return q


def fit(h, F):
    """h ~ relu(F @ w), no bias (the targets have none): least squares on the active set, refined twice."""
    act = h > 1e-9
    if act.sum() < 20:
        return None
    w = np.linalg.lstsq(F[act], h[act], rcond=None)[0]
    for _ in range(2):
        use = act | (F @ w > 0)
        w = np.linalg.lstsq(F[use], h[use], rcond=None)[0]
    return w


def r2(h, p):
    v = h.var()
    return float(1 - ((h - p) ** 2).mean() / v) if v > 0 else 0.0


def label(prims, w):
    if prims and all(p.startswith("x_") and p[2:].isdigit() for p in prims):
        return "filter: relu(" + " ".join(f"{c:+.2f}*{p}" for p, c in zip(prims, w)) + ")"
    if set(prims) <= RAW:
        return "raw input: relu(" + " ".join(f"{c:+.2f}*{p}" for p, c in zip(prims, w)) + ")"
    if len(prims) == 2 and "x_t" in prims:
        o = [p for p in prims if p != "x_t"][0]; a = w[prims.index(o)]; b = w[prims.index("x_t")]
        if a * b < 0 and 0.5 < abs(a / b) < 2:
            return f"comparator: relu(x_t - {o})" if b > 0 else f"comparator: relu({o} - x_t)"
    return "relu(" + " ".join(f"{c:+.2f}*{p}" for p, c in zip(prims, w)) + ")"


def kind(lab):
    return lab.split(":")[0] if ":" in lab else "other history unit"


def run(W, X, rep=None):
    s = np.zeros((len(X), W["W_h"].shape[0]))
    for t in range(1, X.shape[1] + 1):
        s = np.maximum(X[:, t - 1:t] * W["W_x"] + s @ W["W_h"], 0)
        for (tt, j), fn in (rep or {}).items():
            if tt == t:
                s[:, j] = fn(X)
    return s @ W["W_o"]


def unit_map(target, variant="original", n=8000, confidence=True):
    meta, W, X = judge.load_target(target, variant, 0)
    assert meta["forward"] == "rnn_relu", "unit maps are defined for ReLU RNNs (UNITS_PROGRAM §4b)"
    X = X[:n]; Xf, Xt = X[: n // 2], X[n // 2:]
    af, at = {}, {}
    nets.forward("rnn_relu", W, Xf, acts=af); nets.forward("rnn_relu", W, Xt, acts=at)
    ref = run(W, Xt); cells = []
    for t in range(1, X.shape[1] + 1):
        Hf = np.asarray(af[f"h{t}"]).reshape(len(Xf), -1); Ht = np.asarray(at[f"h{t}"]).reshape(len(Xt), -1)
        Lf, Lt = library(Xf, t), library(Xt, t)
        for j in range(Hf.shape[1]):
            if Hf[:, j].var() < 1e-10:
                continue
            best = (-9.0, None, None)
            for k in (1, 2):
                for combo in itertools.combinations(list(Lf), k):
                    w = fit(Hf[:, j], np.stack([Lf[c] for c in combo], 1))
                    if w is None:
                        continue
                    sc = r2(Ht[:, j], np.maximum(np.stack([Lt[c] for c in combo], 1) @ w, 0))   # held-out
                    better = sc > best[0] + (0.02 if k == 2 else 0)
                    tie_raw = set(combo) <= RAW and best[1] is not None and not set(best[1]) <= RAW and sc >= best[0] - 0.02
                    if better or tie_raw:
                        best = (sc, list(combo), w)
            # filter unit (T3): relu of a linear mix of ALL past inputs; preferred over history units on a tie (within 0.02)
            wf = fit(Hf[:, j], Xf[:, :t])
            if wf is not None:
                fsc = r2(Ht[:, j], np.maximum(Xt[:, :t] @ wf, 0))
                raw_best = best[1] is not None and set(best[1]) <= RAW
                if fsc >= best[0] - 0.02 and not (raw_best and best[0] >= fsc - 0.02):
                    best = (fsc, [f"x_{k + 1}" for k in range(t)], wf)
            sc, prims, w = best
            cell = {"t": t, "unit": j, "var": float(Ht[:, j].var()), "fit": sc, "prims": prims,
                    "w": None if w is None else [float(x) for x in w], "label": label(prims, list(w)) if prims else "none"}
            if confidence and prims:
                if prims[0].startswith("x_") and prims[0][2:].isdigit():
                    fn = lambda X_, t=t, w=w: np.maximum(X_[:, :t] @ w, 0)  # noqa: E731
                else:
                    fn = lambda X_, t=t, p=prims, w=w: np.maximum(np.stack([library(X_, t)[c] for c in p], 1) @ w, 0)  # noqa: E731
                mu = float(Hf[:, j].mean())
                cell["sufficiency"] = float(judge.same(ref, run(W, Xt, {(t, j): fn})).mean())
                cell["importance"] = float(1 - judge.same(ref, run(W, Xt, {(t, j): (lambda X_, mu=mu: np.full(len(X_), mu))})).mean())
            if prims:
                pr = (np.maximum(Xt[:, :t] @ np.array(w), 0) if prims[0].startswith("x_") and prims[0][2:].isdigit()
                      else np.maximum(np.stack([Lt[c] for c in prims], 1) @ np.array(w), 0))
                cell["rel_error"] = float(np.sqrt(((Ht[:, j] - pr) ** 2).mean()) / (Ht[:, j].std() + 1e-12))
            cells.append(cell)
    later = [c for c in cells if c["t"] > 1]; tot = sum(c["var"] for c in later) or 1.0
    is_raw = lambda c: set(c["prims"]) <= RAW or c["label"].startswith("filter")  # noqa: E731
    hist = [c for c in later if c["fit"] >= 0.9 and not is_raw(c)]
    raw = [c for c in later if c["fit"] >= 0.9 and is_raw(c)]
    per_unit = {}
    for j in sorted({c["unit"] for c in cells}):
        good = [c for c in cells if c["unit"] == j and c["t"] > 1 and c["fit"] >= 0.9]
        per_unit[j] = Counter(c["label"] for c in good).most_common(1)[0][0] if good else "unexplained"
    e95 = eps95(W, X) if confidence else None
    for c in cells:
        if e95 is not None and "rel_error" in c:
            c["substitution_grade"] = bool(c["rel_error"] <= e95)
    return {"target": target, "variant": variant, "cells": cells, "per_unit": per_unit, "eps95": e95,
            "coverage_history": sum(c["var"] for c in hist) / tot, "coverage_raw": sum(c["var"] for c in raw) / tot}


def eps95(W, X, n=2000):
    """Precision requirement (UNITS_PROGRAM P3): relative hidden-state noise at which decision agreement falls to 95%."""
    X = X[:n]; acts = {}; nets.forward("rnn_relu", W, X, acts=acts)
    sd = {t: np.asarray(acts[f"h{t}"]).reshape(len(X), -1).std(0) for t in range(1, X.shape[1] + 1)}
    def noisy(eps, rng):
        s = np.zeros((len(X), W["W_h"].shape[0]))
        for t in range(1, X.shape[1] + 1):
            s = np.maximum(X[:, t - 1:t] * W["W_x"] + s @ W["W_h"], 0)
            if eps > 0:
                s = s + rng.normal(size=s.shape) * eps * sd[t]
        return s @ W["W_o"]
    ref = noisy(0, None)
    for e in np.geomspace(0.001, 1.0, 31):
        if np.mean([judge.same(ref, noisy(e, np.random.default_rng(k))).mean() for k in range(3)]) < 0.95:
            return float(e)
    return 1.0


def render(m, ctrl):
    L = [f"# Unit map: {m['target']}", "",
         f"Coverage after the first step (variance-weighted share of neuron-steps fitted at R² ≥ 0.9 on held-out inputs): "
         f"history units {m['coverage_history']:.2f}, raw-input units {m['coverage_raw']:.2f}. "
         f"Shuffled-weights control: history {ctrl['coverage_history']:.2f}, raw {ctrl['coverage_raw']:.2f}.", "",
         f"Precision requirement ε95 = {m.get('eps95')}: hidden-state noise (as a share of each neuron's spread) at which 5% of decisions flip. "
         f"Substitution-grade units (relative error ≤ ε95): {sum(1 for c in m['cells'] if c.get('substitution_grade'))} of {len(m['cells'])} neuron-steps.", "",
         "## Per neuron (most common label over its explained steps)", "", "| neuron | label |", "|---|---|"]
    L += [f"| {j} | {lab} |" for j, lab in m["per_unit"].items()]
    L += ["", "## Every neuron-step", "", "| step | neuron | unit | fit (held-out R²) | relative error | sufficiency | importance |", "|---|---|---|---|---|---|---|"]
    for c in m["cells"]:
        L.append(f"| {c['t']} | {c['unit']} | {c['label']} | {c['fit']:.3f} | {c.get('rel_error', float('nan')):.3f} | {100 * c.get('sufficiency', float('nan')):.1f}% | {100 * c.get('importance', float('nan')):.1f}% |")
    L += ["", "*Sufficiency: decisions kept when only this neuron-step is replaced by its unit formula. Importance: decisions lost when it is set to its mean. "
          "Both are on held-out inputs. Units are approximate in dense networks (UNITS_PROGRAM, first-sweep conclusion), so a high fit does not imply exact sufficiency.*"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for tgt in sys.argv[1:]:
        m = unit_map(tgt); c = unit_map(tgt, "shuffled", confidence=False)
        (OUT / f"{tgt}.json").write_text(json.dumps({"map": m, "control": {k: c[k] for k in ("coverage_history", "coverage_raw", "per_unit")}}, indent=0))
        (OUT / f"{tgt}.md").write_text(render(m, c))
        print(f"{tgt}: eps95 {m['eps95']:.4f} | substitution-grade {sum(1 for x in m['cells'] if x.get('substitution_grade'))}/{len(m['cells'])} | history coverage {m['coverage_history']:.2f} (control {c['coverage_history']:.2f}), raw {m['coverage_raw']:.2f} | per neuron: "
              + "; ".join(f"{j}:{lab.split('(')[0].strip()}{'(' + lab.split('(')[1] if lab.startswith('comparator') else ''}" for j, lab in m["per_unit"].items()), flush=True)
