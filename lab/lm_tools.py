"""Ready-made experiments for the reader's session on LM targets (bound in kernel.py, described in LM_SYSTEM).

flow(feature)   WHERE does one input feature live? Pairs of prompts that differ ONLY in that feature (A and B); for
                every depth L and position p the network is run on B with resid{L}@p taken from A. The table gives the
                share of pairs whose answer moves to A's answer (among pairs where A and B have different answers).
                Read down a column to see the feature arrive at a position and leave it.
flow()          without a feature: the share of answers KEPT when resid{L}@p is replaced by its mean (what matters where).
probe(site, values)   held-out R2 (and accuracy, for integer values) of a linear read-out of `values` from a site.
"""
import numpy as np

import lm


def _pairs(task, j, n, rng):
    """(B, A): prompt numbers whose inputs differ only in column j. Needs the task's inputs to be (nearly) a full product."""
    index = task._index
    B, A = [], []
    vals = np.unique(task.X[:, j])
    for b in rng.permutation(task.N):
        row = task.X[b].copy()
        for v in rng.permutation(vals):
            if v == row[j]:
                continue
            row2 = row.copy(); row2[j] = v
            a = index.get(tuple(row2.tolist()))
            if a is not None:
                B.append(b); A.append(a)
                break
        if len(B) >= n:
            break
    return np.asarray(B, dtype=np.int64), np.asarray(A, dtype=np.int64)


def flow(task, feature=None, n=300, positions=None, layers=None, seed=0):
    rng = np.random.default_rng(seed)
    nL, T = task.cfg["n_layers"], task.T
    layers = list(range(nL + 1)) if layers is None else list(layers)
    positions = list(range(1, T)) if positions is None else [int(task.pos(p, np.arange(1))[0]) if isinstance(p, str) else int(p) for p in positions]
    names = {}
    for r, p in task.roles.items():
        if len(set(p.tolist())) == 1:
            names.setdefault(int(p[0]), r)
    out = task.run(np.arange(task.N))
    dec = out.argmax(1)
    head = "depth  " + " ".join(f"{names.get(p, str(p))[:8]:>8s}" for p in positions)
    if feature is None:
        idx = rng.permutation(task.N)[:n]
        rows = []
        for L in layers:
            R = task.resid(L, np.arange(task.N))
            cells = []
            for p in positions:
                o = task.run(idx, hook={f"resid{L}@{p}": np.broadcast_to(R[:, p].mean(0), (len(idx), R.shape[2]))})
                cells.append((o.argmax(1) == dec[idx]).mean())
            rows.append(f"resid{L:<2d}" + " ".join(f"{c:8.0%}" for c in cells))
        return "Share of answers KEPT when resid{L}@p is replaced by its mean over the prompts (low = needed there)\n" + head + "\n" + "\n".join(rows)
    j = task.meta["x_names"].index(feature) if isinstance(feature, str) else int(feature)
    B, A = _pairs(task, j, n * 3, rng)
    diff = dec[B] != dec[A]
    B, A = B[diff][:n], A[diff][:n]
    if len(B) < 10:
        return f"Only {len(B)} pairs of prompts differ only in {feature} AND get different answers: nothing to measure."
    rows = []
    for L in layers:
        R = task.resid(L, np.arange(task.N))
        cells = []
        for p in positions:
            o = task.run(B, hook={f"resid{L}@{p}": R[A, p]})
            cells.append((o.argmax(1) == dec[A]).mean())
        rows.append(f"resid{L:<2d}" + " ".join(f"{c:8.0%}" for c in cells))
    return (f"Where does {task.meta['x_names'][j]} live? {len(B)} pairs (A, B) that differ only in {task.meta['x_names'][j]} and get different answers. "
            "Cell = share of pairs whose answer becomes A's when the network runs on B with resid{L}@p taken from A.\n" + head + "\n" + "\n".join(rows))


def probe(task, site, values, n=None, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    idx = np.arange(task.N) if n is None else rng.permutation(task.N)[:n]
    A = task.acts(idx, [site])[site]
    V = values[idx] if len(values) == task.N else values
    cat = V.ndim == 1 and np.issubdtype(V.dtype, np.integer)
    Y = np.eye(int(V.max()) + 1)[V] if cat else (V[:, None] if V.ndim == 1 else V).astype(float)
    perm = rng.permutation(len(idx)); fit, test = perm[: len(idx) // 2], perm[len(idx) // 2:]
    c = A[fit].mean(0)
    _, sv, Vt = np.linalg.svd(A[fit] - c, full_matrices=False)
    r = max(1, min(int((sv > 1e-6 * max(sv[0], 1e-12)).sum()), max(8, len(fit) // 4)))
    Z = (A - c) @ Vt[:r].T
    W, *_ = np.linalg.lstsq(np.c_[Z[fit], np.ones(len(fit))], Y[fit], rcond=None)
    P = np.c_[Z[test], np.ones(len(test))] @ W
    r2 = 1 - ((P - Y[test]) ** 2).sum() / max(((Y[test] - Y[fit].mean(0)) ** 2).sum(), 1e-12)
    res = {"site": site, "heldout_r2": round(float(r2), 4)}
    if cat:
        res["heldout_accuracy"] = round(float((P.argmax(1) == V[test]).mean()), 4)
    return res


def explain_output(task, n=None, seed=0):
    """WHAT does the decision depend on, and in what form? The network's margin (top answer minus the second, by
    column: logit[k] - logit[0] for each k > 0) is fitted on half the prompts by simple models of the input features and
    scored on the other half: each feature alone (one value per feature value), all features additively, and for every
    pair of features with the same set of values their ORDER (a < b, a == b) and their CIRCULAR difference (b - a mod
    the number of values), alone and added to the additive model; finally the full table of the first two features.
    The model that explains most with fewest parameters is a hypothesis about how the network compares its inputs."""
    import itertools
    import lmjudge
    rng = np.random.default_rng(seed)
    out = task.run(np.arange(task.N))
    Y = out[:, 1:] - out[:, :1]
    clear = lmjudge.confident(out)
    dec = out.argmax(1)
    X, names = task.X, task.meta["x_names"]
    perm = rng.permutation(task.N); f, t = perm[: task.N // 2], perm[task.N // 2:]
    one = np.ones((task.N, 1))
    oh = lambda v: (v[:, None] == np.unique(v)[None]).astype(float)   # noqa: E731
    models = {"constant": one}
    for j, nm in enumerate(names):
        models[f"{nm} alone"] = oh(X[:, j])
    add = np.concatenate([oh(X[:, j]) for j in range(len(names))], 1)
    models["all features, additive"] = add
    for i, j in itertools.combinations(range(len(names)), 2):
        vi, vj = np.unique(X[:, i]), np.unique(X[:, j])
        if len(vi) == len(vj) and (vi == vj).all() and len(vi) > 2:
            a, b = X[:, i], X[:, j]
            order = np.c_[one, b < a, b == a].astype(float)
            circ = oh((b - a) % len(vi))
            models[f"order of {names[j]} vs {names[i]} ({names[j]} < / = / >)"] = order
            models[f"circular difference {names[j]} - {names[i]} (mod {len(vi)})"] = circ
            models[f"order + additive"] = np.c_[order, add]
            models[f"circular difference + additive"] = np.c_[circ, add]
    if len(names) >= 2:
        models[f"full table ({names[0]}, {names[1]})"] = oh(X[:, 0] * 10000 + X[:, 1])
    lines = ["What the decision margin depends on (fitted on half the prompts, scored on the other half; clear decisions only for 'right'):",
             f"{'model':58s} {'params':>6s} {'margin R2':>9s} {'right':>7s}"]
    for k, F in models.items():
        W, *_ = np.linalg.lstsq(F[f], Y[f], rcond=None)
        P = F @ W
        r2 = 1 - ((P[t] - Y[t]) ** 2).sum() / max(((Y[t] - Y[t].mean(0)) ** 2).sum(), 1e-12)
        pd = np.where(P.max(1) > 0, P.argmax(1) + 1, 0)
        acc = (pd[t] == dec[t])[clear[t]].mean()
        lines.append(f"{k[:58]:58s} {F.shape[1]:6d} {r2:9.3f} {acc:7.1%}")
    return "\n".join(lines)
