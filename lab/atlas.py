"""Mechanism atlas v1 (CHANGES A1; intent and certificate in docs/glass/ATLAS.md §2).

The route for fragile networks, where no short approximate account can be faithful:
  1. decompose the weights EXACTLY into rank-1 mechanisms (SPD, lab/spd.py);
  2. give each mechanism a readable TRIGGER, a shallow rule over task features and the network's own neuron states,
     whose held-out balanced accuracy at predicting "the mechanism is needed" (importance > 0.5) is its confidence;
  3. certify the whole: the network rebuilt from its alive mechanisms, each switched on by its trigger and run on its
     OWN state, must reproduce the network's decisions (rule-gated fidelity), reported next to the task rule.

Targets are one-layer ReLU RNNs: h_t = relu(W_x x_t + W_h h_(t-1)), out = W_o h_N (AlgZoo, or the planted known answer).
Transformers come in a later version (ATLAS.md, A5).

    a = build_atlas(algzoo_rnn("zoo_2nd_argmax_8_5"), spd_model=PATH or None, depth=5)
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier, export_text

import spd

LAB = Path(__file__).resolve().parent
MATRICES = ("W_x", "W_h", "W_o")
TRUSTED = 0.9      # a trigger is used to switch its mechanism only at held-out confidence >= 0.9 (the labelling bar used
                   # throughout the units program, U19(c) (iii)); below it the mechanism stays always on and is reported so


@dataclass
class RNNTarget:
    name: str
    N: int                                   # sequence length
    H: int                                   # hidden size
    d_in: int                                # input size per step
    weights: dict                            # {W_x: [H, d_in], W_h: [H, H], W_o: [d_out, H]} as [d_out, d_in] tensors
    sample: Callable                         # n -> torch [n, N, d_in]
    X: np.ndarray                            # domain inputs [n, N, d_in]
    task_rule: float | None                  # agreement of the task's own answer with the network (its accuracy)
    task_features: Callable                  # (x [B, N, d_in], t) -> (names, [B, F]) for the step that reads x_t
    out_scale: float = 1.0
    meta: dict = field(default_factory=dict)

    def forward(self, x, mm):
        h = torch.zeros(x.shape[0], self.H, dtype=x.dtype)
        for t in range(self.N):
            h = torch.relu(mm("W_x", x[:, t]) + mm("W_h", h))
        return mm("W_o", h) / self.out_scale


# ------------------------------------------------------------------ targets

def _scalar_seq_features(x, t):
    """Task features of step t (1-based) for scalar-sequence tasks (2nd argmax, argmedian)."""
    s = x[:, :, 0]; B = len(s); prev = s[:, :t - 1]; cur = s[:, t - 1]; upto = s[:, :t]
    mx_p = prev.max(1) if t > 1 else np.full(B, np.nan); s2_p = np.sort(prev, 1)[:, -2] if t > 2 else np.full(B, np.nan)
    mn_p = prev.min(1) if t > 1 else np.full(B, np.nan)
    rank = (upto > cur[:, None]).sum(1) + 1
    amax = upto.argmax(1); a2 = np.argsort(upto, 1)[:, -2] if t > 1 else np.zeros(B, int); amin = upto.argmin(1)
    f = {"step t": np.full(B, t), "x_t": cur, "rank of x_t so far (1 = largest)": rank,
         "x_t minus the previous max": np.nan_to_num(cur - mx_p), "x_t minus the previous 2nd": np.nan_to_num(cur - s2_p),
         "x_t minus the previous min": np.nan_to_num(cur - mn_p),
         "steps since the max": t - 1 - amax, "steps since the 2nd": np.where(t > 1, t - 1 - a2, -1), "steps since the min": t - 1 - amin}
    return list(f), np.stack(list(f.values()), 1).astype(float)


def algzoo_rnn(target_id, variant="original", n=16000):
    import judge
    meta, W, X = judge.load_target(target_id, variant, 0)
    N = int(W["seq_len"][0]); H = W["W_h"].shape[0]
    Ws = {"W_x": torch.tensor(W["W_x"].T.copy(), dtype=torch.float64), "W_h": torch.tensor(W["W_h"].T.copy(), dtype=torch.float64),
          "W_o": torch.tensor(W["W_o"].T.copy(), dtype=torch.float64)}
    Xn = np.asarray(X[:n], dtype=np.float64)[:, :, None]
    with torch.no_grad():
        h = torch.zeros(len(Xn), H, dtype=torch.float64); xt = torch.tensor(Xn[:8000])
        h = torch.zeros(len(xt), H, dtype=torch.float64)
        for t in range(N):
            h = torch.relu(xt[:, t] @ Ws["W_x"].T + h @ Ws["W_h"].T)
        scale = float((h @ Ws["W_o"].T).std())
    return RNNTarget(name=f"{target_id}|{variant}", N=N, H=H, d_in=1, weights=Ws,
                     sample=lambda k: torch.randn(k, N, 1, dtype=torch.float64), X=Xn,
                     task_rule=float(meta["test_acc"]) if variant == "original" else None,
                     task_features=_scalar_seq_features, out_scale=scale, meta={"target": target_id, "variant": variant})


def planted_rnn(K=5, N=6, p_on=0.03, n=16000):
    """The U19(b') known answer: sum K sparse features over N steps; W_x = W_h = W_o = I."""
    I = torch.eye(K, dtype=torch.float64)

    def sample(k, g=None):
        on = (torch.rand(k, N, K, dtype=torch.float64, generator=g) < p_on).double()
        return on * torch.rand(k, N, K, dtype=torch.float64, generator=g)
    X = sample(n, torch.Generator().manual_seed(123)).numpy()

    def feats(x, t):
        names = [f"feature {j} arrives now (x_t[{j}] > 0)" for j in range(K)]
        return names, (x[:, t - 1] > 0).astype(float)
    return RNNTarget(name="planted_rnn", N=N, H=K, d_in=K, weights={"W_x": I, "W_h": I, "W_o": I}, sample=sample, X=X,
                     task_rule=None, task_features=feats, meta={"target": "planted_rnn"})


# ------------------------------------------------------------------ decomposition

def decompose(tgt, steps=30000, batch=512, seed=0, C=20, log=1000):
    return spd.decompose(tgt.forward, tgt.weights, tgt.sample, C=C, steps=steps, coef_min=3e-3, p=1.0,
                         coef_faith=1.0, lr=1e-3, batch=batch, seed=seed, log=log)


def load_decomposition(tgt, path, C=20):
    d = torch.load(path, weights_only=False)
    mods = torch.nn.ModuleDict({k: spd.Decomposed(v, C).double() for k, v in tgt.weights.items()})
    mods.load_state_dict(d["mods"])
    return mods


# ------------------------------------------------------------------ features, states, gated runs

def _features(tgt, x, t, h, end=False):
    """Features for the matrices that act at step t (W_x reads x_t, W_h reads h = h_(t-1)); end=True: W_o reads h = h_N."""
    names, F = tgt.task_features(x, t)
    lab = "in the final state" if end else "in the state before this step"
    names = list(names) + [f"neuron {j} active {lab}" for j in range(tgt.H)]
    return names, np.concatenate([F, (h > 1e-9).astype(float)], 1)


def _apply(mods, name, a, mask):
    m = mods[name]
    return ((a @ m.V.detach().numpy().T) * mask) @ m.U.detach().numpy()


def run_gated(tgt, mods, x, gate):
    """The network rebuilt from mechanisms: gate(name, F_names, F, a) -> mask [B, C]; runs on its own state."""
    B = len(x); h = np.zeros((B, tgt.H))
    for t in range(1, tgt.N + 1):
        names, F = _features(tgt, x, t, h)
        h = np.maximum(_apply(mods, "W_x", x[:, t - 1], gate("W_x", names, F, x[:, t - 1]))
                       + _apply(mods, "W_h", h, gate("W_h", names, F, h)), 0)
    names, F = _features(tgt, x, tgt.N, h, end=True)
    return _apply(mods, "W_o", h, gate("W_o", names, F, h)) / tgt.out_scale


def _network_calls(tgt, mods, x):
    """Per matrix: the features at each call (network's own states) and the importances: {name: (names, F, G)}."""
    B = len(x); h = np.zeros((B, tgt.H)); Fx = []; Gx = []; Gh = []
    W = {k: v.numpy() for k, v in tgt.weights.items()}
    with torch.no_grad():
        ci = lambda k, a: mods[k].ci_pre(torch.tensor(a)).clamp(0, 1).numpy()   # noqa: E731
        for t in range(1, tgt.N + 1):
            names, F = _features(tgt, x, t, h); Fx.append(F); Gx.append(ci("W_x", x[:, t - 1])); Gh.append(ci("W_h", h))
            h = np.maximum(x[:, t - 1] @ W["W_x"].T + h @ W["W_h"].T, 0)
        names_o, Fo = _features(tgt, x, tgt.N, h, end=True); Go = ci("W_o", h)
    F = np.concatenate(Fx)
    return {"W_x": (names, F, np.concatenate(Gx)), "W_h": (names, F, np.concatenate(Gh)), "W_o": (names_o, Fo, Go)}


# ------------------------------------------------------------------ readable rules

def rule_text(tree, names):
    """The tree as a short readable rule: the conditions under which the mechanism is ON, with subtrees whose leaves
    all agree collapsed (sklearn keeps splits that don't change the class)."""
    t = tree.tree_
    cls = lambda i: bool(t.value[i][0].argmax() == list(tree.classes_).index(True)) if True in tree.classes_ else False  # noqa: E731

    def uniform(i):                     # None if mixed, else the class of every leaf below i
        if t.children_left[i] == -1:
            return cls(i)
        a, b = uniform(t.children_left[i]), uniform(t.children_right[i])
        return a if a is not None and a == b else None

    def paths(i, conds):
        u = uniform(i)
        if u is not None:
            return [conds] if u else []
        f, thr = names[t.feature[i]], t.threshold[i]
        binary = set(np.unique(tree._fit_values[:, t.feature[i]])) <= {0.0, 1.0} if hasattr(tree, "_fit_values") else False
        lo = f"not ({f})" if binary else f"{f} <= {thr:.2f}"
        hi = f"{f}" if binary else f"{f} > {thr:.2f}"
        return paths(t.children_left[i], conds + [lo]) + paths(t.children_right[i], conds + [hi])
    ps = paths(0, [])
    if not ps:
        return "never"
    if ps == [[]]:
        return "always"
    return "\n".join("ON when " + " AND ".join(p) for p in ps)


# ------------------------------------------------------------------ the atlas

def build_atlas(tgt, spd_model=None, depth=5, n=16000, steps=30000, batch=512, seed=0):
    import judge
    mods = load_decomposition(tgt, spd_model) if spd_model else decompose(tgt, steps, batch, seed)
    X = tgt.X[:n]; half = len(X) // 2; xtr, xte = X[:half], X[half:]
    tr, te = _network_calls(tgt, mods, xtr), _network_calls(tgt, mods, xte)
    C = mods["W_h"].U.shape[0]
    alive = {k: [c for c in range(C) if tr[k][2][:, c].mean() > 0.01] for k in MATRICES}
    atlas = {"target": tgt.meta, "N": tgt.N, "H": tgt.H, "task_rule": tgt.task_rule, "depth": depth, "matrices": {}}
    trees = {}
    for k in MATRICES:
        names, Ftr, Gtr = tr[k]; _, Fte, Gte = te[k]; mechs = []
        with torch.no_grad():
            faith = float((mods[k].W - mods[k].weight()).norm() / mods[k].W.norm())
            alive_err = float((mods[k].W - sum(torch.outer(mods[k].U[c], mods[k].V[c]) for c in alive[k])).norm() / mods[k].W.norm()) if alive[k] else 1.0
        for c in alive[k]:
            ytr, yte = Gtr[:, c] > 0.5, Gte[:, c] > 0.5
            if ytr.all() or (~ytr).all():
                const = bool(ytr.all()); trees[(k, c)] = const
                trig = {"rule": "always" if const else "never", "confidence": float((yte == const).mean())}
            else:
                tree = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", min_samples_leaf=20, random_state=0).fit(Ftr, ytr)
                tree._fit_values = Ftr
                p = tree.predict(Fte)
                conf = 0.5 * (p[yte].mean() + (~p[~yte]).mean()) if yte.any() and (~yte).any() else float((p == yte).mean())
                trees[(k, c)] = tree
                trig = {"rule": rule_text(tree, names), "confidence": float(conf), "tree": export_text(tree, feature_names=names, decimals=2)}
            m = mods[k]
            mechs.append({"id": f"{k}.{c}", "in_use": float(yte.mean()), "U": m.U[c].tolist(), "V": m.V[c].tolist(), "trigger": trig})
        atlas["matrices"][k] = {"mechanisms": mechs, "faithfulness_rel_error": faith, "alive_only_rel_error": alive_err}

    am = {k: np.isin(np.arange(C), alive[k]).astype(float) for k in MATRICES}
    with torch.no_grad():
        net = tgt.forward(torch.tensor(xte), lambda name, a: mods[name].apply(a)).numpy()

    conf = {(k, int(x["id"].split(".")[1])): x["trigger"]["confidence"] for k in MATRICES for x in atlas["matrices"][k]["mechanisms"]}

    def g_rules(only=None, trusted=None):
        """trusted = a confidence bar: a mechanism whose trigger is below it is kept always on (its trigger is not used)."""
        def gate(k, names, F, a):
            m = np.zeros((len(F), C))
            if only is not None and k not in only:
                return m + am[k]
            for c in alive[k]:
                t = trees[(k, c)]
                if trusted is not None and conf[(k, c)] < trusted:
                    m[:, c] = 1.0
                else:
                    m[:, c] = float(t) if isinstance(t, bool) else t.predict(F)
            return m
        return gate

    def g_ci(binary):
        def gate(k, names, F, a):
            with torch.no_grad():
                z = mods[k].ci_pre(torch.tensor(a)).clamp(0, 1).numpy()
            return ((z > 0.5).astype(float) if binary else z) * am[k]
        return gate
    fid = {}
    for label, gate in (("all_alive", lambda k, n_, F, a: np.repeat(am[k][None], len(F), 0)), ("importance_masked", g_ci(False)),
                        ("importance_binary", g_ci(True)), ("rule_gated", g_rules()), ("rule_gated_W_h_only", g_rules({"W_h"})),
                        ("rule_gated_trusted", g_rules(trusted=TRUSTED))):
        fid[label] = float(judge.same(net, run_gated(tgt, mods, xte, gate)).mean())
    atlas["fidelity"] = fid
    return atlas


def summary(a):
    lines = [f"{a['target']}: task rule {a['task_rule']}, fidelity " + ", ".join(f"{k} {v:.3f}" for k, v in a["fidelity"].items())]
    for k, m in a["matrices"].items():
        confs = [x["trigger"]["confidence"] for x in m["mechanisms"]]
        lines.append(f"  {k}: {len(confs)} mechanisms, trigger confidence " + (f"min {min(confs):.3f} median {np.median(confs):.3f}, "
                     f"{sum(c >= 0.9 for c in confs)}/{len(confs)} >= 0.9" if confs else "-") + f"; alive-only rel. error {m['alive_only_rel_error']:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    tid = sys.argv[1]; model = sys.argv[2] if len(sys.argv) > 2 else None; depth = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    a = build_atlas(planted_rnn() if tid == "planted_rnn" else algzoo_rnn(tid), spd_model=model, depth=depth)
    print(summary(a))
    out = LAB.parent / "results" / "glass" / "atlas"; out.mkdir(parents=True, exist_ok=True)
    (out / f"{tid}_d{depth}.json").write_text(json.dumps(a, indent=1))
