"""Build step (3) prototype: the construction "network = SPD mechanisms, each switched on by a readable rule".
W_x and W_o use their alive components; each alive W_h component is switched on at a step when its rule (a shallow tree
over interpretable features of the inputs so far and of the construction's OWN hidden state) says so.
Reports decision fidelity vs the network for: continuous importances, binary importances, and rules (task features;
task + neuron-state features), against the task rule. argv: target [variant] [depth]."""
import sys, json
from pathlib import Path
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier, export_text
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(ROOT / "lab"))
from spd_core import importance
from explain_triggers import load, step_features
import judge

target = sys.argv[1] if len(sys.argv) > 1 else "zoo_2nd_argmax_8_5"
variant = sys.argv[2] if len(sys.argv) > 2 else "original"
depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
meta, W, X, N, H, mods, forward = load(target, variant, f"{target}_{variant}_30000_512")
X = np.asarray(X, dtype=np.float64)
xtr, xte = X[:8000], X[8000:16000] if len(X) >= 16000 else X[4000:8000]
g_tr = importance(mods, forward, torch.tensor(xtr)); names, Ftr = step_features(xtr, W, N, H)
alive = {k: [c for c in range(20) if g_tr[k][:, :, c].mean() > 0.01] for k in mods}
U = {k: mods[k].U.detach().numpy() for k in mods}; V = {k: mods[k].V.detach().numpy() for k in mods}
task_cols = [i for i, k in enumerate(names) if not k.startswith("neuron")]
trees = {}
for label, cols in (("task", task_cols), ("task+neurons", list(range(len(names))))):
    trees[label] = {}
    for c in alive["W_h"]:
        y = g_tr["W_h"][:, :, c].numpy().reshape(-1) > 0.5
        trees[label][c] = (DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=0).fit(Ftr[:, cols], y)
                           if 0 < y.mean() < 1 else bool(y.mean() >= 0.5)), cols


def lin(k, a, mask):           # apply matrix k's alive components with per-sample masks [B, C]
    return ((a @ V[k].T) * mask) @ U[k]


def run(x, gate):
    """gate(t, x, h) -> [B, C] mask for W_h at step t (h = the construction's state before the step)."""
    B = len(x); h = np.zeros((B, H))
    am = {k: np.isin(np.arange(20), alive[k]).astype(float)[None].repeat(B, 0) for k in mods}
    for t in range(1, N + 1):
        h = np.maximum(lin("W_x", x[:, t - 1:t], am["W_x"]) + lin("W_h", h, gate(t, x, h) * am["W_h"]), 0)
    return lin("W_o", h, am["W_o"])


def feats_at(t, x, h):
    """step_features for one step, from the construction's own state h."""
    B = len(x); prev = x[:, :t - 1]; cur = x[:, t - 1]; upto = x[:, :t]
    mx_p = prev.max(1) if t > 1 else np.full(B, -9.0); s2_p = np.sort(prev, 1)[:, -2] if t > 2 else np.full(B, -9.0)
    rank = (upto > cur[:, None]).sum(1) + 1; amax = upto.argmax(1); a2 = np.argsort(upto, 1)[:, -2] if t > 1 else np.zeros(B, int)
    cols = [np.full(B, t), cur, rank, np.where(t > 1, cur - mx_p, 0), np.where(t > 2, cur - s2_p, 0), t - 1 - amax, np.where(t > 1, t - 1 - a2, -1)]
    cols += [(h[:, j] > 1e-9).astype(float) for j in range(H)]
    return np.stack(cols, 1)


ref = judge.judge if False else None
with torch.no_grad():
    net = forward(torch.tensor(xte), lambda name, a: mods[name].apply(a)).numpy()
g_te = importance(mods, forward, torch.tensor(xte))["W_h"].numpy()                     # [N, B, C]


def gate_rules(label):
    def gate(t, x, h):
        F = feats_at(t, x, h); m = np.zeros((len(x), 20))
        for c, (tree, cols) in trees[label].items():
            m[:, c] = tree.predict(F[:, cols]) if not isinstance(tree, bool) else float(tree)
        return m
    return gate


res = {"target": target, "variant": variant, "depth": depth, "task_rule": float(meta["test_acc"]), "alive": {k: len(v) for k, v in alive.items()}}
for label, gate in (("continuous importances", lambda t, x, h: g_te[t - 1]), ("binary importances (> 0.5)", lambda t, x, h: (g_te[t - 1] > 0.5).astype(float)),
                    ("all alive mechanisms on", lambda t, x, h: np.ones((len(x), 20))),
                    ("rules: task features", gate_rules("task")), ("rules: task + neuron states", gate_rules("task+neurons"))):
    out = run(xte, gate); fid = float(judge.same(net, out).mean()); res[label] = fid
    print(f"{target} {variant} depth {depth} | {label:30s} fidelity {fid:.3f}  (task rule {res['task_rule']:.3f})", flush=True)
res["rules"] = {str(c): export_text(t, feature_names=[names[i] for i in cols], decimals=2) for c, (t, cols) in trees["task+neurons"].items() if not isinstance(t, bool)}
(ROOT / f"results/glass/units/rule_construction_{target}_{variant}_d{depth}.json").write_text(json.dumps(res, indent=1))
