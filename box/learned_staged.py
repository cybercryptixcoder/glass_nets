"""The box on a trained network whose variables are not planted (DIARY 06:05): an MLP trained only on the verdict.

Network: ten per-square feature planes of 6,000 real positions (the three the verdict needs and seven distractors,
640 inputs; the full 3,525-input version overfit at this sample size, held-out R2 -0.49, DIARY 06:09), two hidden
layers of 32 ReLU units, trained only to output the verdict relu(g2 - g1) * g3 (g1, g2, g3 the mover's hanging
pieces, the opponent's hanging pieces, passed pawns), with early stopping on a validation split. Nothing ties any unit
to any ingredient.

The box's side:
1. Map and define together, in one search: every position-level expression of the vocabulary up to length 4 (60 probe
   positions enumerate them; values on 3,000 training positions) is a candidate variable; it is located if a single
   direction of hidden layer 1 reads it on held-out positions (linear probe, R2), against a line from controls (the
   same probe on a network of the same shape with its first layer's weights permuted). The candidates are kept in
   order of probe R2, dropping any that is nearly a function of one already kept.
2. Define the verdict through the located variables (the search with them as parts).
3. Certify by interchange interventions on the probe directions (box/verify_vec.py), against random-direction and
   weight-shuffled controls.
Run: python3 box/learned_staged.py"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box.search import best_definitions, enumerate_exprs, vocab  # noqa: E402
from box.blueprint import Blueprint, Part  # noqa: E402
from box.verify_vec import VecChecker  # noqa: E402

t0 = time.time()
torch.manual_seed(0); rng = np.random.default_rng(0)
items = json.load(open(os.path.expanduser("~/kdata/chess/LD2/meta.json")))
items = items["items"] if isinstance(items, dict) else items
fens = [it["fen"] for it in items[:6000]]
d = vocab.get("chess")
ctx = vocab.featurize(fens)
ev = lambda e: np.asarray(vocab.evaluate(e, d, ctx), float)  # noqa: E731
g1, g2, g3 = ev("(sum hanging_own)"), ev("(sum hanging_opp)"), ev("(sum passed_pawn)")
v = np.maximum(g2 - g1, 0) * g3
sq_names = [n for n in d.atoms if d.atoms[n].type == "Sq"]
planes_used = ["hanging_own", "hanging_opp", "passed_pawn", "own_P", "opp_P", "att_us", "att_them", "mob_us", "defended", "pinned"]
cols = [sq_names.index(n) for n in planes_used]
X = torch.tensor(ctx["P"][:, :, cols].reshape(len(fens), -1), dtype=torch.float32)
tr, va, te = np.arange(0, 4000), np.arange(4000, 4500), np.arange(4500, 6000)
net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
Y = torch.tensor(v, dtype=torch.float32)[:, None]
best_state, best_va = None, 1e9
for step in range(8000):
    idx = torch.tensor(rng.choice(tr, 512))
    loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0:
        with torch.no_grad():
            lv = float(((net(X[torch.tensor(va)]) - Y[torch.tensor(va)]) ** 2).mean())
        if lv < best_va:
            best_va, best_state = lv, {k: t.clone() for k, t in net.state_dict().items()}
net.load_state_dict(best_state)
with torch.no_grad():
    H1 = torch.relu(net[0](X)).numpy().astype(np.float64)
    OUT = net(X).numpy()[:, 0]
r2 = lambda p, y: float(1 - np.mean((p - y) ** 2) / np.var(y))  # noqa: E731
print(f"trained {round(time.time() - t0)} s; held-out R2 of the network against the verdict {r2(OUT[te], v[te]):.3f}", flush=True)

# shuffled-first-layer control network (same shape), for the location line
with torch.no_grad():
    W = net[0].weight.detach().numpy().ravel().copy(); rng.shuffle(W)
    Hc = np.maximum(X.numpy() @ W.reshape(net[0].weight.shape).T + net[0].bias.numpy(), 0).astype(np.float64)


def probe_r2(H, y):
    A = np.column_stack([H[tr], np.ones(len(tr))])
    w, *_ = np.linalg.lstsq(A, y[tr], rcond=1e-6)
    return r2(np.column_stack([H[te], np.ones(len(te))]) @ w, y[te]), w


# 1. candidate variables: distinct position-level expressions up to length 4
probe_idx = list(range(60))
cprobe = {k: (val[probe_idx] if k != "fens" else [val[i] for i in probe_idx]) for k, val in ctx.items()}
pool = enumerate_exprs(d, cprobe, 4, cap=60000)
cands = [t for t, _v, n in pool["Pos"] if n >= 2]
print(f"{len(cands)} position-level candidates", round(time.time() - t0), "s", flush=True)
scored = []
for t in cands:
    y = ev(t)
    if y[tr].std() == 0 or y[te].std() == 0:
        continue
    s, w = probe_r2(H1, y)
    scored.append((s, t, y, w))
scored.sort(key=lambda z: -z[0])
# probe R2 only ranks candidates; a random-feature layer can read any linear function of its input almost as well
# (the shuffled-layer control below shows it), so location is decided by the interventions, not by the probe
ctrl = sorted([probe_r2(Hc, z[2])[0] for z in scored[:200]], reverse=True)
line_loc = float(np.quantile(ctrl, 0.99))
kept = []
for s, t, y, w in scored:
    if any(abs(np.corrcoef(y, k[2])[0, 1]) > 0.95 for k in kept):
        continue
    kept.append((s, t, y, w))
    if len(kept) >= 8:
        break
print(f"(the same probe on the shuffled-layer control reaches {line_loc:.3f}); candidate variables, best probe R2 first:", flush=True)
for s, t, y, w in kept:
    print(f"   {t}: probe R2 {s:.3f}", flush=True)

# 2. define the verdict through the located variables
pf = {f"u{i}": ("Pos", k[2][:1000]) for i, k in enumerate(kept)}
pt = {f"u{i}": ("Pos", k[2][te]) for i, k in enumerate(kept)}
ctx_fit = {k: (val[:1000] if k != "fens" else val[:1000]) for k, val in ctx.items()}
ctx_te = {k: (val[te] if k != "fens" else [val[i] for i in te]) for k, val in ctx.items()}
through, _ = best_definitions(d, ctx_fit, OUT[:1000], 6, "Pos", k=1, ctx_test=ctx_te, target_test=OUT[te],
                              parts_fit=pf, parts_test=pt, only_types=("Pos",))
print(f"verdict through the located variables: {through[0]}", flush=True)

# 3. certify by interventions on the probe directions
W1, b1 = net[0].weight.detach().numpy().astype(np.float64), net[0].bias.detach().numpy().astype(np.float64)
W2, b2 = net[2].weight.detach().numpy().astype(np.float64), net[2].bias.detach().numpy().astype(np.float64)
W3, b3 = net[4].weight.detach().numpy().astype(np.float64), net[4].bias.detach().numpy().astype(np.float64)
Xn = X.numpy().astype(np.float64)


class Net:
    def __init__(self, target=None, weights=None):
        w = weights or {"W_x": W1, "W_h": W2, "W_o": W3}
        self.Wx, self.Wh, self.Wo = (np.asarray(w[k], np.float64) for k in ("W_x", "W_h", "W_o"))
        self.target = target

    def sites_for(self, T):
        return ["h1", "out"]

    def run(self, Xi, edits=None, return_sites=False):
        edits = edits or {}
        i = np.asarray(Xi, int).ravel()
        h1 = np.maximum(Xn[i] @ self.Wx.T + b1, 0)
        if "h1" in edits:
            h1 = edits["h1"](h1)
        out = np.maximum(h1 @ self.Wh.T + b2, 0) @ self.Wo.T + b3
        if "out" in edits:
            out = edits["out"](out)
        return (out, {"h1": h1, "out": out}) if return_sites else out

    def outputs(self, Xi, edits=None):
        return np.rint(self.run(Xi, edits)[:, 0]).astype(int)


def ctx_fn(Xi):
    i = np.asarray(Xi, int).ravel()
    return {"P": ctx["P"][i], "G": ctx["G"][i], "fens": [ctx["fens"][j] for j in i]}


used = sorted({int(t[2:]) for t in through[0]["definition"].replace("(", " ").replace(")", " ").split() if t.startswith("@u")})
parts = []
for j in used:
    s, t, y, w = kept[j]
    u = w[:-1] / np.linalg.norm(w[:-1])
    parts.append(Part(f"u{j}", "rule", "Pos", "h1", basis=u, encoding="numerical", scale=float(np.linalg.norm(w[:-1])), definition=t))
parts.append(Part("verdict", "rule", "Pos", "out", basis=np.array([1.0]), encoding="numerical", definition=through[0]["definition"],
                  upstream=[f"u{j}" for j in used]))
bp = Blueprint("chess", parts, "verdict", name="learned staged network")
idx_all = np.arange(len(fens))[:, None]
chk = VecChecker(Net(), idx_all, test_idx=te, seed=0, n_pairs=800, control_reps=200, ctx_fn=ctx_fn)
own = chk.run_alone(bp, te)
rep = chk.certify(bp, own - 0.01)
print(f"verdict: {rep['verdict']}; lines {rep.get('control_lines')}", flush=True)
for c in rep["checks"]:
    print(f"   {'ok ' if c[1] else 'NO '} {c[0]}: {c[2]}", flush=True)
json.dump({"network_r2": r2(OUT[te], v[te]), "location_line": line_loc,
           "located": [{"definition": t, "probe_r2": s} for s, t, y, w in kept], "verdict_definition": through[0],
           "certify": {"verdict": rep["verdict"], "checks": [(c[0], bool(c[1]), c[2]) for c in rep["checks"]],
                       "lines": rep.get("control_lines")}, "seconds": round(time.time() - t0)},
          open("results/build/learned_staged_session2.json", "w"), indent=1, default=float)
print("done", round(time.time() - t0), "s")
