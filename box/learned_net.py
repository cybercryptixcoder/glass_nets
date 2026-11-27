"""A trained network with a known answer, for the box's Gate 2 rehearsal (the development analog of the bench's S3:
a verdict computed in stages from intermediate concepts, learned, not planted).

The network: a 640-32-32-1 ReLU MLP on ten per-square feature planes of 20,000 real positions (LD2's position set),
trained only to output v = relu(A - B) * C, where A = (sum att_us), B = (sum att_them) (all attacks by the side to
move and by the opponent) and C = (count own_P) (the mover's pawns). Seven of the ten planes are distractors. Nothing
ties any unit to any ingredient. (The first attempt used the sparse ingredients of the staged rehearsal, hanging
pieces and passed pawns; that verdict is zero on 95% of positions and the network did not learn it from 6,000
positions, held-out R2 -0.01; DIARY 06:20.)

Sites, in order: z1 (layer 1 before its ReLU), h1, z2, h2, out. The network's decision is its output rounded to the
nearest point (the verdict's own unit; no other grain is chosen).

This module trains (or loads) the network and provides the adapter; run as main, it checks the measurement on the
known answer before the box runs: the true account must pass and wrong accounts must fail, under the checker as it is.
Run: python3 -m box.learned_net"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box.search import vocab  # noqa: E402

N = 20000
PLANES = ["att_us", "att_them", "own_P", "hanging_own", "hanging_opp", "passed_pawn", "opp_P", "mob_us", "defended",
          "pinned"]
TRUTH = {"A": "(sum att_us)", "B": "(sum att_them)", "C": "(count own_P)", "verdict": "(mul (relu (sub @A @B)) @C)"}
WEIGHTS = "results/build/learned_net_session2.npz"
TR, VA, TE = np.arange(0, 14000), np.arange(14000, 16000), np.arange(16000, 20000)
SITES = ["z1", "h1", "z2", "h2", "out"]


def data():
    items = json.load(open(os.path.expanduser("~/kdata/chess/LD2/meta.json")))
    items = items["items"] if isinstance(items, dict) else items
    fens = [it["fen"] for it in items[:N]]
    d = vocab.get("chess")
    cache = "/tmp/learned_feat_20k.npz"  # this module's own featurization of the same positions (not repository data)
    if os.path.exists(cache):
        z = np.load(cache)
        ctx = {"P": z["P"], "G": z["G"], "fens": fens}
    else:
        ctx = vocab.featurize(fens)
        np.savez(cache, P=ctx["P"], G=ctx["G"])
    sq = [n for n in d.atoms if d.atoms[n].type == "Sq"]
    X = ctx["P"][:, :, [sq.index(n) for n in PLANES]].reshape(N, -1).astype(np.float64)
    return d, ctx, X


def evaluate(d, ctx, expr, rows=None):
    c = ctx if rows is None else {k: (val[rows] if k != "fens" else [val[i] for i in rows]) for k, val in ctx.items()}
    return np.asarray(vocab.evaluate(expr, d, c), float)


def verdict(d, ctx):
    A, B, C = (evaluate(d, ctx, TRUTH[k]) for k in "ABC")
    return np.maximum(A - B, 0) * C


def train(X, v, seed=0, steps=20000):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Xt, Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(v / 10.0, dtype=torch.float32)[:, None]
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 32),
                              torch.nn.ReLU(), torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    best, best_state = 1e9, None
    vt = torch.tensor(VA)
    for step in range(steps):
        idx = torch.tensor(rng.choice(TR, 512))
        loss = ((net(Xt[idx]) - Yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0 or step == steps - 1:
            with torch.no_grad():
                lv = float(((net(Xt[vt]) - Yt[vt]) ** 2).mean())
            if lv < best:
                best, best_state = lv, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    g = lambda t: t.detach().numpy().astype(np.float64)  # noqa: E731
    return {"W1": g(net[0].weight), "b1": g(net[0].bias), "W2": g(net[2].weight), "b2": g(net[2].bias),
            "W3": g(net[4].weight) * 10.0, "b3": g(net[4].bias) * 10.0}


class Net:
    """The adapter for box/verify_vec.py. Inputs are row indices into X; weights W_x, W_h, W_o are layers 1 to 3
    (the checker's weight-shuffled controls permute each)."""

    def __init__(self, target, weights=None):
        self.target = target  # (X, params)
        X, p = target
        self.X, self.p = X, p
        w = weights or {"W_x": p["W1"], "W_h": p["W2"], "W_o": p["W3"]}
        self.Wx, self.Wh, self.Wo = (np.asarray(w[k], np.float64) for k in ("W_x", "W_h", "W_o"))

    def sites_for(self, _T):
        return SITES

    def run(self, Xi, edits=None, return_sites=False):
        edits = edits or {}
        i = np.asarray(Xi, int).ravel()
        s = {}
        h = self.X[i] @ self.Wx.T + self.p["b1"]
        for name, f in (("z1", None), ("h1", lambda z: np.maximum(z, 0)), ("z2", lambda h: h @ self.Wh.T + self.p["b2"]),
                        ("h2", lambda z: np.maximum(z, 0)), ("out", lambda h: h @ self.Wo.T + self.p["b3"])):
            if f is not None:
                h = f(h)
            if name in edits:
                h = edits[name](h)
            s[name] = h
        return (h, s) if return_sites else h

    def outputs(self, Xi, edits=None):
        return np.rint(self.run(Xi, edits)[:, 0]).astype(int)


def load(verbose=True):
    t0 = time.time()
    d, ctx, X = data()
    v = verdict(d, ctx)
    if os.path.exists(WEIGHTS):
        p = dict(np.load(WEIGHTS))
    else:
        p = train(X, v)
        np.savez(WEIGHTS, **p)
    net = Net((X, p))
    out = net.run(np.arange(N))[:, 0]
    r2 = 1 - np.mean((out[TE] - v[TE]) ** 2) / np.var(v[TE])
    if verbose:
        print(f"network ready {round(time.time() - t0)} s: held-out R2 {r2:.4f}, RMSE {np.sqrt(np.mean((out[TE] - v[TE]) ** 2)):.2f} "
              f"(verdict sd {v[TE].std():.1f}); decision agreement with the verdict {np.mean(np.rint(out[TE]) == v[TE]):.3f}", flush=True)
    return d, ctx, X, v, net, {"heldout_r2": float(r2)}


def ctx_fn_for(ctx):
    def ctx_fn(Xi):
        i = np.asarray(Xi, int).ravel()
        return {k: (val[i] if k != "fens" else [val[j] for j in i]) for k, val in ctx.items()}
    return ctx_fn


def probe(sites, site, y, rows=TR):
    H = sites[site][rows]
    A = np.column_stack([H, np.ones(len(rows))])
    w, *_ = np.linalg.lstsq(A, y[rows], rcond=1e-6)
    pred = np.column_stack([sites[site][TE], np.ones(len(TE))]) @ w
    r2 = 1 - np.mean((pred - y[TE]) ** 2) / np.var(y[TE])
    return w[:-1], float(r2)


if __name__ == "__main__":
    from box.blueprint import Blueprint, Part
    from box.verify_vec import VecChecker
    d, ctx, X, v, net, info = load()
    _, sites = net.run(np.arange(N), return_sites=True)
    vals = {k: evaluate(d, ctx, TRUTH[k]) for k in "ABC"}
    vals["D"] = vals["A"] - vals["B"]
    DEFS = dict(TRUTH, D="(sub (sum att_us) (sum att_them))")
    results = {"network": info, "accounts": {}}

    def account(site, dirs, names, verdict_def, name=""):
        parts = [Part(k, "rule", "Pos", site, basis=dirs[k] / np.linalg.norm(dirs[k]), encoding="numerical",
                      scale=float(np.linalg.norm(dirs[k])), definition=DEFS[k]) for k in names]
        parts.append(Part("verdict", "rule", "Pos", "out", basis=np.array([1.0]), encoding="numerical",
                          definition=verdict_def, upstream=list(names)))
        return Blueprint("chess", parts, "verdict", name=name)

    # the checker sees held-out positions only (probe directions are fitted on training positions); DIARY 06:50
    for joint in (False, True):
        ver = "v1" if joint else "v0"
        chk = VecChecker(net, TE[:, None], seed=0, n_pairs=800, control_reps=200, ctx_fn=ctx_fn_for(ctx), joint=joint)
        for site in ("z1", "h1"):
            dirs, r2s = {}, {}
            for k in "ABCD":
                dirs[k], r2s[k] = probe(sites, site, vals[k])
            print(f"[{ver}] site {site}: probe R2 " + ", ".join(f"{k} {r2s[k]:.4f}" for k in "ABCD") +
                  "; cosines AB %.2f AC %.2f BC %.2f DC %.2f" % tuple(
                      float(dirs[a] @ dirs[b] / np.linalg.norm(dirs[a]) / np.linalg.norm(dirs[b])) for a, b in ("AB", "AC", "BC", "DC")), flush=True)
            true_bp = account(site, dirs, "ABC", TRUTH["verdict"], "true account")
            ra = chk.run_alone(true_bp)
            line = ra - 2.5 * np.sqrt(ra * (1 - ra) / len(TE))
            cases = [("true account", true_bp),
                     ("the difference A - B as one part", account(site, dirs, "DC", "(mul (relu @D) @C)")),
                     ("must fail: locations permuted", account(site, {"A": dirs["B"], "B": dirs["C"], "C": dirs["A"]}, "ABC", TRUTH["verdict"])),
                     ("must fail: A and B exchanged in the verdict", account(site, dirs, "ABC", "(mul (relu (sub @B @A)) @C)"))]
            for name, bp in cases:
                rep = chk.certify(bp, line)
                print(f"   [{ver}] {name} at {site}: {rep['verdict']}; lines {rep.get('control_lines')}", flush=True)
                for c in rep["checks"]:
                    print(f"      {'ok ' if c[1] else 'NO '} {c[0]}: {c[2]}", flush=True)
                results["accounts"][f"{ver} | {name} @ {site}"] = {
                    "verdict": rep["verdict"], "lines": rep.get("control_lines"),
                    "checks": [(c[0], bool(c[1]), c[2]) for c in rep["checks"]]}
            results[f"{ver} run_alone_line @ {site}"] = line
    results["run_alone_line_note"] = ("the true account's own run-alone agreement on the held-out positions less 2.5 "
                                     "standard errors (R2's line on planted networks: the true account's agreement and its spread)")
    json.dump(results, open("results/build/learned_truth_session2.json", "w"), indent=1, default=float)
    print("done")
