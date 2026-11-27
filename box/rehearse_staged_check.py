"""Closing the loop on the staged rehearsal (DIARY 02:05): the blueprint the search produced (three ingredient parts at
layer-1 units, the verdict through them at the output) is put to the checker's interventions on the rehearsal network.
Known case: it must be certified; a variant with the ingredients' locations swapped (u0 at u1's unit and back) must be
rejected. Run: python3 box/rehearse_staged_check.py (re-trains the rehearsal network with the same seed)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import box.rehearse_staged as R  # noqa: E402  (trains the rehearsal network on import)
from box.blueprint import Blueprint, Part  # noqa: E402
from box.verify_vec import VecChecker  # noqa: E402

Xin = R.Xin
W1 = R.net["l1"].weight.detach().numpy().astype(np.float64); b1 = R.net["l1"].bias.detach().numpy().astype(np.float64)
W2 = R.net["l2"].weight.detach().numpy().astype(np.float64); b2 = R.net["l2"].bias.detach().numpy().astype(np.float64)
W3 = R.net["out"].weight.detach().numpy().astype(np.float64); b3 = R.net["out"].bias.detach().numpy().astype(np.float64)


class MLPNet:
    """Adapter: inputs are indices into the rehearsal dataset; sites h1 (layer-1 units) and out; the decision is the
    output rounded to the nearest integer (the verdict takes integer values)."""
    def __init__(self, target=None, weights=None):
        w = weights or {"W_x": W1, "W_h": W2, "W_o": W3}
        self.Wx, self.Wh, self.Wo = (np.asarray(w[k], np.float64) for k in ("W_x", "W_h", "W_o"))
        self.target = target

    def sites_for(self, T):
        return ["h1", "out"]

    def run(self, X, edits=None, return_sites=False):
        edits = edits or {}
        idx = np.asarray(X, int).ravel()
        h1 = Xin[idx].astype(np.float64) @ self.Wx.T + b1
        if "h1" in edits:
            h1 = edits["h1"](h1)
        h2 = np.maximum(np.maximum(h1, 0) @ self.Wh.T + b2, 0)
        out = h2 @ self.Wo.T + b3
        if "out" in edits:
            out = edits["out"](out)
        return (out, {"h1": h1, "out": out}) if return_sites else out

    def outputs(self, X, edits=None):
        return np.rint(self.run(X, edits)[:, 0]).astype(int)


def ctx_fn(X):
    idx = np.asarray(X, int).ravel()
    return {"P": R.ctx["P"][idx], "G": R.ctx["G"][idx], "fens": [R.ctx["fens"][i] for i in idx]}


def P(pid, definition, upstream, site, unit, scale=1.0, dim=8):
    u = np.zeros(dim); u[unit] = 1.0
    return Part(pid, "rule", "Pos", site, basis=u, encoding="numerical", scale=scale, definition=definition, upstream=upstream)


net = MLPNet()
idx_all = np.arange(len(Xin))[:, None]                  # the "inputs" are dataset indices (one column)
chk = VecChecker(net, idx_all, test_idx=R.te, seed=0, n_pairs=600, control_reps=200, ctx_fn=ctx_fn)
true = Blueprint("chess", [P("u0", "(count hanging_own)", [], "h1", 0), P("u1", "(count hanging_opp)", [], "h1", 1),
                           P("u2", "(count passed_pawn)", [], "h1", 2),
                           P("verdict", "(relu (mul @u2 (sub @u1 @u0)))", ["u0", "u1", "u2"], "out", 0, dim=1)],
                 "verdict", name="rehearsal blueprint (found by the search)")
swapped = Blueprint("chess", [P("u0", "(count hanging_own)", [], "h1", 1), P("u1", "(count hanging_opp)", [], "h1", 0),
                              P("u2", "(count passed_pawn)", [], "h1", 2),
                              P("verdict", "(relu (mul @u2 (sub @u1 @u0)))", ["u0", "u1", "u2"], "out", 0, dim=1)],
                    "verdict", name="the same blueprint with u0 and u1's locations swapped")
own = chk.run_alone(true, R.te)
agree = (chk.decision(true, idx_all[R.te]) == chk.base[R.te]).astype(float)
rng = np.random.default_rng(0)
spread = float(np.std([agree[rng.integers(0, len(agree), len(agree))].mean() for _ in range(500)]))
line = own - spread
print(f"run-alone line: {own:.4f} - {spread:.4f} = {line:.4f}")
for bp, want in ((true, "certified"), (swapped, "rejected")):
    r = chk.certify(bp, line)
    print(f"{'PASS' if r['verdict'] == want else 'FAIL'} {bp.name}: {r['verdict']} (expected {want}); lines {r.get('control_lines')}")
    for c in r["checks"]:
        print(f"   {'ok ' if c[1] else 'NO '} {c[0]}: {c[2]}")
