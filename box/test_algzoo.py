"""The saved S5a cases on AlgZoo M2,2 (scripts/glass/info_measure.py CASES): the real account must pass; a copy must
fail. Written before the first run. Two questions:
1. Do the saved accounts, as given (Python programs), get through the checker? (No account can: they are code.)
2. Translated into blueprints in the starting vocabulary: is the real account (round 1, run a7c8, "prior-only") certified,
   and is the integer copy of the network, written as a blueprint of its own units with rounded weights, rejected?
   Expected by SPEC R9: the truth tests alone cannot reject a copy; if the copy is certified, copy rejection must come
   from R9's own tests (answering without running; the bounded reader), not from the checker's truth tests.
Run: python3 box/test_algzoo.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box.adapters.algzoo_rnn import AlgZooRNN, ROOT  # noqa: E402
from box.blueprint import Blueprint, Part  # noqa: E402
from box.verify_vec import VecChecker, fit_location  # noqa: E402


def P(pid, definition, upstream, site="input", basis=None, scale=1.0, typ="rule"):
    return Part(pid, typ, "Num", site, basis=basis, encoding="numerical", scale=scale, definition=definition, upstream=upstream)


def account_blueprint(chk, fit_idx):
    """The round 1 M2,2 account (run 20261118-015915-a7c8), part for part, with each located part's direction and
    scale fitted by least squares on the fitting half of the inputs."""
    parts = [P("x0", "(at x 0)", []), P("x1", "(at x 1)", []),
             P("h1_pos", "(relu @x0)", ["x0"], "h1"),
             P("h1_neg", "(relu (neg @x0))", ["x0"], "h1"),
             P("rec", "(sub @h1_pos @h1_neg)", ["h1_pos", "h1_neg"], "h1"),
             P("h2_up", "(relu (sub @rec @x1))", ["rec", "x1"], "h2"),
             P("h2_un", "(relu (sub @x1 @rec))", ["x1", "rec"], "h2"),
             P("decision", "(gt @h2_up @h2_un)", ["h2_up", "h2_un"], "readout")]
    bp = Blueprint("seq", parts, "decision", name="M2,2 real account (a7c8), as a blueprint")
    vals = chk.values(bp, chk.X)
    for p in parts:
        if p.site not in ("input", "readout"):
            p.basis, p.scale = fit_location(chk.sites, p.site, vals[p.id], fit_idx)
    return bp


def int_copy_blueprint(net):
    """The network itself, unit by unit, with every weight rounded to an integer (T2's 'int copy'), written in the
    vocabulary: each part within L = 7 and at most 2 constants; each unit located at its own coordinate."""
    Wx, Wh, Wo = np.rint(net.Wx[0]), np.rint(net.Wh), np.rint(net.Wo)
    e = lambda k: np.eye(2)[k]  # noqa: E731
    f = lambda v: repr(float(v))  # noqa: E731
    parts = [P("x0", "(at x 0)", []), P("x1", "(at x 1)", []),
             P("u1_0", f"(max2 (mul @x0 {f(Wx[0])}) (mul @x0 0))", ["x0"], "h1", e(0)),
             P("u1_1", f"(max2 (mul @x0 {f(Wx[1])}) (mul @x0 0))", ["x0"], "h1", e(1)),
             P("r0", f"(add (mul @u1_0 {f(Wh[0, 0])}) (mul @u1_1 {f(Wh[1, 0])}))", ["u1_0", "u1_1"]),
             P("r1", f"(add (mul @u1_0 {f(Wh[0, 1])}) (mul @u1_1 {f(Wh[1, 1])}))", ["u1_0", "u1_1"]),
             P("u2_0", f"(relu (add (mul @x1 {f(Wx[0])}) @r0))", ["x1", "r0"], "h2", e(0)),
             P("u2_1", f"(relu (add (mul @x1 {f(Wx[1])}) @r1))", ["x1", "r1"], "h2", e(1)),
             P("s0", f"(add (mul @u2_0 {f(Wo[0, 0])}) (mul @u2_1 {f(Wo[1, 0])}))", ["u2_0", "u2_1"]),
             P("s1", f"(add (mul @u2_0 {f(Wo[0, 1])}) (mul @u2_1 {f(Wo[1, 1])}))", ["u2_0", "u2_1"]),
             P("decision", "(gt @s1 @s0)", ["s0", "s1"], "readout")]
    # r0, r1 are linear in h1 and s0, s1 linear in h2: locate them too (direction and scale), as the copy would
    for p in parts:
        if p.id in ("r0", "r1"):
            w = Wh[:, int(p.id[1])]; p.site, p.basis, p.scale = "h1", w / np.linalg.norm(w), float(np.linalg.norm(w))
        if p.id in ("s0", "s1"):
            w = Wo[:, int(p.id[1])]; p.site, p.basis, p.scale = "h2", w / np.linalg.norm(w), float(np.linalg.norm(w))
    return Blueprint("seq", parts, "decision", name="M2,2 integer copy (the network's units, rounded weights), as a blueprint")


def main():
    net = AlgZooRNN("zoo_2nd_argmax_2_2")
    X = np.load(os.path.join(ROOT, "lab", "models", "zoo_2nd_argmax_2_2", "inputs.npy"))[:4000]
    fit_idx, test_idx = np.arange(0, 2000), np.arange(2000, 4000)
    chk = VecChecker(net, X, test_idx=test_idx, seed=0, n_pairs=1500, control_reps=200)
    # 1. the saved programs, as given
    src = open(os.path.join(ROOT, "lab", "constructions", "zoo_2nd_argmax_2_2", "20261118-015915-a7c8", "construction.py")).read()
    print("as given (code):", chk.certify(src, 1.0)["verdict"], "-", chk.form(src)[1])
    # 2. as blueprints
    real = account_blueprint(chk, fit_idx)
    copy = int_copy_blueprint(net)
    own = chk.run_alone(real, test_idx)
    agree = (chk.decision(real, X[test_idx]) == chk.base[test_idx]).astype(float)
    rng = np.random.default_rng(0)
    spread = float(np.std([agree[rng.integers(0, len(agree), len(agree))].mean() for _ in range(500)]))
    line = own - spread
    print(f"run-alone line from the real account's own held-out agreement: {own:.4f} - spread {spread:.4f} = {line:.4f}")
    for bp in (real, copy):
        r = chk.certify(bp, line)
        print(f"\n{bp.name}: {r['verdict'].upper()}  (family-wise IIA lines {r.get('control_lines')})")
        for c in r["checks"]:
            print(f"   {'ok ' if c[1] else 'NO '} {c[0]}: {c[2]}")


if __name__ == "__main__":
    main()
