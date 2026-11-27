"""The deciding saved S5a case on AlgZoo M4,3 (X0: must pass): round 1's account c928 (run 20261118-015037-c928),
translated part for part into the starting vocabulary, against a near-copy of the network (its own units with its
exact weights, written in the same vocabulary within the same limits). Written before the first run.
Expected by SPEC R9: the truth tests certify both (a copy is true), so copy rejection must come from R9's own tests.
Run: python3 box/test_algzoo_m43.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box.adapters.algzoo_rnn import AlgZooRNN, ROOT  # noqa: E402
from box.blueprint import Blueprint, Part  # noqa: E402
from box.verify_vec import VecChecker, fit_location  # noqa: E402

f = lambda v: repr(round(float(v), 4))  # noqa: E731


def P(pid, definition, upstream, site="input"):
    return Part(pid, "rule", "Num", site, encoding="numerical", definition=definition, upstream=upstream)


def argmax3(parts, s):
    """The task's decision (argmax of three scores) as readout parts."""
    parts += [P("b10", f"(gt @{s[1]} @{s[0]})", [s[0], s[1]], "readout"), P("b20", f"(gt @{s[2]} @{s[0]})", [s[0], s[2]], "readout"),
              P("b21", f"(gt @{s[2]} @{s[1]})", [s[1], s[2]], "readout"), P("w2", "(mul @b20 @b21)", ["b20", "b21"], "readout"),
              P("nb21", "(add (neg @b21) 1)", ["b21"], "readout"), P("w1", "(mul @b10 @nb21)", ["b10", "nb21"], "readout"),
              P("decision", "(add (mul @w2 2) @w1)", ["w2", "w1"], "readout")]


def c928_blueprint():
    g = [3.9653, 4.6995, -4.7163, -3.9652]
    C1 = np.array([-0.2780, 1.6167, 0.2178, -2.4174]); C2 = np.array([-0.2274, 2.1552, 0.1764, -2.6103])
    flip = np.array([1.0, -1.0, 1.0, -1.0]); CT = np.stack([C1, C2, C2 * flip, C1 * flip])
    Pk = [1.5, 14.902, -14.659]; Tk = [1.0352, 1.1725, 0.7286]
    lib = {}
    parts = [P("x0", "(at x 0)", []), P("x1", "(at x 1)", []), P("x2", "(at x 2)", []),
             P("p1", "(relu @x0)", ["x0"], "h1"), P("n1", "(relu (neg @x0))", ["x0"], "h1"),
             P("h1_u1", f"(mul @p1 {f(g[0])})", ["p1"], "h1"), P("h1_u2", f"(mul @p1 {f(g[1])})", ["p1"], "h1"),
             P("h1_u3", f"(mul @n1 {f(-g[2])})", ["n1"], "h1"), P("h1_u4", f"(mul @n1 {f(-g[3])})", ["n1"], "h1")]
    for t in (1, 2):
        u = [f"h{t}_u{k}" for k in range(1, 5)]
        site = f"h{t}"
        parts += [P(f"m1_{t}", f"(add @{u[0]} @{u[3]})", [u[0], u[3]], site), P(f"s1_{t}", f"(sub @{u[0]} @{u[3]})", [u[0], u[3]], site),
                  P(f"m2_{t}", f"(add @{u[1]} @{u[2]})", [u[1], u[2]], site), P(f"s2_{t}", f"(sub @{u[1]} @{u[2]})", [u[1], u[2]], site)]
        for k in range(4):
            c = CT[k]
            name = f"unit{t}{k}"
            lib[name] = {"kind": "operation", "args": ["Num"] * 5, "ret": "Num",
                         "source": f"def {name}(m1, s1, m2, s2, x):\n    return {f(c[0])} * m1 + {f(c[1])} * s1 + "
                                   f"{f(c[2])} * m2 + {f(c[3])} * s2 + {f(g[k])} * x\n"}
            parts.append(P(f"h{t + 1}_u{k + 1}", f"(relu (${name} @m1_{t} @s1_{t} @m2_{t} @s2_{t} @x{t}))",
                           [f"m1_{t}", f"s1_{t}", f"m2_{t}", f"s2_{t}", f"x{t}"], f"h{t + 1}"))
    parts += [P("m1", "(add @h3_u1 @h3_u4)", ["h3_u1", "h3_u4"], "h3"), P("m2", "(add @h3_u2 @h3_u3)", ["h3_u2", "h3_u3"], "h3")]
    for k in range(3):
        parts.append(P(f"score{k}", f"(mul (sub @m2 (mul @m1 {f(Tk[k])})) {f(Pk[k])})", ["m1", "m2"], "out"))
    argmax3(parts, ["score0", "score1", "score2"])
    return Blueprint("seq", parts, "decision", library=lib, name="M4,3 real account (c928), as a blueprint")


def near_copy_blueprint(net):
    Wx, Wh, Wo = net.Wx[0], net.Wh, net.Wo
    parts = [P("x0", "(at x 0)", []), P("x1", "(at x 1)", []), P("x2", "(at x 2)", [])]
    for k in range(4):
        parts.append(P(f"h1_{k}", f"(relu (mul @x0 {f(Wx[k])}))", ["x0"], "h1"))
    lib = {}
    for t in (1, 2):
        h = [f"h{t}_{j}" for j in range(4)]
        for k in range(4):
            name = f"pre{t}{k}"
            lib[name] = {"kind": "operation", "args": ["Num"] * 5, "ret": "Num",
                         "source": f"def {name}(a, b, c, d, x):\n    return {f(Wh[0, k])} * a + {f(Wh[1, k])} * b + "
                                   f"{f(Wh[2, k])} * c + {f(Wh[3, k])} * d + {f(Wx[k])} * x\n"}
            parts.append(P(f"h{t + 1}_{k}", f"(relu (${name} @{h[0]} @{h[1]} @{h[2]} @{h[3]} @x{t}))", h + [f"x{t}"], f"h{t + 1}"))
    h = [f"h3_{j}" for j in range(4)]
    for c in range(3):
        name = f"readout{c}"
        lib[name] = {"kind": "operation", "args": ["Num"] * 4, "ret": "Num",
                     "source": f"def {name}(a, b, c, d):\n    return {f(Wo[0, c])} * a + {f(Wo[1, c])} * b + "
                               f"{f(Wo[2, c])} * c + {f(Wo[3, c])} * d\n"}
        parts.append(P(f"score{c}", f"(${name} @{h[0]} @{h[1]} @{h[2]} @{h[3]})", h, "out"))
    argmax3(parts, ["score0", "score1", "score2"])
    return Blueprint("seq", parts, "decision", library=lib, name="M4,3 near-copy (the network's own units, exact weights), as a blueprint")


def main():
    net = AlgZooRNN("zoo_2nd_argmax_4_3")
    X = np.load(os.path.join(ROOT, "lab", "models", "zoo_2nd_argmax_4_3", "inputs.npy"))[:4000]
    fit_idx, test_idx = np.arange(0, 2000), np.arange(2000, 4000)
    chk = VecChecker(net, X, test_idx=test_idx, seed=0, n_pairs=1500, control_reps=100)
    real, copy = c928_blueprint(), near_copy_blueprint(net)
    import re
    unit = re.compile(r"^h(\d)_u?(\d)$")
    score = re.compile(r"^score(\d)$")
    for bp, one_based in ((real, True), (copy, False)):
        vals = chk.values(bp, chk.X)
        loc = {}
        for p in bp.order():
            if p.site in ("input", "readout"):
                continue
            ms_ = score.match(p.id)
            if ms_ and p.site == "out":   # a class score is the network's logit for that class (scale fitted)
                k = int(ms_.group(1)); u = np.zeros(chk.sites["out"].shape[1]); u[k] = 1.0
                hk = chk.sites["out"][fit_idx, k]; v = vals[p.id][fit_idx]
                p.basis, p.scale = u, float(hk @ v / (hk @ hk))
                loc[p.id] = True
                continue
            m = unit.match(p.id)
            if m:   # a unit the account names: declared, not fitted (units that move together on the data cannot be told apart by fitting)
                k = int(m.group(2)) - (1 if one_based else 0)
                u = np.zeros(chk.sites[p.site].shape[1]); u[k] = 1.0
                p.basis, p.scale = u, 1.0
            elif p.upstream and all(q in loc and bp.part(q).site == p.site for q in p.upstream):
                # linear in parts already located at this site: its location follows from theirs
                A = np.stack([vals[q][fit_idx] for q in p.upstream], 1)
                beta, *_ = np.linalg.lstsq(A, vals[p.id][fit_idx], rcond=None)
                w = sum(b * bp.part(q).scale * np.asarray(bp.part(q).basis) for b, q in zip(beta, p.upstream))
                n = float(np.linalg.norm(w)); p.basis, p.scale = w / n, n
            else:
                p.basis, p.scale = fit_location(chk.sites, p.site, vals[p.id], fit_idx, prefer_units=False)
            loc[p.id] = True
        print(bp.name, "-", len(bp.parts), "parts,", len(loc), "located")
    own = chk.run_alone(real, test_idx)
    agree = (chk.decision(real, X[test_idx]) == chk.base[test_idx]).astype(float)
    rng = np.random.default_rng(0)
    spread = float(np.std([agree[rng.integers(0, len(agree), len(agree))].mean() for _ in range(500)]))
    line = own - spread
    print(f"run-alone line from the real account's own held-out agreement: {own:.4f} - {spread:.4f} = {line:.4f}")
    for bp in (real, copy):
        r = chk.certify(bp, line)
        fails = [c for c in r["checks"] if not c[1]]
        print(f"\n{bp.name}: {r['verdict'].upper()}; lines {r.get('control_lines')}")
        for c in r["checks"][:2]:
            print(f"   {'ok ' if c[1] else 'NO '} {c[0]}: {c[2]}")
        iia = [c for c in r["checks"] if c[0].startswith("interventions")]
        print(f"   interventions: {sum(c[1] for c in iia)} of {len(iia)} located parts above the line")
        for c in fails[:12]:
            print(f"   NO  {c[0]}: {c[2]}")


if __name__ == "__main__":
    main()
