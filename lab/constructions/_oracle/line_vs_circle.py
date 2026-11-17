"""Decisive test of the month-rollover finding: does the decision use a LINE (calendar position) or the CIRCLE?

At each site where the construction places a month's position, encode three things from the month index with the
judge's kind of affine map (activation = c + variables @ M): the line (the month's measured position U or V), the
circle (cos, sin of 2 pi month / 12), and a random 2-D control. Then remove each subspace from the real activations
(set the component along it to its mean) and count how many of the network's clear decisions survive.
Also reports how well each code can be READ from the site (held-out R2): a code can be present but unused.
Run: python3 lab/constructions/_oracle/line_vs_circle.py   -> line_vs_circle.json
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import lm, lmjudge  # noqa: E402

T = lm.get_task("gpt2_month_rollover")
ns = {"np": np}; exec(compile((HERE / "gpt2_month_rollover_linear_positions.py").read_text(), "c", "exec"), ns)
U, V = ns["U"], ns["V"]
X, N = T.X, T.N
out = T.run(np.arange(N))
rng = np.random.default_rng(0)
perm = rng.permutation(N); fit, test = perm[: N // 2], perm[N // 2:][:600]
clean = out[test]


def basis(M):
    Q, _ = np.linalg.qr(M.T)
    return Q


def readout_r2(A, Y):
    c = A[fit].mean(0)
    _, sv, Vt = np.linalg.svd(A[fit] - c, full_matrices=False)
    Z = (A - c) @ Vt[:200].T
    W, *_ = np.linalg.lstsq(np.c_[Z[fit], np.ones(len(fit))], Y[fit], rcond=None)
    P = np.c_[Z[test], np.ones(len(test))] @ W
    return float(1 - ((P - Y[test]) ** 2).sum() / ((Y[test] - Y[test].mean(0)) ** 2).sum())


res = {}
for month, col, line in (("month1", 0, U), ("month2", 1, V)):
    m = X[:, col]
    codes = {"line": line[m][:, None], "circle": np.c_[np.cos(2 * np.pi * m / 12), np.sin(2 * np.pi * m / 12)]}
    for L in (4, 6, 8, 10):
        sites = ["year", "m2"] if month == "month1" else ["m2"]
        A = {p: T.resid(L, np.arange(N), p) for p in sites}
        Qs, r2 = {}, {}
        for p in sites:
            c = A[p][fit].mean(0)
            Ml, *_ = np.linalg.lstsq(codes["line"][fit] - codes["line"][fit].mean(0), A[p][fit] - c, rcond=None)
            Mc, *_ = np.linalg.lstsq(codes["circle"][fit] - codes["circle"][fit].mean(0), A[p][fit] - c, rcond=None)
            ql = basis(Ml)
            Mc_perp = Mc - (Mc @ ql) @ ql.T                     # the circle plane with the line direction taken out
            Qs[p] = {"line": ql, "circle": basis(Mc_perp), "random 2-D": basis(rng.normal(size=(2, A[p].shape[1])))}
            r2[p] = {k: round(readout_r2(A[p], Y), 3) for k, Y in codes.items()}
        row = {"readable_r2": r2}
        for k in ("line", "circle", "random 2-D"):
            hook = {}
            for p in sites:
                Ap = A[p][test]; c = A[p][fit].mean(0); Q = Qs[p][k]
                hook[f"resid{L}@{p}"] = Ap - ((Ap - c) @ Q) @ Q.T
            row[f"decisions kept, {k} removed"] = round(lmjudge.cmatch(T.run(test, hook=hook), clean), 4)
        res[f"{month} @ depth {L}"] = row
        print(month, "depth", L, "| readable R2:", r2, "|", {k: v for k, v in row.items() if k.startswith("decisions")}, flush=True)
(HERE / "line_vs_circle.json").write_text(json.dumps(res, indent=1))
