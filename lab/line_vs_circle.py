"""Line or circle? For any month-comparison task: which geometry of the months does the network's decision use?

  python3 lab/line_vs_circle.py <target> [depths...]

1. Behaviour: the network's next-vs-same margin modelled by (a) the ORDER of month2 vs month1 on the calendar line,
   (b) their CIRCULAR difference, (c) additive month effects (fitted on half the prompts, scored on the other half).
2. Each month's line position = its additive effect on the margin (measured on this model, so nothing is borrowed
   from another model). At each depth, for month1 at the year and month2 tokens and month2 at its own token, the line
   (1-D), the circle (cos, sin of 2 pi m / 12, orthogonal to the line) and a random 2-D plane are encoded with an
   affine map and REMOVED; the share of the network's clear decisions kept is reported. Because one answer can dominate the clear decisions (GPT-2 medium: 98%), the
   share of the margin's variation over prompts that survives each removal (margin R2) is reported too.
Writes lab/models/<target>/line_vs_circle.json. Generalises lab/constructions/_oracle/line_vs_circle.py (GPT-2 small).
"""
import json, sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import lm, lmjudge  # noqa: E402


def run(tid, depths=None, seed=0):
    T = lm.get_task(tid)
    X, N = T.X, T.N
    out = T.run(np.arange(N))
    mg = out[:, 1] - out[:, 0]
    dec, clear = out.argmax(1), lmjudge.confident(out)
    a, b = X[:, 0], X[:, 1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N); fit, test = perm[: N // 2], perm[N // 2:][:600]
    oh = lambda v: np.eye(12)[v]  # noqa: E731
    one = np.ones((N, 1))
    beh = {}
    for name, F in (("order (3 params)", np.c_[one, b < a, b == a]), ("circular difference (12)", oh((b - a) % 12)),
                    ("additive months (24)", np.c_[oh(a), oh(b)])):
        W, *_ = np.linalg.lstsq(F[fit], mg[fit], rcond=None); P = F @ W
        beh[name] = {"margin_r2": round(float(1 - ((P[test] - mg[test]) ** 2).sum() / ((mg[test] - mg[test].mean()) ** 2).sum()), 3),
                     "clear_decisions_right": round(float(((P[test] > 0) == (dec[test] == 1))[clear[test]].mean()), 3)}
    W, *_ = np.linalg.lstsq(np.c_[oh(a), oh(b)], mg, rcond=None)
    U, V = W[:12], W[12:]
    from scipy.stats import spearmanr
    res = {"target": tid, "model": T.meta["model"], "behaviour": beh,
           "line_positions": {"month1": np.round(U - U.mean(), 3).tolist(), "month2": np.round(V - V.mean(), 3).tolist(),
                              "rank_corr_with_calendar": [round(float(spearmanr(range(12), U)[0]), 3), round(float(spearmanr(range(12), V)[0]), 3)]},
           "constant_baseline": round(float(lmjudge.cmatch(np.broadcast_to(out[fit].mean(0), out[test].shape), out[test])), 3), "ablations": {}}
    nL = T.cfg["n_layers"]
    depths = depths or [nL // 3, nL // 2, 2 * nL // 3, 5 * nL // 6]
    clean = out[test]
    for month, col, line, sites in (("month1", 0, U, ["year", "m2"]), ("month2", 1, V, ["m2"])):
        m = X[:, col]
        codes = {"line": line[m][:, None], "circle": np.c_[np.cos(2 * np.pi * m / 12), np.sin(2 * np.pi * m / 12)]}
        for L in depths:
            A = {p: T.resid(L, np.arange(N), p) for p in sites}
            Qs = {}
            for p in sites:
                c = A[p][fit].mean(0)
                Ml, *_ = np.linalg.lstsq(codes["line"][fit] - codes["line"][fit].mean(0), A[p][fit] - c, rcond=None)
                Mc, *_ = np.linalg.lstsq(codes["circle"][fit] - codes["circle"][fit].mean(0), A[p][fit] - c, rcond=None)
                ql, _ = np.linalg.qr(Ml.T)
                qc, _ = np.linalg.qr((Mc - (Mc @ ql) @ ql.T).T)
                qr, _ = np.linalg.qr(rng.normal(size=(A[p].shape[1], 2)))
                Qs[p] = {"line": ql, "circle": qc, "random 2-D": qr}
                for d_ in range(5):     # month-related but UNORDERED: a random number per month, encoded like the line
                    rv = np.random.default_rng(100 + d_).normal(size=12)[m][:, None]
                    Mr, *_ = np.linalg.lstsq(rv[fit] - rv[fit].mean(0), A[p][fit] - c, rcond=None)
                    Qs[p][f"random month code {d_}"], _ = np.linalg.qr(Mr.T)
            row = {}
            for k in ["line", "circle", "random 2-D"] + [f"random month code {d_}" for d_ in range(5)]:
                hook = {}
                for p in sites:
                    Ap = A[p][test]; c = A[p][fit].mean(0); Q = Qs[p][k]
                    hook[f"resid{L}@{p}"] = Ap - ((Ap - c) @ Q) @ Q.T
                o = T.run(test, hook=hook)
                mg_o, mg_c = o[:, 1] - o[:, 0], clean[:, 1] - clean[:, 0]
                row[k] = round(float(lmjudge.cmatch(o, clean)), 4)
                # not saturated by a dominant answer: how much of the margin's variation over prompts survives the removal
                row[k + " margin_r2"] = round(float(1 - ((mg_o - mg_c) ** 2).sum() / ((mg_c - mg_c.mean()) ** 2).sum()), 3)
            rmc = [row[f"random month code {d_} margin_r2"] for d_ in range(5)]
            row["random month code, mean margin_r2"] = round(float(np.mean(rmc)), 3)
            row["random month code, min margin_r2"] = round(float(np.min(rmc)), 3)
            res["ablations"][f"{month} @ depth {L}"] = row
            print(f"{month} depth {L}: clear decisions kept / margin R2 kept with line removed {row['line']:.1%} / {row['line margin_r2']:.2f}, "
                  f"circle removed {row['circle']:.1%} / {row['circle margin_r2']:.2f}, random 2-D removed {row['random 2-D']:.1%} / {row['random 2-D margin_r2']:.2f}, "
                  f"random month code (1-D, 5 draws) margin R2 mean {row['random month code, mean margin_r2']:.2f} min {row['random month code, min margin_r2']:.2f}", flush=True)
    (T.dir / "line_vs_circle.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = run(sys.argv[1], [int(d) for d in sys.argv[2:]] or None)
    print(json.dumps({k: r[k] for k in ("model", "behaviour", "line_positions", "constant_baseline")}, indent=1))
