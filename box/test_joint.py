"""Known cases for the vector-site checker's interventions when two parts share a site (CHANGES.md C2, written before
the change). A toy network holds A and B in a code whose directions are not orthogonal, z = (A, A + B, noise), and
outputs A * B. The true account locates A along (1, 0, ...) and B along (-1, 1, 0, ...)/sqrt 2, both exact readouts.
- The translation error itself: an intervention on A must leave B's readout unchanged. v0 (orthogonal swap along the
  part's own direction) moves it; v1 (the smallest change that sets the part's readout and holds the other parts at
  the site) must not.
- The true account is an exact mechanism, so its interventions must score at least 0.99, and it must be certified.
  (As first written, this test also expected v0 to reject the true account. v0 certifies it, at 0.10 and 0.11 against
  a line of 0.08: the defect shows in the scores, not in v0's verdict, on this toy. Recorded in CHANGES.md C2.)
- Locations exchanged, a wrong definition and a random direction must be rejected.
Run: python3 -m box.test_joint"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box.blueprint import Blueprint, Part  # noqa: E402
from box.verify_vec import VecChecker  # noqa: E402

rng = np.random.default_rng(0)
N, D = 300, 8
DATA = np.column_stack([rng.integers(0, 10, N), rng.integers(0, 10, N), rng.integers(0, 10, (N, D - 2))]).astype(float)
WX = np.eye(D); WX[1, 0] = 1.0                       # z0 = A, z1 = A + B, z2.. = noise inputs
WH = np.zeros((2, D)); WH[0, 0] = 1.0; WH[1, 0] = -1.0; WH[1, 1] = 1.0   # the network decodes A and B
WO = np.array([[1.0]])


class Toy:
    def __init__(self, target, weights=None):
        self.target = target
        w = weights or {"W_x": WX, "W_h": WH, "W_o": WO}
        self.Wx, self.Wh, self.Wo = (np.asarray(w[k], float) for k in ("W_x", "W_h", "W_o"))

    def sites_for(self, _T):
        return ["z", "out"]

    def run(self, Xi, edits=None, return_sites=False):
        edits = edits or {}
        X = self.target[np.asarray(Xi, int).ravel()]
        z = X @ self.Wx.T
        if "z" in edits:
            z = edits["z"](z)
        h = z @ self.Wh.T
        out = (h[:, 0] * h[:, 1] * self.Wo[0, 0])[:, None]
        if "out" in edits:
            out = edits["out"](out)
        return (out, {"z": z, "out": out}) if return_sites else out

    def outputs(self, Xi, edits=None):
        return np.rint(self.run(Xi, edits)[:, 0]).astype(int)


def account(dA, dB, verdict="(mul @A @B)", name=""):
    unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)  # noqa: E731
    parts = [Part("A", "rule", "Num", "z", basis=unit(dA), encoding="numerical", scale=1.0, definition="(at x 0)"),
             Part("B", "rule", "Num", "z", basis=unit(dB), encoding="numerical", scale=1.0, definition="(at x 1)"),
             Part("out", "rule", "Num", "out", basis=np.array([1.0]), encoding="numerical", definition=verdict,
                  upstream=["A", "B"])]
    return Blueprint("seq", parts, "out", name=name)


eA = np.eye(D)[0]; eB = (np.eye(D)[1] - np.eye(D)[0])
cases = [("true account", account(eA, eB), None),
         ("must fail: locations exchanged", account(eB, eA), None),
         ("must fail: wrong definition", account(eA, eB, "(add @A @B)"), None),
         ("must fail: random direction for A", account(rng.normal(size=D), eB), None)]


def readout_moved(chk, bp, part, other):
    """Share of decisive pairs on which intervening on `part` changes the readout of `other` (same site)."""
    p, q = bp.part(part), bp.part(other)
    u, w = np.asarray(p.basis, float), np.asarray(q.basis, float)
    a, b = chk.pa, chk.pb
    h = chk.sites[p.site][a]
    fn = chk._swap_joint(bp, p, u, b) if chk.joint else chk._swap(u, p.site, b)
    return float(np.mean(np.abs(fn(h) @ w - h @ w) > 1e-9))


def main():
    ok_all = True
    for joint in (False, True):
        v = "v1" if joint else "v0"
        chk = VecChecker(Toy(DATA), np.arange(N)[:, None], seed=0, n_pairs=400, control_reps=200,
                         ctx_fn=lambda Xi: {"x": DATA[np.asarray(Xi, int).ravel()]}, joint=joint)
        true_bp = cases[0][1]
        moved = readout_moved(chk, true_bp, "A", "B")
        ok = moved == 0.0
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {v} intervening on A leaves B's readout unchanged: moved on {moved:.0%} of pairs")
        for name, bp, _want in cases:
            rep = chk.certify(bp, run_alone_line=1.0 - 1e-9)
            iia = {c[0]: float(c[2].split()[1]) for c in rep["checks"] if c[0].startswith("interventions")}
            if name == "true account":
                ok = rep["verdict"] == "certified" and min(iia.values()) >= 0.99
                want = "certified, every intervention at least 0.99"
            else:
                ok = rep["verdict"] == "rejected"
                want = "rejected"
            ok_all &= ok
            detail = "; ".join(f"{c[0]} {c[2]}" for c in rep["checks"] if c[0].startswith("interventions"))
            print(f"{'PASS' if ok else 'FAIL'}  {v} {name}: {rep['verdict']} (want {want}); {detail}")
    print("all pass" if ok_all else "SOME FAIL (expected under v0 only)")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
