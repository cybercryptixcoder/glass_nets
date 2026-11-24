"""K1 known-answer test for the law extractor (written before it runs on any trained model).

(a)  exact simulator derivative on held-out frames -> exactly the true terms, every coefficient within 2%
(a') reported: the same with the Richardson two-step estimate on the exact solver (method bias of the estimator)
(b)  must fail: targets shuffled across frames -> R^2 < 0.1
(c)  must fail: an untrained video model -> NOT (true term set and R^2 >= 0.9)

Only this test opens the sealed parameters (to build the ground truth); the extractor never sees them.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch  # noqa: E402

from kepler.core.law import extract_law  # noqa: E402
from kepler.targets.fluid.library import implied_derivative, term_names, terms  # noqa: E402
from kepler.targets.fluid.model import load  # noqa: E402
from kepler.targets.fluid.sim import Flow  # noqa: E402

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))
D = os.path.join(KDATA, "fluid")
TRUE_TERMS = {"adv", "lap_omega", "omega", "sin(4y)", "v"}


def true_coefs(p):
    return {"adv": -1.0, "lap_omega": p["nu"], "omega": -p["mu"], "sin(4y)": p["F"], "v": -p["beta"]}


def design(frames, ys, stride=2):
    """Stack library rows (every `stride`-th pixel) for a list of frames and targets."""
    from kepler.targets.fluid.sim import Spectral
    sp = Spectral(frames[0].shape[-1])
    sel = np.zeros((sp.n, sp.n), dtype=bool)
    sel[::stride, ::stride] = True
    sel = sel.reshape(-1)
    T = np.concatenate([terms(sp, f.astype(np.float64))[sel] for f in frames])
    y = np.concatenate([yy.astype(np.float64).reshape(-1)[sel] for yy in ys])
    return T, y


def split_frames(val, per_traj=6):
    tr_idx, va_idx = [], []
    for i in range(val.shape[0]):
        for t in np.linspace(2, val.shape[1] - 3, per_traj).astype(int):
            (tr_idx if i < 48 else va_idx).append((i, t))
    return tr_idx, va_idx


def run_extract(val, deriv_fn, shuffle=False, seed=0):
    tr_idx, va_idx = split_frames(val)
    fr_tr = [val[i, t] for i, t in tr_idx]
    fr_va = [val[i, t] for i, t in va_idx]
    y_tr = [deriv_fn(f) for f in fr_tr]
    y_va = [deriv_fn(f) for f in fr_va]
    if shuffle:
        rng = np.random.default_rng(seed)
        y_tr = [y_tr[k] for k in rng.permutation(len(y_tr))]
        y_va = [y_va[k] for k in rng.permutation(len(y_va))]
    T_tr, Y_tr = design(fr_tr, y_tr)
    T_va, Y_va = design(fr_va, y_va)
    return extract_law(T_tr, Y_tr, T_va, Y_va, term_names())


def main():
    p = json.load(open(os.path.join(D, "sealed_params.json")))
    val = np.load(os.path.join(D, "val.npy"))
    flow = Flow(n=p["n"], nu=p["nu"], mu=p["mu"], F=p["F"], kf=p["kf"], beta=p["beta"], dt=p["dt"])
    tc = true_coefs(p)
    res = {}

    law = run_extract(val, lambda f: flow.rhs(f.astype(np.float64)))
    errs = {k: abs(law["coefs"].get(k, 0.0) - v) / abs(v) for k, v in tc.items()}
    ok_a = set(law["terms"]) == TRUE_TERMS and max(errs.values()) < 0.02
    res["a"] = {"pass": ok_a, "terms": law["terms"], "coefs": law["coefs"], "rel_err": errs, "r2": law["r2_val"]}
    print("(a) exact derivative:", "PASS" if ok_a else "FAIL", law["terms"], {k: round(v, 5) for k, v in law["coefs"].items()},
          "max rel err", round(max(errs.values()), 4), "R2", round(law["r2_val"], 5))

    nsub = int(round(0.05 / flow.dt))

    def sim_step(f):
        wh = flow.sp.fft(f.astype(np.float64)) * flow.sp.mask
        return flow.sp.ifft(flow.step_hat(wh, nsub))

    def richardson(f):
        m1 = sim_step(f)
        m2 = sim_step(m1)
        return (4 * m1 - m2 - 3 * f) / (2 * 0.05)

    law_r = run_extract(val, richardson)
    errs_r = {k: abs(law_r["coefs"].get(k, 0.0) - v) / abs(v) for k, v in tc.items()}
    res["a_prime"] = {"terms": law_r["terms"], "coefs": law_r["coefs"], "rel_err": errs_r, "r2": law_r["r2_val"]}
    print("(a') Richardson on exact solver:", law_r["terms"], {k: round(v, 5) for k, v in law_r["coefs"].items()},
          "max rel err", round(max(errs_r.values()), 4), "R2", round(law_r["r2_val"], 5))

    law_s = run_extract(val, lambda f: flow.rhs(f.astype(np.float64)), shuffle=True)
    ok_b = law_s["r2_val"] < 0.1
    res["b"] = {"pass": ok_b, "r2": law_s["r2_val"], "terms": law_s["terms"]}
    print("(b) shuffled targets:", "PASS" if ok_b else "FAIL", "R2", round(law_s["r2_val"], 4), "terms", law_s["terms"][:8])

    rand, _ = load(os.path.join(D, "ckpt_random.pt"))
    with torch.no_grad():
        def rand_deriv(f):
            return implied_derivative(rand, torch.tensor(f[None].astype(np.float32)), 0.05)[0].numpy()
        law_c = run_extract(val, rand_deriv)
    bad = set(law_c["terms"]) == TRUE_TERMS and law_c["r2_val"] >= 0.9
    res["c"] = {"pass": not bad, "terms": law_c["terms"], "r2": law_c["r2_val"]}
    print("(c) untrained model:", "PASS" if not bad else "FAIL", "R2", round(law_c["r2_val"], 4), "n_terms", len(law_c["terms"]))

    res["all_pass"] = ok_a and ok_b and not bad
    os.makedirs("results/kepler/fluid", exist_ok=True)
    with open("results/kepler/fluid/k1_known_test.json", "w") as f:
        json.dump(res, f, indent=1)
    print("ALL PASS" if res["all_pass"] else "SOME FAIL")


if __name__ == "__main__":
    main()
