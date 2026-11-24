"""Level 4 for the fluid video model: read the governing law out of the trained model (K1 held-out verdict).

    python3 scripts/kepler/fluid_law.py [MODEL=results/kepler/fluid/model.pt]

1. The model's implied d(omega)/dt (two-step Richardson estimate) on held-out frames.
2. Law extraction over the 38-term library (the K1-proven extractor, unchanged).
3. Only then the sealed truth is opened: term set, coefficients, verdict.
4. Rollouts: does the extracted law track the model's own rollouts better than the true law does?
5. Out of distribution (stronger, finer fields than any training start): does the same law come back (a law),
   or does the model's implied dynamics stop being that law (a lookup)?
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch  # noqa: E402

from kepler.core.law import extract_law, term_importance  # noqa: E402
from kepler.targets.fluid.library import implied_derivative, term_names, terms  # noqa: E402
from kepler.targets.fluid.model import load  # noqa: E402
from kepler.targets.fluid.sim import Flow, Spectral  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from fluid_known_test import TRUE_TERMS, design, split_frames, true_coefs  # noqa: E402

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))
D = os.path.join(KDATA, "fluid")
DT = 0.05


def model_deriv(model):
    def f(w):
        with torch.no_grad():
            return implied_derivative(model, torch.tensor(w[None].astype(np.float32)), DT)[0].numpy()
    return f


def extract(frames_tr, frames_va, deriv):
    T_tr, Y_tr = design(frames_tr, [deriv(f) for f in frames_tr])
    T_va, Y_va = design(frames_va, [deriv(f) for f in frames_va])
    law = extract_law(T_tr, Y_tr, T_va, Y_va, term_names())
    law["importance"] = term_importance(T_va[:, [term_names().index(t) for t in law["terms"]]], Y_va, law)
    return law


def law_flow(coefs, n=64):
    """A simulator running a law given as library coefficients (only the five physical terms are supported)."""
    supported = {"adv", "lap_omega", "omega", "v", "sin(4y)"}
    if not set(coefs) <= supported or abs(coefs.get("adv", 0) + 1) > 0.5:
        return None
    adv = -coefs.get("adv", -1.0)
    return Flow(n=n, nu=coefs.get("lap_omega", 0.0), mu=-coefs.get("omega", 0.0), F=coefs.get("sin(4y)", 0.0), kf=4,
                beta=-coefs.get("v", 0.0), dt=0.0125, adv=adv)


def rollout_compare(model, extracted, true_flow, states, steps=10):
    """Distance to the model's rollout after `steps` frames for the extracted law and for the true law."""
    out = {"extracted": [], "true": []}
    ef = law_flow(extracted)
    with torch.no_grad():
        for w0 in states:
            wm = torch.tensor(w0[None].astype(np.float32))
            for _ in range(steps):
                wm = model(wm)
            wm = wm[0].numpy().astype(np.float64)
            for key, fl in [("extracted", ef), ("true", true_flow)]:
                if fl is None:
                    continue
                wh = fl.sp.fft(w0.astype(np.float64)) * fl.sp.mask
                wh = fl.step_hat(wh, steps * int(round(DT / fl.dt)))
                ws = fl.sp.ifft(wh)
                out[key].append(float(np.linalg.norm(ws - wm) / np.linalg.norm(wm - w0)))
    return {k: (float(np.mean(v)) if v else None) for k, v in out.items()}


def main():
    path = os.environ.get("MODEL", "results/kepler/fluid/model.pt")
    model, extra = load(path)
    val = np.load(os.path.join(D, "val.npy"))
    ood = np.load(os.path.join(D, "ood.npy"))
    tr_idx, va_idx = split_frames(val)
    fr_tr = [val[i, t] for i, t in tr_idx]
    fr_va = [val[i, t] for i, t in va_idx]
    deriv = model_deriv(model)
    law = extract(fr_tr, fr_va, deriv)
    print("extracted law:", {k: round(v, 5) for k, v in law["coefs"].items()}, "R2", round(law["r2_val"], 4))

    # one-step prediction quality of the model on held-out pairs (context for the verdict)
    with torch.no_grad():
        x = torch.tensor(val[48:, 5:35:5].reshape(-1, 64, 64))
        y = torch.tensor(val[48:, 6:36:5].reshape(-1, 64, 64))
        rel = float(((model(x) - y) ** 2).sum().sqrt() / ((y - x) ** 2).sum().sqrt())

    # out of distribution: frames 0..10 of strong, fine-scale videos (split by video)
    o_tr = [ood[i, t] for i in range(24) for t in (0, 3, 6, 9)]
    o_va = [ood[i, t] for i in range(24, 32) for t in (0, 3, 6, 9)]
    law_ood = extract(o_tr, o_va, deriv)
    print("OOD law:", {k: round(v, 5) for k, v in law_ood["coefs"].items()}, "R2", round(law_ood["r2_val"], 4))

    # --- only now: the sealed truth ---
    p = json.load(open(os.path.join(D, "sealed_params.json")))
    tc = true_coefs(p)
    true_flow = Flow(n=p["n"], nu=p["nu"], mu=p["mu"], F=p["F"], kf=p["kf"], beta=p["beta"], dt=p["dt"])
    exact = set(law["terms"]) == TRUE_TERMS
    verdict = exact and law["r2_val"] >= 0.9
    dev = {k: (law["coefs"].get(k, 0.0) - v) / abs(v) for k, v in tc.items()}
    roll = rollout_compare(model, law["coefs"], true_flow, [val[i, 20] for i in range(48, 64)])
    # true law's fit to the model's implied derivative on the same rows, for comparison
    T_va, Y_va = design(fr_va, [deriv(f) for f in fr_va])
    names = term_names()
    yt = sum(T_va[:, names.index(k)] * v for k, v in tc.items())
    r2_true = 1 - float(np.mean((Y_va - yt) ** 2) / np.var(Y_va))
    res = {"model": path, "steps": extra.get("steps"), "one_step_rel_err_of_change": rel,
           "law": law, "law_ood": law_ood, "true_coefs": tc, "relative_deviation": dev,
           "r2_of_true_law_on_model": r2_true, "rollout_distance": roll,
           "verdict": "PASS" if verdict else "FAIL", "exact_term_set": exact}
    os.makedirs("results/kepler/fluid", exist_ok=True)
    with open("results/kepler/fluid/k1_verdict.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"one-step error of the change: {rel:.3f}")
    print("true:", tc)
    print("relative deviation:", {k: round(v, 3) for k, v in dev.items()})
    print(f"R2 of the TRUE law on the model's implied dynamics: {r2_true:.4f}")
    print("rollout distance (10 frames) to the model's own rollout:", roll)
    print("K1 VERDICT:", res["verdict"], "(exact term set:", exact, ", R2", round(law["r2_val"], 4), ")")


if __name__ == "__main__":
    main()
