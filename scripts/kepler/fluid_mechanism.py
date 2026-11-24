"""Levels 1 and 3 for the fluid video model: where each physical quantity lives, and how velocity is computed and used.

    python3 scripts/kepler/fluid_mechanism.py

1. Atlas: per-pixel ridge probes (site channels -> quantity) on held-out videos, trained model vs untrained copy.
2. Learned Biot-Savart kernel: the response of the model's internal velocity (the probe at the best site) to a point
   of vorticity, measured by the Jacobian, compared with the exact kernel of u = d(psi)/dy, lap psi = -omega.
3. Velocity swap (interchange intervention): replace the velocity part of state A's activations with state B's at the
   best site. If the model advects with its internal velocity, the change of its output follows
   -(u_B - u_A).grad(omega_A) - beta (v_B - v_A) (the law with the swapped velocity). Control: a random 2-D subspace.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch  # noqa: E402

from kepler.targets.fluid.model import load  # noqa: E402
from kepler.targets.fluid.sim import Spectral  # noqa: E402

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))
D = os.path.join(KDATA, "fluid")
SP = Spectral(64)
QUANT = ["omega", "psi", "u", "v", "omega_x", "omega_y", "lap_omega", "adv", "strain_n", "strain_s", "okubo_weiss",
         "ke", "grad_omega_sq"]
DT = 0.05


def acts_of(model, frames):
    with torch.no_grad():
        _, acts = model(torch.tensor(frames), return_acts=True)
    return [a.numpy().transpose(0, 2, 3, 1).reshape(-1, a.shape[1]) for a in acts]


def quantities(frames):
    out = {q: [] for q in QUANT}
    for w in frames:
        f = SP.fields(w.astype(np.float64))
        for q in QUANT:
            out[q].append(f[q].reshape(-1))
    return {q: np.concatenate(v) for q, v in out.items()}


def ridge(Xtr, Ytr, lam=1e-3):
    A = np.column_stack([Xtr, np.ones(len(Xtr))])
    return np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]) * 1e-6, A.T @ Ytr)


def apply(W, X):
    return np.column_stack([X, np.ones(len(X))]) @ W


def atlas(model, rand, val):
    tr = np.concatenate([val[i, [5, 15, 25, 35]] for i in range(48)])
    te = np.concatenate([val[i, [5, 15, 25, 35]] for i in range(48, 64)])
    qtr, qte = quantities(tr), quantities(te)
    res = {}
    for label, m in (("trained", model), ("control", rand)):
        Atr, Ate = acts_of(m, tr), acts_of(m, te)
        for s in range(len(Atr)):
            Ytr = np.stack([qtr[q] for q in QUANT], 1)
            W = ridge(Atr[s], Ytr)
            P = apply(W, Ate[s])
            for j, q in enumerate(QUANT):
                r2 = 1 - np.mean((qte[q] - P[:, j]) ** 2) / np.var(qte[q])
                res.setdefault(q, {}).setdefault(label, []).append(round(float(r2), 4))
    return res


def velocity_probe(model, val, site):
    tr = np.concatenate([val[i, [5, 15, 25, 35]] for i in range(48)])
    q = quantities(tr)
    A = acts_of(model, tr)[site]
    return ridge(A, np.stack([q["u"], q["v"]], 1))


def biot_savart(model, Wuv, site, val, n_states=8):
    """Average Jacobian of the probed u at the box centre with respect to the input vorticity field."""
    J = np.zeros((2, 64, 64))
    for i in range(n_states):
        w = torch.tensor(val[48 + i, 20][None].astype(np.float32), requires_grad=True)
        _, acts = model(w, return_acts=True)
        h = acts[site][0, :, 32, 32]
        for j in range(2):
            u = (h * torch.tensor(Wuv[:-1, j], dtype=torch.float32)).sum()
            g, = torch.autograd.grad(u, w, retain_graph=True)
            J[j] += g[0].numpy() / n_states
    # exact kernel: response of u (= d psi/dy) and v (= -d psi/dx) at the centre to a unit vorticity at each pixel
    delta = np.zeros((64, 64))
    delta[32, 32] = 1.0
    f = SP.fields(delta)
    # by translation invariance the response at the centre to a point at x equals the field at -x of a centred point
    Ku = np.roll(np.flip(np.roll(f["u"], -32, (0, 1)), (0, 1)), 32, (0, 1))
    Kv = np.roll(np.flip(np.roll(f["v"], -32, (0, 1)), (0, 1)), 32, (0, 1))
    out = {}
    for j, (K, name) in enumerate(((Ku, "u"), (Kv, "v"))):
        out[name] = {"corr": float(np.corrcoef(J[j].reshape(-1), K.reshape(-1))[0, 1]),
                     "scale": float((J[j] * K).sum() / (K * K).sum())}
        # how far does the learned kernel reach? share of |kernel| mass beyond radius 8 pixels
        yy, xx = np.mgrid[0:64, 0:64]
        rr = np.sqrt((yy - 32) ** 2 + (xx - 32) ** 2)
        out[name]["learned_mass_beyond_r8"] = float(np.abs(J[j])[rr > 8].sum() / np.abs(J[j]).sum())
        out[name]["exact_mass_beyond_r8"] = float(np.abs(K)[rr > 8].sum() / np.abs(K).sum())
    np.save("results/kepler/fluid/biot_savart_kernels.npy", np.stack([J[0], J[1], Ku, Kv]))
    return out


def velocity_swap(model, Wuv, site, val, beta, n_pairs=24, seed=0):
    """Interchange intervention on the velocity subspace (the probe's 2-D readout directions)."""
    rng = np.random.default_rng(seed)
    C = Wuv.shape[0] - 1
    B = Wuv[:-1]                                   # [C, 2] readout directions
    # patch: h' = h + B (B^T B)^-1 (v_target - v_current) shifts the readout to the target, minimal change in norm
    G = B @ np.linalg.inv(B.T @ B)
    R = np.linalg.qr(rng.normal(size=(C, 2)))[0]   # random 2-D control subspace
    out = {"model_vs_law_corr": [], "control_corr": [], "rel_change": []}
    for k in range(n_pairs):
        a, b = rng.integers(48, 64, 2)
        wa = val[a, 10 + k % 20].astype(np.float32)
        wb = val[b, 25 - k % 20].astype(np.float32)
        with torch.no_grad():
            _, acts_b = model(torch.tensor(wb[None]), return_acts=True)
            hb = acts_b[site][0].numpy().transpose(1, 2, 0).reshape(-1, C)
            vb = hb @ B + Wuv[-1]
            base = model(torch.tensor(wa[None]))[0].numpy()

            def edit(s, h, use=G, target=vb):
                if s != site:
                    return h
                hh = h[0].numpy().transpose(1, 2, 0).reshape(-1, C)
                cur = hh @ B + Wuv[-1]
                hh = hh + (target - cur) @ use.T
                return torch.tensor(hh.reshape(64, 64, C).transpose(2, 0, 1)[None].astype(np.float32))
            patched = model(torch.tensor(wa[None]), edit=edit)[0].numpy()
            # control: same-size change along a random subspace
            def edit_c(s, h):
                if s != site:
                    return h
                hh = h[0].numpy().transpose(1, 2, 0).reshape(-1, C)
                cur = hh @ B + Wuv[-1]
                delta = (vb - cur) @ G.T
                hh = hh + (delta @ R) @ R.T * (np.linalg.norm(delta) / (np.linalg.norm((delta @ R) @ R.T) + 1e-12))
                return torch.tensor(hh.reshape(64, 64, C).transpose(2, 0, 1)[None].astype(np.float32))
            ctrl = model(torch.tensor(wa[None]), edit=edit_c)[0].numpy()
        fa, fb = SP.fields(wa.astype(np.float64)), SP.fields(wb.astype(np.float64))
        du, dv = fb["u"] - fa["u"], fb["v"] - fa["v"]
        pred = DT * (-(du * fa["omega_x"] + dv * fa["omega_y"]) - beta * dv)
        obs, obs_c = patched - base, ctrl - base
        out["model_vs_law_corr"].append(float(np.corrcoef(obs.reshape(-1), pred.reshape(-1))[0, 1]))
        out["control_corr"].append(float(np.corrcoef(obs_c.reshape(-1), pred.reshape(-1))[0, 1]))
        out["rel_change"].append(float(np.linalg.norm(obs) / (np.linalg.norm(base - wa) + 1e-12)))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()}


def main():
    model, _ = load(os.environ.get("MODEL", "results/kepler/fluid/model.pt"))
    rand, _ = load(os.path.join(D, "ckpt_random.pt"))
    val = np.load(os.path.join(D, "val.npy"))
    at = atlas(model, rand, val)
    for q in QUANT:
        print(f"{q:14s} trained {at[q]['trained']}  control-max {max(at[q]['control']):.3f}")
    uv = [(at["u"]["trained"][s] + at["v"]["trained"][s]) / 2 for s in range(model.n_sites)]
    site = int(np.argmax(uv))
    Wuv = velocity_probe(model, val, site)
    bs = biot_savart(model, Wuv, site, val)
    print(f"velocity best site {site} (R2 {uv[site]:.3f}); learned Biot-Savart kernel:", bs)
    beta = 0.6  # used only to form the law's prediction for the swap test; the swap's verdict is the correlation
    sw = velocity_swap(model, Wuv, site, val, beta)
    print("velocity swap:", sw)
    json.dump({"atlas": at, "velocity_site": site, "biot_savart": bs, "velocity_swap": sw},
              open("results/kepler/fluid/mechanism.json", "w"), indent=1)


if __name__ == "__main__":
    main()
