"""The physics term library for law extraction, and the implied time derivative of a video model.

Every candidate term is computed exactly (spectrally) from a vorticity field. The library holds the true terms of the
hidden law among many distractors; the engine never knows which are true.
"""
import numpy as np
import torch

from .sim import Spectral

TRIG_K = range(1, 7)


def term_names():
    names = ["adv", "lap_omega", "bilap_omega", "omega", "omega^2", "omega^3", "u", "v", "psi", "omega_x", "omega_y",
             "ke", "omega*ke", "1"]
    for k in TRIG_K:
        names += [f"sin({k}y)", f"cos({k}y)", f"sin({k}x)", f"cos({k}x)"]
    return names


def terms(sp: Spectral, w):
    """w: [n, n] -> Theta [n*n, n_terms] (float64)."""
    f = sp.fields(w)
    cols = [f["adv"], f["lap_omega"], f["bilap_omega"], f["omega"], f["omega"] ** 2, f["omega"] ** 3, f["u"], f["v"],
            f["psi"], f["omega_x"], f["omega_y"], f["ke"], f["omega"] * f["ke"], np.ones_like(w)]
    for k in TRIG_K:
        cols += [np.sin(k * sp.Y), np.cos(k * sp.Y), np.sin(k * sp.X), np.cos(k * sp.X)]
    return np.stack([c.reshape(-1) for c in cols], 1)


TRUE_SIGNS = {"adv": "-1 (advection)", "lap_omega": "+nu (viscosity)", "omega": "-mu (drag)",
              "sin(4y)": "+F (forcing)", "v": "-beta (beta effect)"}


@torch.no_grad()
def implied_derivative(model, w, dt, second_order=True):
    """The model's implied d(omega)/dt at frames w [B, n, n] (torch). Second order (Richardson):
    (4 M(w) - M(M(w)) - 3 w) / (2 dt) cancels the O(dt) error of a single step."""
    m1 = model(w)
    if not second_order:
        return (m1 - w) / dt
    m2 = model(m1)
    return (4 * m1 - m2 - 3 * w) / (2 * dt)
