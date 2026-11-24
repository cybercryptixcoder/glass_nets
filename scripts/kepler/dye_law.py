"""K2: the smoke video model. Find the hidden velocity inside it, then read its transport law using that velocity.

    python3 scripts/kepler/dye_law.py known     # (a) exact dye derivative + true velocity; (b) shuffled velocity
    python3 scripts/kepler/dye_law.py verdict   # probes for (u, v) at every site vs untrained copy; law from the model

Videos: 48 held-out smoke videos (val_dye.npy) with their vorticity (val_omega.npy, never shown to the model; used to
train the velocity probe on videos 0-35 and to build the known-answer test). Law rows come from videos 36-47 only.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch  # noqa: E402

from kepler.core.law import extract_law, term_importance  # noqa: E402
from kepler.targets.fluid.model import load  # noqa: E402
from kepler.targets.fluid.sim import DyeFlow, Spectral  # noqa: E402

KDATA = os.environ.get("KDATA", os.path.expanduser("~/kdata"))
D = os.path.join(KDATA, "dye")
DT, K = 0.05, 3
TRUE = {"adv_c", "lap_c"}
SP = Spectral(64)


def velocity(w):
    f = SP.fields(w.astype(np.float64))
    return f["u"], f["v"]


def dye_terms(c, u, v):
    """Library for d(c)/dt given a dye frame and a velocity field (true or read out of the model)."""
    ch = SP.fft(c.astype(np.float64)) * SP.mask
    cx, cy = SP.ifft(1j * SP.KX * ch), SP.ifft(1j * SP.KY * ch)
    lap = SP.ifft(-SP.K2 * ch)
    bil = SP.ifft(SP.K2 ** 2 * ch)
    adv = SP.ifft(SP.fft(u * cx + v * cy) * SP.mask)
    cc = SP.ifft(ch)
    cols = {"adv_c": adv, "lap_c": lap, "bilap_c": bil, "c": cc, "c^2": cc ** 2, "c_x": cx, "c_y": cy, "u": u, "v": v,
            "u*c": u * cc, "v*c": v * cc, "|grad c|^2": cx ** 2 + cy ** 2, "1": np.ones_like(cc)}
    for k in (1, 2, 3, 4):
        cols[f"sin({k}y)"], cols[f"cos({k}y)"] = np.sin(k * SP.Y), np.cos(k * SP.Y)
    names = list(cols)
    return np.stack([cols[n].reshape(-1) for n in names], 1), names


def rows(frames_c, vel, ys, stride=2):
    sel = np.zeros((64, 64), bool)
    sel[::stride, ::stride] = True
    sel = sel.reshape(-1)
    T, Y = [], []
    for c, (u, v), y in zip(frames_c, vel, ys):
        th, names = dye_terms(c, u, v)
        T.append(th[sel])
        Y.append(y.reshape(-1)[sel])
    return np.concatenate(T), np.concatenate(Y), names


def pick(vids, ts=(6, 12, 18, 24)):
    return [(i, t) for i in vids for t in ts]


def known():
    p = json.load(open(os.path.join(D, "sealed_params.json")))
    flow = DyeFlow(n=p["n"], nu=p["nu"], mu=p["mu"], F=p["F"], kf=p["kf"], beta=p["beta"], dt=p["dt"], kappa=p["kappa"])
    dye, om = np.load(os.path.join(D, "val_dye.npy")), np.load(os.path.join(D, "val_omega.npy"))
    tr, va = pick(range(36)), pick(range(36, 48))
    res = {}
    for label, shuffle in (("a", False), ("b", True)):
        sets = []
        for idx in (tr, va):
            cs = [dye[i, t] for i, t in idx]
            vel = [velocity(om[i, t]) for i, t in idx]
            ys = [flow.dye_rhs(om[i, t].astype(np.float64), dye[i, t].astype(np.float64)) for i, t in idx]
            if shuffle:
                perm = np.random.default_rng(5).permutation(len(vel))
                vel = [vel[k] for k in perm]
            sets.append(rows(cs, vel, ys))
        (Ttr, Ytr, names), (Tva, Yva, _) = sets
        law = extract_law(Ttr, Ytr, Tva, Yva, names)
        res[label] = law
        print(f"({label})", "shuffled velocity" if shuffle else "true velocity", {k: round(v, 5) for k, v in law["coefs"].items()},
              "R2", round(law["r2_val"], 4))
    tc = {"adv_c": -1.0, "lap_c": p["kappa"]}
    errs = {k: abs(res["a"]["coefs"].get(k, 0) - v) / abs(v) for k, v in tc.items()}
    ok_a = set(res["a"]["terms"]) == TRUE and max(errs.values()) < 0.02
    adv_b = res["b"]["coefs"].get("adv_c", 0.0)
    ok_b = res["b"]["r2_val"] < 0.3 and not (abs(adv_b + 1) < 0.2)
    print("(a)", "PASS" if ok_a else "FAIL", "max rel err", round(max(errs.values()), 4), "| (b)", "PASS" if ok_b else "FAIL")
    os.makedirs("results/kepler/dye", exist_ok=True)
    json.dump({"a": res["a"], "b": res["b"], "pass_a": ok_a, "pass_b": ok_b}, open("results/kepler/dye/k2_known_test.json", "w"), indent=1)


def site_acts(model, x):
    with torch.no_grad():
        _, acts = model(x, return_acts=True)
    return [a.numpy() for a in acts]


def probe_velocity(model, dye, om, vids_tr, vids_te, ts=(6, 12, 18, 24)):
    """Per-pixel ridge from the C channels at each site to (u, v). Returns R2 per site and the fitted maps."""
    def collect(vids):
        X = [[] for _ in range(model.n_sites)]
        U = []
        for i in vids:
            x = torch.tensor(np.stack([dye[i, t - K + 1:t + 1] for t in ts]))
            acts = site_acts(model, x)
            for s in range(model.n_sites):
                X[s].append(acts[s].transpose(0, 2, 3, 1).reshape(-1, acts[s].shape[1]))
            for t in ts:
                u, v = velocity(om[i, t])
                U.append(np.stack([u.reshape(-1), v.reshape(-1)], 1))
        return [np.concatenate(a) for a in X], np.concatenate(U)
    Xtr, Utr = collect(vids_tr)
    Xte, Ute = collect(vids_te)
    out = []
    for s in range(model.n_sites):
        A = np.column_stack([Xtr[s], np.ones(len(Xtr[s]))])
        W = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ Utr)
        P = np.column_stack([Xte[s], np.ones(len(Xte[s]))]) @ W
        r2 = [1 - np.mean((Ute[:, j] - P[:, j]) ** 2) / np.var(Ute[:, j]) for j in range(2)]
        out.append({"site": s, "r2_u": float(r2[0]), "r2_v": float(r2[1]), "W": W})
    return out


def verdict():
    model, extra = load(os.path.join(D, "model.pt"))
    rand, _ = load(os.path.join(D, "model_random.pt"))
    dye, om = np.load(os.path.join(D, "val_dye.npy")), np.load(os.path.join(D, "val_omega.npy"))
    vtr, vte = range(0, 36), range(36, 48)
    pt = probe_velocity(model, dye, om, vtr, vte)
    pc = probe_velocity(rand, dye, om, vtr, vte)
    for a, b in zip(pt, pc):
        print(f"site {a['site']}: velocity R2 u {a['r2_u']:.3f} v {a['r2_v']:.3f} | untrained u {b['r2_u']:.3f} v {b['r2_v']:.3f}")
    best = max(pt, key=lambda e: (e["r2_u"] + e["r2_v"]) / 2)
    bs = best["site"]
    r2_best = (best["r2_u"] + best["r2_v"]) / 2
    r2_ctrl = (pc[bs]["r2_u"] + pc[bs]["r2_v"]) / 2
    c1 = r2_best >= 0.5 and r2_best - r2_ctrl >= 0.3
    W = best["W"]

    def read_velocity(i, t, frames=None):
        x = torch.tensor((frames if frames is not None else dye[i, t - K + 1:t + 1])[None])
        a = site_acts(model, x)[bs][0].transpose(1, 2, 0).reshape(-1, W.shape[0] - 1)
        P = np.column_stack([a, np.ones(len(a))]) @ W
        return P[:, 0].reshape(64, 64), P[:, 1].reshape(64, 64)

    def implied(i, t):
        with torch.no_grad():
            f = torch.tensor(dye[i, t - K + 1:t + 1][None])
            m1 = model(f)
            f2 = torch.cat([f[:, 1:], m1[:, None]], 1)
            m2 = model(f2)
            return ((4 * m1 - m2 - 3 * f[:, -1]) / (2 * DT))[0].numpy()

    # law rows from held-out videos only; velocity read out of the model (never from the simulator)
    idx_tr, idx_va = pick(range(36, 44)), pick(range(44, 48))
    sets = []
    for idx in (idx_tr, idx_va):
        sets.append(rows([dye[i, t] for i, t in idx], [read_velocity(i, t) for i, t in idx], [implied(i, t) for i, t in idx]))
    (Ttr, Ytr, names), (Tva, Yva, _) = sets
    law = extract_law(Ttr, Ytr, Tva, Yva, names)
    law["importance"] = term_importance(Tva[:, [names.index(t) for t in law["terms"]]], Yva, law)
    c2 = set(law["terms"]) == TRUE
    c3 = law["r2_val"] >= 0.8
    # for comparison: the same extraction with the TRUE velocity
    sets_true = []
    for idx in (idx_tr, idx_va):
        sets_true.append(rows([dye[i, t] for i, t in idx], [velocity(om[i, t]) for i, t in idx], [implied(i, t) for i, t in idx]))
    law_true_vel = extract_law(sets_true[0][0], sets_true[0][1], sets_true[1][0], sets_true[1][1], names)
    p = json.load(open(os.path.join(D, "sealed_params.json")))
    print(f"best site {bs}: velocity R2 {r2_best:.3f} (untrained {r2_ctrl:.3f}) ->", "OK" if c1 else "fails (1)")
    print("law with the model's own velocity:", {k: round(v, 5) for k, v in law["coefs"].items()}, "R2", round(law["r2_val"], 4))
    print("law with the true velocity:", {k: round(v, 5) for k, v in law_true_vel["coefs"].items()}, "R2", round(law_true_vel["r2_val"], 4))
    print("true: adv_c -1, lap_c", p["kappa"])
    v = "PASS" if (c1 and c2 and c3) else "FAIL"
    print("K2 VERDICT:", v, {"1_velocity_decodable": c1, "2_exact_terms": c2, "3_r2": c3})
    for e in pt + pc:
        e.pop("W", None)
    json.dump({"probe_trained": pt, "probe_untrained": pc, "best_site": bs, "law_model_velocity": law,
               "law_true_velocity": law_true_vel, "true": {"adv_c": -1.0, "lap_c": p["kappa"]}, "verdict": v,
               "criteria": {"1": c1, "2": c2, "3": c3}}, open("results/kepler/dye/k2_verdict.json", "w"), indent=1)


if __name__ == "__main__":
    {"known": known, "verdict": verdict}[sys.argv[1]]()
