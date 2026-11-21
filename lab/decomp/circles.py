"""D1 stage 1: find circles in a set of vectors (rows of an embedding or columns of an unembedding), blind.

No order of the symbols is assumed. A "circle plane" is a 2-D projection in which the symbols sit at (nearly) equal
radius; their angles then sit on a lattice of N equally spaced points. Each symbol gets a coordinate j in Z_N.

find_planes(X)      -> list of planes (2-D orthonormal bases, radius uniformity, energy)
read_lattice(X, Q)  -> N, coordinates j (int, -1 for symbols off the circle), fit quality
"""
import numpy as np
import torch


def _whiten(X, keep=0.999, full=False, ratio=0.2):
    """Centre and reduce. full=True also equalises the singular values: if X = F B with F the symbols' character values
    (orthogonal columns over the symbols) and B any full-rank mixing, the rows of the whitened matrix are F rotated by
    an orthogonal matrix, so every character plane becomes an exact circle even when B mixes the planes (as in an
    unembedding, whose frequency directions are far from orthogonal)."""
    X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    if full:
        k = int((S / S[0] > ratio).sum())
        k = max(2, min(k, 24))
        return U[:, :k] * np.sqrt(len(X)), np.diag(1 / S[:k]) @ Vt[:k] * np.sqrt(len(X)), S
    e = np.cumsum(S ** 2) / (S ** 2).sum()
    k = int(np.searchsorted(e, keep) + 1)
    k = max(2, min(k, 24))
    return U[:, :k] * S[:k], Vt[:k], S


def _uniformity(Y):
    r2 = (Y ** 2).sum(1)
    return r2.var() / r2.mean() ** 2


def find_planes(X, restarts=24, iters=300, max_planes=12, cv_max=0.08, min_energy=0.02, seed=0, mask=None, ellipse=False, full=False, ratio=0.2):
    """Greedy: best radius-uniform plane, project it out, repeat. mask = symbols to use (others ignored when fitting:
    lets a symbol that sits at the centre, like 0 under multiplication, not spoil the fit)."""
    g = torch.Generator().manual_seed(seed)
    Z, basis, S = _whiten(X, full=full, ratio=ratio)
    total = (Z ** 2).sum()
    use = np.ones(len(X), bool) if mask is None else mask.copy()
    planes = []
    R = Z.copy()
    for _ in range(max_planes):
        if (R ** 2).sum() < min_energy * total:
            break
        Rt = torch.tensor(R[use], dtype=torch.float64)
        best = None
        for _r in range(restarts):
            A = torch.randn(R.shape[1], 2, generator=g, dtype=torch.float64).requires_grad_(True)
            opt = torch.optim.Adam([A], lr=0.05)
            for _i in range(iters):
                Q, _ = torch.linalg.qr(A)
                Y = Rt @ Q
                if ellipse:                       # in-plane whitening: an ellipse with evenly spaced points becomes a circle
                    C = Y.T @ Y / len(Y) + 1e-9 * torch.eye(2, dtype=Y.dtype)
                    L = torch.linalg.cholesky(torch.linalg.inv(C))
                    Y = Y @ L * C.trace().sqrt() / 2 ** 0.5
                r2 = (Y ** 2).sum(1)
                loss = r2.var() / r2.mean() ** 2 - 0.05 * r2.mean() / (Rt ** 2).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            Q, _ = torch.linalg.qr(A.detach())
            Qn = Q.numpy(); Y = R[use] @ Qn
            M = np.eye(2)
            if ellipse:
                C = Y.T @ Y / len(Y); M = np.linalg.cholesky(np.linalg.inv(C + 1e-12 * np.eye(2)))
            cv = float(_uniformity(Y @ M)); en = float((Y ** 2).sum() / total)
            if en > min_energy and (best is None or cv < best[0]):
                best = (cv, en, Qn, M)
        if best is None or best[0] > cv_max:
            break
        cv, en, Qn, M = best
        planes.append({"cv": cv, "energy": en, "Q": basis.T @ Qn @ M, "Qo": basis.T @ Qn})   # back to the original space (d x 2)
        R = R - (R @ Qn) @ Qn.T
    return planes


def read_lattice(X, Q, n_max=None, mask=None, center=None):
    """Angles of the symbols in plane Q; the lattice period N is the smallest m with |mean exp(i m theta)| ~ 1."""
    mu = X.mean(0) if center is None else center
    Y = (X - mu) @ Q
    th = np.arctan2(Y[:, 1], Y[:, 0]); rad = np.hypot(Y[:, 0], Y[:, 1])
    on = rad > 0.5 * np.median(rad)
    if mask is not None:
        on &= mask
    # lattice period from the gaps between neighbouring angles: points closer than a third of the typical gap are one
    # lattice point (a 2-to-1 map puts two symbols on the same point); the typical gap between distinct points is 2*pi/N
    a = np.sort(th[on]); gaps = np.diff(np.r_[a, a[0] + 2 * np.pi])
    big = gaps[gaps > 0.25 * np.median(gaps)] if len(gaps) else gaps
    cand = []
    if len(big) >= 2:
        n0 = 2 * np.pi / np.median(big)
        for N in sorted({max(2, int(round(n0)) + d) for d in range(-3, 4)}):
            phi = np.angle(np.exp(1j * N * th[on]).mean()) / N
            r = np.angle(np.exp(1j * N * (th[on] - phi))) / N          # residual to the nearest lattice point (radians)
            cand.append((float(np.abs(r).mean() * N / np.pi), N, phi))   # 0 = on the lattice, 0.5 = random
    if not cand:
        return {"N": None, "j": None, "lattice_fit": 0.0, "on": on}
    err, N, phi = min(cand)
    j = np.round(N * (th - phi) / (2 * np.pi)).astype(int) % N
    j = np.where(on, j, -1)
    return {"N": int(N), "j": j, "lattice_fit": 1 - 2 * err, "on": on, "distinct": int(len(set(j[on].tolist())))}


def units(N):
    from math import gcd
    return [k for k in range(1, N) if gcd(k, N) == 1]


def align(j_ref, j_other, N_ref, N_other):
    """Best map j_other = k * j_ref + c (mod N_other) when N_other divides N_ref (a character of a coarser circle)."""
    if N_ref % N_other:
        return None
    ok = (j_ref >= 0) & (j_other >= 0)
    best = (0, None, None)
    for k in range(N_other):
        for c in range(N_other):
            acc = float(((k * j_ref[ok] + c) % N_other == j_other[ok]).mean())
            if acc > best[0]:
                best = (acc, k, c)
    return {"acc": best[0], "k": best[1], "c": best[2]}


def joint_decode(X, planes, N, center=None, iters=6, use=None):
    """Joint coordinates from several circle planes that share one lattice Z_N.

    Plane i shows angle theta_i(t) = 2*pi*m_i*s(t)/N + phi_i + noise. One plane alone is too noisy to pin s(t) (its
    angular jitter is close to the lattice spacing); several planes with different multipliers m_i pin it, the way a
    network's own readout combines frequencies. The first plane fixes the scale (m_0 = 1).
    Returns s (int, -1 off-circle), m (multipliers), phi, per-symbol margin, coherence per plane."""
    mu = X.mean(0) if center is None else center
    TH, RAD = [], []
    for pl in planes:
        Y = (X - mu) @ pl["Q"]; TH.append(np.arctan2(Y[:, 1], Y[:, 0])); RAD.append(np.hypot(Y[:, 0], Y[:, 1]))
    TH = np.array(TH); RAD = np.array(RAD)
    on = (RAD > 0.5 * np.median(RAD, axis=1, keepdims=True)).all(0) if use is None else use
    P = len(planes)
    s = np.round(N * TH[0] / (2 * np.pi)).astype(int) % N
    m = np.zeros(P, int); m[0] = 1; phi = np.zeros(P); coh = np.zeros(P)
    grid = 2 * np.pi * np.arange(N) / N
    for _ in range(iters):
        for i in range(P):
            ks = range(N) if i else [1]
            best = (-1, 1, 0.0)
            for k in ks:
                z = np.exp(1j * (TH[i][on] - 2 * np.pi * k * s[on] / N)).mean()
                if abs(z) > best[0]:
                    best = (abs(z), k, np.angle(z))
            coh[i], m[i], phi[i] = best
        # decode every symbol with all planes, weighted by coherence
        w = coh / coh.sum()
        score = np.zeros((len(X), N))
        for i in range(P):
            score += w[i] * np.cos(TH[i][:, None] - (m[i] * grid[None, :] + phi[i]))
        s_new = score.argmax(1)
        srt = np.sort(score, 1)
        margin = srt[:, -1] - srt[:, -2]
        if np.array_equal(s_new, s):
            break
        s = s_new
    s = np.where(on, s, -1)
    return {"s": s, "m": m, "phi": phi, "coh": coh, "margin": margin, "on": on, "N": N,
            "distinct": int(len(set(s[on].tolist()))), "n_on": int(on.sum())}


def decode_auto(X, planes, n_range=None, center=None):
    """Choose N by the joint coherence (all planes) over candidate lattice sizes."""
    mu = X.mean(0) if center is None else center
    RAD = np.array([np.hypot(*((X - mu) @ pl["Q"]).T) for pl in planes])
    n_on = int((RAD > 0.5 * np.median(RAD, axis=1, keepdims=True)).all(0).sum())
    cands = n_range or sorted(set(range(max(2, n_on - 3), n_on + 4)) | {2 * n_on - 1, 2 * n_on, 2 * n_on + 1})
    best = None
    for N in cands:
        r = joint_decode(X, planes, N, center)
        q = float(r["coh"].mean())
        if best is None or q > best[0] + 1e-9:
            best = (q, r)
    return best[1]


def find_axes(X, restarts=24, iters=300, max_axes=6, cv_max=0.03, min_energy=0.02, seed=1):
    """Real characters (order 2): 1-D directions on which every symbol sits at +r or -r. Returned as pseudo-planes
    (second column zero) so that their 'angle' is 0 or pi."""
    g = torch.Generator().manual_seed(seed)
    Z, basis, S = _whiten(X)
    total = (Z ** 2).sum()
    axes, R = [], Z.copy()
    for _ in range(max_axes):
        Rt = torch.tensor(R, dtype=torch.float64)
        best = None
        for _r in range(restarts):
            a = torch.randn(R.shape[1], generator=g, dtype=torch.float64).requires_grad_(True)
            opt = torch.optim.Adam([a], lr=0.05)
            for _i in range(iters):
                u = a / a.norm(); y2 = (Rt @ u) ** 2
                loss = y2.var() / y2.mean() ** 2 - 0.05 * y2.mean() / (Rt ** 2).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            u = (a / a.norm()).detach().numpy(); y = R @ u
            y2 = y ** 2; cv = float(y2.var() / y2.mean() ** 2); en = float(y2.sum() / total)
            balanced = abs(int((y > 0).sum()) - int((y < 0).sum())) <= 0.1 * len(y)   # an order-2 character splits the group in half
            if en > min_energy and balanced and (best is None or cv < best[0]):
                best = (cv, en, u)
        if best is None or best[0] > cv_max:
            break
        cv, en, u = best
        Q = np.zeros((X.shape[1], 2)); Q[:, 0] = basis.T @ u
        axes.append({"cv": cv, "energy": en, "Q": Q, "axis": True})
        R = R - np.outer(R @ u, u)
    return axes, R, basis


def find_structure(X, **kw):
    """Axes (order-2 characters) first, then circle planes in what is left."""
    axes, R, basis = find_axes(X)
    Xr = R @ basis                                  # remainder, back in the original coordinates (centred)
    planes = find_planes(Xr, **kw)
    return axes + planes
