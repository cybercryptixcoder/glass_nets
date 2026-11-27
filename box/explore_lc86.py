"""Exploratory (Gate 1 planning; DIARY 01:28): can lc_8_6's real account (round 2 b, run d7df, the census rule) be
completed as a blueprint? It locates only the token identities; its census quantities (fp: fixed points, t2: 2-cycles,
nP: the largest preimage count among fixed points) are computed from them. Are they linearly present in the network,
at the last position after layer 0 and after layer 1? Held-out R2 of a least-squares direction (fit on half, scored on
the other half), against the same fit on a weight-shuffled network. Run: python3 box/explore_lc86.py"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = dict(np.load(os.path.join(ROOT, "lab", "models", "lc_8_6", "weights.npz")))
X = np.load(os.path.join(ROOT, "lab", "models", "lc_8_6", "inputs.npy"))


def run(W, X):
    x = W["W_E"][X] + W["W_pos"][None]
    sites = {"embed": x.copy()}
    for l in (0, 1):
        q = x @ W[f"L{l}.W_Q"][0]; k = x @ W[f"L{l}.W_K"][0]; v = x @ W[f"L{l}.W_V"][0]
        a = q @ k.transpose(0, 2, 1) / np.sqrt(8.0)
        a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        x = x + (a @ v) @ W[f"L{l}.W_O"][0]
        sites[f"L{l}"] = x.copy()
    return sites, x[:, -1] @ W["W_U"]


def census(X):
    T = X.astype(float); pos = np.arange(6)[None]
    m = (np.abs(T - pos) < 0.5).astype(float); fp = m.sum(1)
    cnt = np.stack([(np.abs(T - j) < 0.5).sum(1) for j in range(6)], 1).astype(float)
    t2 = np.zeros(len(X))
    for p in range(6):
        for q in range(p + 1, 6):
            t2 += ((np.abs(T[:, p] - q) < 0.5) & (np.abs(T[:, q] - p) < 0.5))
    nP = (m * cnt).max(1)
    return {"fp (fixed points)": fp, "t2 (2-cycles)": t2, "nP (largest preimage count at a fixed point)": nP}


def heldout_r2(H, y, rng):
    idx = rng.permutation(len(y)); a, b = idx[: len(y) // 2], idx[len(y) // 2:]
    A = np.column_stack([H[a], np.ones(len(a))]); B = np.column_stack([H[b], np.ones(len(b))])
    w, *_ = np.linalg.lstsq(A, y[a], rcond=None)
    return 1 - np.mean((B @ w - y[b]) ** 2) / np.var(y[b])


rng = np.random.default_rng(0)
sites, out = run(W, X)
Ws = {k: (rng.permutation(v.ravel()).reshape(v.shape) if k != "out_positions" else v) for k, v in W.items()}
ssites, _ = run(Ws, X)
print("network accuracy against the labels:", float((out.argmax(1) == np.load(os.path.join(ROOT, "lab", "models", "lc_8_6", "labels.npy"))).mean()))
for name, y in census(X).items():
    row = []
    for site in ("embed", "L0", "L1"):
        r = heldout_r2(sites[site][:, -1], y, rng); rc = heldout_r2(ssites[site][:, -1], y, rng)
        row.append(f"{site}: {r:.3f} (shuffled {rc:.3f})")
    print(f"{name}: " + "; ".join(row))
