"""The next level down on Leela LD2 (DIARY 06:00): define each of the top level's four block-10 variables through
variables of block 8, and certify that link by interventions at block 8.

1. Map block 8: a sparse dictionary (512 variables, k = 8, the same recipe as block 10's), trained on LD2's block-8
   activations for the 20,000 positions; board-wide totals.
2. For each top-level variable c (its block-10 total): an invented linear readout of at most four block-8 variables
   plus an offset (the five-constant budget), chosen greedily on training positions, scored on held-out positions.
   Control line, per c: the same greedy search over a dictionary of an untrained (weight-permuted) LD2 at block 8.
3. Certify each link: swap block-8 variable g's board map (through its decoder direction) from position b into
   position a, run LD2 on to block 10, and compare the change in c's total with the change the readout predicts.
   Control: swap a block-8 variable outside the readout instead. Line: 99th percentile of the best of K controls.
4. Define the block-8 parts from the board with the vocabulary search, against a control line (three block-8
   variables outside every readout).
Run: python3 box/leela_levels.py"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_feats as F  # noqa: E402
from rosetta.run_feats import L, CV, CI  # noqa: E402
from rosetta.sae import train  # noqa: E402
from rosetta.fresh_where import totals  # noqa: E402
from box.search import best_definitions, vocab  # noqa: E402

t0 = time.time()
rng = np.random.default_rng(0); torch.manual_seed(0)
K = os.path.expanduser("~/kdata")
top = json.load(open("results/build/leela_box_session2.json"))["top_level"]["parts"]
T10 = totals(np.asarray(CV, dtype=np.float32), CI)
TR, TE = np.asarray(L.TR), np.asarray(L.TE)
N = len(L.items)
r2 = lambda p, y: float(1 - np.mean((p - y) ** 2) / np.var(y))  # noqa: E731


def dictionary(H):
    """H [N, 128, 64] -> (sae, totals [N, 512], codes (values, indices) per square)."""
    V = H.transpose(0, 2, 1).reshape(-1, 128).astype(np.float32)
    sae = train(V, epochs=6)
    vs, is_ = [], []
    with torch.no_grad():
        for j in range(0, len(V), 65536):
            v, i = sae.codes(torch.tensor(V[j:j + 65536])); vs.append(v.numpy()); is_.append(i.numpy())
    Vv = np.concatenate(vs).reshape(len(H), 64, -1); Ii = np.concatenate(is_).reshape(len(H), 64, -1)
    return sae, totals(Vv, Ii), (Vv, Ii)


H8 = np.asarray(np.load(os.path.join(K, "chess", "LD2", "acts_site8.npy"), mmap_mode="r"), dtype=np.float32).reshape(N, 128, 64)
sae8, T8, (V8, I8) = dictionary(H8)
print("block-8 dictionary trained", round(time.time() - t0), "s", flush=True)
mu = L.get_net(untrained_seed=0)
Hu = L.site_acts(mu, np.arange(N), site=8).reshape(N, 128, 64).astype(np.float32)
_, T8u, _ = dictionary(Hu)
print("untrained control dictionary", round(time.time() - t0), "s", flush=True)


def greedy(Tsrc, y, k=4):
    S = []
    for _ in range(k):
        best = (-9, None)
        for g in range(Tsrc.shape[1]):
            if g in S or Tsrc[TR, g].std() == 0:
                continue
            A = np.column_stack([Tsrc[TR][:, S + [g]], np.ones(len(TR))])
            w, *_ = np.linalg.lstsq(A, y[TR], rcond=None)
            s = r2(A @ w, y[TR])
            if s > best[0]:
                best = (s, g)
        S.append(best[1])
    A = np.column_stack([Tsrc[TR][:, S], np.ones(len(TR))])
    w, *_ = np.linalg.lstsq(A, y[TR], rcond=None)
    return S, w, r2(np.column_stack([Tsrc[TE][:, S], np.ones(len(TE))]) @ w, y[TE])


links = {}
for c in top:
    y = T10[:, c].astype(float)
    S, w, s = greedy(T8, y)
    _, _, su = greedy(T8u, y)
    links[c] = {"parts": [int(g) for g in S], "weights": [float(x) for x in w], "heldout_r2": s, "control_r2_untrained": su}
    print(f"f{c} (block 10) = readout of block-8 {S}: held-out R2 {s:.3f}; untrained-network control {su:.3f}"
          f"  [{round(time.time() - t0)} s]", flush=True)

# ---- interventions at block 8, effect on the block-10 totals
m = L.get_net()
D8 = sae8.dec.weight.detach()                                         # [128, 512]
pairs = rng.choice(TE, (200, 2)); pairs = pairs[pairs[:, 0] != pairs[:, 1]]
A_rows, B_rows = pairs[:, 0], pairs[:, 1]


def map8(g, rows):
    z = np.zeros((len(rows), 64), np.float32)
    v, i = V8[rows], I8[rows]
    for k in range(v.shape[2]):
        z += np.where(i[:, :, k] == g, v[:, :, k], 0.0)
    return z


def totals10_after(edit, rows):
    out = []
    with torch.no_grad():
        for j in range(0, len(rows), 500):
            h = m(L.planes(rows[j:j + 500]), upto=10, edit=edit).flatten(2).numpy()
            v, i = F.sae.codes(torch.tensor(h.transpose(0, 2, 1).reshape(-1, 128), dtype=torch.float32))
            out.append(totals(v.numpy().reshape(len(h), 64, -1), i.numpy().reshape(len(h), 64, -1)))
    return np.concatenate(out)


base10 = totals10_after(None, A_rows)


def edit8(g, za, zb):
    d = D8[:, g].view(1, 128, 1, 1)

    def ed(site, h):
        if site == 8:
            h = h + d * torch.tensor(zb - za, dtype=h.dtype).view(-1, 1, 8, 8)
        return h
    return ed


cert = {}
outside = [g for g in range(512) if all(g not in links[c]["parts"] for c in top)]
for c in top:
    lk = links[c]; cert[c] = {}
    for gi, g in enumerate(lk["parts"]):
        za, zb = map8(g, A_rows), map8(g, B_rows)
        after = totals10_after(edit8(g, za, zb), A_rows)
        actual = after[:, c] - base10[:, c]
        pred = lk["weights"][gi] * (T8[B_rows, g] - T8[A_rows, g])
        cert[c][g] = float(np.corrcoef(pred, actual)[0, 1]) if pred.std() > 0 and actual.std() > 0 else 0.0
best = []
for _ in range(12):
    sc = []
    for c in top:
        lk = links[c]; gi = int(rng.integers(len(lk["parts"]))); g = lk["parts"][gi]
        h_ = int(rng.choice(outside))
        after = totals10_after(edit8(h_, map8(h_, A_rows), map8(h_, B_rows)), A_rows)
        actual = after[:, c] - base10[:, c]
        pred = lk["weights"][gi] * (T8[B_rows, g] - T8[A_rows, g])
        sc.append(float(np.corrcoef(pred, actual)[0, 1]) if pred.std() > 0 and actual.std() > 0 else 0.0)
    best.append(max(sc))
line = float(np.quantile(best, 0.99))
for c in top:
    print(f"f{c}: links " + ", ".join(f"g{g} {cert[c][g]:.3f}" for g in links[c]["parts"]) + f" (line {line:.3f})", flush=True)

# ---- define the block-8 parts from the board
items = L.items
probe = list(TR[:60]); test = list(TE[:600])
d = vocab.get("chess")
cp = vocab.featurize([items[i]["fen"] for i in probe]); ct = vocab.featurize([items[i]["fen"] for i in test])
ctrl = []
for g in rng.choice(outside, 3, replace=False):
    o, _ = best_definitions(d, cp, T8[probe, int(g)], 5, "Pos", k=1, ctx_test=ct, target_test=T8[test, int(g)])
    ctrl.append(o[0]["heldout_corr"])
line_def = max(ctrl)
defs = {}
for g in sorted({g for c in top for g in links[c]["parts"]}):
    o, _ = best_definitions(d, cp, T8[probe, g], 5, "Pos", k=1, ctx_test=ct, target_test=T8[test, g])
    defs[g] = o[0]
    print(f"block-8 g{g}: {o[0]['definition']} held-out corr {o[0]['heldout_corr']} (line {line_def:.3f}) -> "
          f"{'defined' if o[0]['heldout_corr'] > line_def else 'unresolved'}", flush=True)
json.dump({"links": {str(c): links[c] for c in top}, "link_interventions": {str(c): {str(g): v for g, v in cert[c].items()} for c in top},
           "link_line": line, "block8_definitions": {str(g): defs[g] for g in defs}, "definition_line": line_def,
           "definition_controls": ctrl, "seconds": round(time.time() - t0)},
          open("results/build/leela_levels_session2.json", "w"), indent=1, default=float)
print("done", round(time.time() - t0), "s")
