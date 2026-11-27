"""Gate 2 rehearsal in miniature (builder-made, not a bench case; DIARY 01:39): a verdict computed in stages, the
shape that broke the last build. Can the search find the network's intermediate variables and define the verdict
through them, where it cannot define the verdict from the board within L?

Planted mechanism (chosen before the run; a builder's rehearsal, so not sealed):
  g1 = (sum hanging_own)       the mover's hanging pieces
  g2 = (sum hanging_opp)       the opponent's hanging pieces
  g3 = (sum passed_pawn)       passed pawns on the board
  verdict v = relu(g2 - g1) * g3: six symbols through its ingredients, nine from the board (over L = 6)
A small MLP reads the 64 x 55 per-square features and the 5 position features: hidden layer 1 (8 units), hidden
layer 2 (8 units), output v. Units 0, 1, 2 of layer 1 are set to g1, g2, g3 by construction (so the intermediate variables
exist at known units) and the later layers are trained to compute v from layer 1.
The box's side, blind to which units are planted: (1) for every unit of layer 1, search the vocabulary for a
definition (held-out correlation); (2) search for the verdict from the board alone, within L; (3) search for the
verdict through the units found in (1), as parts.
Yes if (1) recovers g1, g2, g3 at held-out correlation >= 0.99 for the planted units and (3) defines the verdict at
held-out R2 >= 0.95 where (2) cannot. Run: python3 box/rehearse_staged.py"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box.search import best_definitions, vocab  # noqa: E402

t0 = time.time()
torch.manual_seed(0)
items = json.load(open(os.path.expanduser("~/kdata/chess/LD2/meta.json")))
items = items["items"] if isinstance(items, dict) else items
fens = [it["fen"] for it in items[:3000]]
d = vocab.get("chess")
ctx = vocab.featurize(fens)
G = {k: np.asarray(vocab.evaluate(e, d, ctx), float) for k, e in
     {"g1": "(sum hanging_own)", "g2": "(sum hanging_opp)", "g3": "(sum passed_pawn)"}.items()}
v = np.maximum(G["g2"] - G["g1"], 0) * G["g3"]
print("built features and targets", round(time.time() - t0), "s; verdict std", round(float(v.std()), 3), flush=True)

Xin = np.concatenate([ctx["P"].reshape(len(fens), -1), ctx["G"]], 1).astype(np.float32)
Xn = torch.tensor(Xin)    # raw features (small integers): the ingredients are exact sums of them, so a linear layer can hold them
tr, te = np.arange(0, 2000), np.arange(2000, 3000)
net = torch.nn.ModuleDict({"l1": torch.nn.Linear(Xn.shape[1], 8), "l2": torch.nn.Linear(8, 32), "out": torch.nn.Linear(32, 1)})
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
Yg = torch.tensor(np.stack([G["g1"], G["g2"], G["g3"]], 1), dtype=torch.float32)
Yv = torch.tensor(v, dtype=torch.float32)[:, None]


def forward(x):
    h1 = net["l1"](x)                       # layer 1 pre-activations: units 0..2 carry g1..g3 (linear, so exact)
    h2 = torch.relu(net["l2"](torch.relu(h1) + 0 * h1))
    return h1, h2, net["out"](h2)


# Layer 1 is planted by construction (a builder's rehearsal; the bench trains its own with interchange objectives):
# units 0..2 sum their feature over the 64 squares; units 3..7 are random projections (distractors). Layers 2 and out
# are trained to compute the verdict from layer 1.
names = list(vocab.get("chess").atoms)
sq_names = [n for n in names if vocab.get("chess").atoms[n].type == "Sq"]
with torch.no_grad():
    Wl1 = torch.zeros(8, Xn.shape[1]); F = len(sq_names)
    for unit, feat in enumerate(("hanging_own", "hanging_opp", "passed_pawn")):
        j = sq_names.index(feat)
        Wl1[unit, torch.arange(64) * F + j] = 1.0
    Wl1[3:] = torch.randn(5, Xn.shape[1]) * 0.05
    net["l1"].weight.copy_(Wl1); net["l1"].bias.zero_()
net["l1"].requires_grad_(False)
opt = torch.optim.Adam([p_ for n_, p_ in net.named_parameters() if not n_.startswith("l1")], lr=3e-3)
for step in range(3000):
    idx = torch.tensor(tr)
    h1, h2, o = forward(Xn[idx])
    loss = ((o - Yv[idx]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    h1, h2, o = forward(Xn)
H1, OUT = h1.numpy(), o.numpy()[:, 0]
r2 = lambda p, y: float(1 - np.mean((p - y) ** 2) / np.var(y))  # noqa: E731
print("trained", round(time.time() - t0), "s; held-out: units 0-2 R2 vs g1-g3",
      [round(r2(H1[te, k], G[g][te]), 3) for k, g in enumerate(("g1", "g2", "g3"))], "output R2 vs v", round(r2(OUT[te], v[te]), 3), flush=True)

# the box's side: probes are the first 60 training positions; held-out scoring on the test positions
probe = list(range(60))
cprobe = {k: (val[probe] if k != "fens" else [val[i] for i in probe]) for k, val in ctx.items()}
ctest = {k: (val[te] if k != "fens" else [val[i] for i in te]) for k, val in ctx.items()}
found = {}
for k in range(8):
    out, _ = best_definitions(d, cprobe, H1[probe, k], 5, "Pos", k=1, ctx_test=ctest, target_test=H1[te, k])
    found[k] = out[0]
    print(f"layer-1 unit {k}: {out[0]['definition']}  held-out corr {out[0]['heldout_corr']}", flush=True)
board_only, _ = best_definitions(d, cprobe, OUT[probe], 6, "Pos", k=1, ctx_test=ctest, target_test=OUT[te])
print(f"verdict from the board alone (L=6): {board_only[0]['definition']}  held-out affine R2 {board_only[0]['heldout_r2_affine']}", flush=True)
# through the units as parts (their network values, as a located part would supply them)
good = [k for k in range(8) if found[k]["heldout_corr"] >= 0.99]
pf = {f"u{k}": ("Pos", H1[probe, k]) for k in good}
pt = {f"u{k}": ("Pos", H1[te, k]) for k in good}
# through the parts: position-level expressions only, so many more probe inputs fit in memory (the verdict is
# sparse: zero on most positions, so 60 probes are too few to tell candidates apart)
probe2 = list(range(1000))
cprobe2 = {k: (val[probe2] if k != "fens" else [val[i] for i in probe2]) for k, val in ctx.items()}
pf = {f"u{k}": ("Pos", H1[probe2, k]) for k in good}
through, _ = best_definitions(d, cprobe2, OUT[probe2], 6, "Pos", k=1, ctx_test=ctest, target_test=OUT[te],
                              parts_fit=pf, parts_test=pt, only_types=("Pos",))
print(f"verdict through parts {sorted(pf)}: {through[0]['definition']}  held-out affine R2 {through[0]['heldout_r2_affine']}", flush=True)
res = {"units": found, "board_only": board_only[0], "through_parts": through[0], "planted_units_recovered":
       [found[k]["heldout_corr"] >= 0.99 for k in range(3)], "seconds": round(time.time() - t0)}
json.dump(res, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "build", "rehearse_staged_session1.json"), "w"), indent=1)
print("done", round(time.time() - t0), "s")
