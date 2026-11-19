"""Test for CHANGES A5 (atlas for transformers), on the planted transformer (copy the previous token).
    python3 scripts/glass/atlas_tf_test.py [SPD .pt for planted_transformer]"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
import atlas
if not hasattr(atlas, "TransformerTarget"):
    check(False, "atlas has TransformerTarget"); print(f"{len(fails)} FAILED"); sys.exit(1)
model = sys.argv[1] if len(sys.argv) > 1 else ROOT / "results/glass/atlas/spd/spd_planted_transformer_original_40000_2048.pt"
t = atlas.planted_transformer(); a = atlas.build_atlas(t, spd_model=model, depth=4)
print(atlas.summary(a))
check(a["fidelity"]["rule_gated_trusted"] >= 0.99, f"rule-gated fidelity >= 0.99 (got {a['fidelity']['rule_gated_trusted']:.4f})")
V = t.V
# Alignment is measured functionally (A5 test fix, CHANGES): the raw-weight cosine is not identifiable, because the planted
# inputs have null and constant directions (sum of token one-hots = sum of position one-hots = 1; empty slot dims), where
# weight does nothing on the data. So: input side = |correlation| over the data between the mechanism's input read V_c . a
# and the indicator 'the copied token is v'; output side (W_O) = cosine of the mechanism's centred logit effect W_U U_c
# (softmax ignores constant shifts) with the centred one-hot of token v.
import torch
xx = t.X[:4000]; rec = {}
def mm(name, a_):
    rec[name] = a_.detach().numpy(); return a_ @ t.weights[name].T
with torch.no_grad():
    t.forward(torch.tensor(xx), mm)
tok = xx.argmax(-1)                                        # [B, P]
relevant = {"L0_W_V": tok.reshape(-1), "W_U": tok[:, t.P - 2]}     # W_V reads token at its own position; W_U reads the copied token
# W_V acts at every position, but only its value at the attended position (P-2) reaches the output; elsewhere its weights
# are unidentifiable on the data, like the null directions. So its alignment is measured on the rows of position P-2.
rows = {"L0_W_V": (np.arange(len(xx) * t.P) % t.P) == t.P - 2, "W_U": np.ones(len(xx), bool)}
WU = t.weights["W_U"].numpy(); cen = np.eye(V) - 1.0 / V
for k, side in (("L0_W_V", "V"), ("L0_W_O", "U"), ("W_U", "V")):
    mech = a["matrices"][k]["mechanisms"]
    check(len(mech) == V, f"{k}: {V} mechanisms (got {len(mech)})")
    tok_ = []
    for x in mech:
        if side == "V":
            r = rec[k] @ np.array(x["V"]); sel = rows[k]
            cs = [abs(np.corrcoef(r[sel], relevant[k][sel] == v)[0, 1]) for v in range(V)]
            if k == "L0_W_V":
                print(f"   {x['id']}: |corr| over all positions {max(abs(np.corrcoef(r, relevant[k] == v)[0, 1]) for v in range(V)):.3f}, at the attended position {max(cs):.3f}")
        else:
            eff = cen @ (WU @ np.array(x["U"]))
            cs = [abs(eff @ cen[v]) / (np.linalg.norm(eff) * np.linalg.norm(cen[v]) + 1e-12) for v in range(V)]
        tok_.append((int(np.argmax(cs)), float(max(cs))))
    check(sorted(j for j, _ in tok_) == list(range(V)), f"{k}: one mechanism per token ({sorted(j for j, _ in tok_)})")
    check(all(c >= 0.9 for _, c in tok_), f"{k}: aligned with its token, functionally (|corr| or cosine {[round(c, 3) for _, c in tok_]})")
    check(all(x["trigger"]["confidence"] >= 0.95 for x in mech), f"{k}: triggers >= 0.95 ({[round(x['trigger']['confidence'], 3) for x in mech]})")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
