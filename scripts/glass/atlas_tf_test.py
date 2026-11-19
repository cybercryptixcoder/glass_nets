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
for k, side in (("L0_W_V", "V"), ("L0_W_O", "U"), ("W_U", "V")):
    mech = a["matrices"][k]["mechanisms"]
    check(len(mech) == V, f"{k}: {V} mechanisms (got {len(mech)})")
    tok = []
    for x in mech:
        v = np.array(x[side]); v = v / (np.linalg.norm(v) + 1e-12)
        if k == "L0_W_V":
            seg = v[:V]                                 # input = residual; token dims 0..V-1
        elif k == "L0_W_O":
            seg = v[t.V + t.P:]                         # output = residual slot dims
        else:
            seg = v[t.V + t.P:]                         # W_U input = residual slot dims
        tok.append((int(np.argmax(np.abs(seg))), float(np.abs(seg).max())))
    check(sorted(j for j, _ in tok) == list(range(V)), f"{k}: one mechanism per token ({sorted(j for j, _ in tok)})")
    check(all(c >= 0.9 for _, c in tok), f"{k}: aligned with its token (cosines {[round(c, 3) for _, c in tok]})")
    check(all(x["trigger"]["confidence"] >= 0.95 for x in mech), f"{k}: triggers >= 0.95 ({[round(x['trigger']['confidence'], 3) for x in mech]})")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
