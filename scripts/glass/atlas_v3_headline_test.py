"""Test for CHANGES A22 (certificate headline = v3). Fails before, passes after."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
a = json.loads((ROOT / "results/glass/atlas/zoo_2nd_argmax_16_10_d4.json").read_text())
v3 = a["fidelity"]["rule_gated_trusted_v3"]
md = atlas.render(a)
head = [l for l in md.splitlines() if "of the network's decisions" in l][0]
check(f"**{100 * v3:.1f}%**" in head, f"certificate leads with v3 {100 * v3:.1f}% ({head[:140]})")
st = atlas.findings(a)["statements"]
s4 = [x for x in st if x.startswith("4.")][0]
check(f"{100 * v3:.1f}%" in s4, f"findings statement 4 uses v3 ({s4})")
P = atlas._load_play("zoo_2nd_argmax_16_10"); tgt = P["tgt"]
x = tgt.X[-200:]
batch = atlas.judge_decide(tgt.run_gated(P["mods"], x, P["gate"]) if hasattr(tgt, "run_gated") else atlas.run_gated(tgt, P["mods"], x, P["gate"]))
import atlas as A
gv3 = A.trusted_gate(P["trees"], P["alive"], P.get("conf_clear", P["conf"]), P["mods"][next(iter(P["mods"]))].U.shape[0], tau=A.GATE_TAU) if "tau" in A.trusted_gate.__code__.co_varnames else None
check(gv3 is not None, "trusted_gate takes tau")
if gv3 is not None:
    ref = atlas.judge_decide(atlas.run_gated(tgt, P["mods"], x, gv3))
    check((batch == ref).all(), f"playground gate is the v3 gate ({(batch == ref).mean():.3f} agree)")
    one = [atlas.play("zoo_2nd_argmax_16_10", x=x[i])["rebuilt_decision"] for i in range(50)]
    check(all(int(o) == int(r) for o, r in zip(one, ref[:50])), "play() single-input decisions equal the batch v3 run")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
