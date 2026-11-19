"""Test for CHANGES A30 (certificate headline = v4). Fails before, passes after."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
t = "zoo_2nd_argmax_16_5_s2"
a = json.loads((ROOT / f"results/glass/atlas/{t}_d4.json").read_text())
v4 = a["fidelity"]["rule_gated_trusted_v4"]
md = atlas.render(a)
head = [l for l in md.splitlines() if "of the network's decisions" in l][0]
check(f"**{100 * v4:.1f}%**" in head, f"certificate leads with v4 {100 * v4:.1f}% ({head[:150]})")
s4 = [x for x in atlas.findings(a)["statements"] if x.startswith("4.")][0]
check(f"{100 * v4:.1f}%" in s4, f"findings statement 4 uses v4 ({s4})")
P = atlas._load_play(t); tgt = P["tgt"]; x = tgt.X[-200:]
check("sel" in P, "the model file carries v4's switched set")
if "sel" in P:
    C = P["mods"][next(iter(P["mods"]))].U.shape[0]
    g = atlas.trusted_gate(P["trees"], P["alive"], P["conf_clear"], C, tau=atlas.GATE_TAU, only=set(map(tuple, P["sel"])))
    ref = atlas.judge_decide(atlas.run_gated(tgt, P["mods"], x, g))
    one = [atlas.play(t, x=x[i])["rebuilt_decision"] for i in range(50)]
    check(all(int(o) == int(r) for o, r in zip(one, ref[:50])), "play() decisions equal the batch v4 run")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
