"""A3 evaluation against the pre-registered criteria (DIARY, A3 PRE-REGISTRATION). Reads results/glass/atlas/<t>_d4.json."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; A = ROOT / "results/glass/atlas"
TARGETS = ["zoo_argmedian_8_5", "zoo_argmedian_8_7", "zoo_argmedian_16_5", "zoo_argmedian_16_7", "zoo_2nd_argmax_16_10", "zoo_2nd_argmax_16_10_s0", "zoo_2nd_argmax_8_8"]
CONTROLLED = ["zoo_argmedian_16_5", "zoo_2nd_argmax_16_10"]
rows, n_ok = [], 0
for t in sys.argv[1:] or TARGETS:
    f = A / f"{t}_d4.json"
    if not f.exists():
        rows.append(f"| {t} | (not built yet) | | | | | |"); continue
    a = json.loads(f.read_text()); mech = [x for m in a["matrices"].values() for x in m["mechanisms"]]
    i_ = max(m["alive_only_rel_error"] for m in a["matrices"].values()) <= 0.05
    ii = a["fidelity"]["rule_gated_trusted"] >= a["task_rule"]
    na = [x for x in mech if x["trigger"]["rule"] != "always"]
    tr = sum(x["trigger"]["confidence"] >= 0.9 for x in na) / max(len(na), 1); iii = tr >= 0.5
    spec = np.median([x["effect"]["specificity"] for x in mech]); iv = spec >= 10
    g = sum(len(v) for v in a["findings"]["groups"].values()) if a.get("findings") else 0
    v = "-"
    if a.get("findings") and a["findings"].get("control"):
        gc = a["findings"]["control"]
        cf = A / f"{t}_shuffled_d4.json"
        if cf.exists():
            import importlib; sys.path.insert(0, str(ROOT / "lab")); atlas = importlib.import_module("atlas")
            gcount = sum(len(v_) for v_ in atlas.findings(json.loads(cf.read_text()))["groups"].values())
            v = f"{g} vs {gcount} → {'yes' if g > gcount else 'no'}"
    ok = i_ and ii and iii and iv; n_ok += ok
    rows.append(f"| {t} | {max(m['alive_only_rel_error'] for m in a['matrices'].values()):.3f} {'✓' if i_ else '✗'} | "
                f"{a['fidelity']['rule_gated_trusted']:.3f} vs {a['task_rule']:.3f} {'✓' if ii else '✗'} | {tr:.2f} {'✓' if iii else '✗'} | "
                f"{spec:.0f}× {'✓' if iv else '✗'} | {v} | {'PASS' if ok else 'fail'} |")
print("| network | (i) residual ≤ 0.05 | (ii) rule-gated vs task rule | (iii) trusted share ≥ 0.5 | (iv) median specificity ≥ 10 | (v) functional groups vs control | (i)–(iv) |")
print("|---|---|---|---|---|---|---|"); print("\n".join(rows)); print(f"\n(i)–(iv) hold in {n_ok} networks (bar: ≥ 4 of 7).")
