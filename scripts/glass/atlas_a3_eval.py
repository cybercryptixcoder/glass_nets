"""A3 evaluation against the pre-registered criteria (DIARY, A3 PRE-REGISTRATION). Reads results/glass/atlas/<t>_d4.json.
--v2: A15 gate v2 (criterion (ii) uses rule_gated_trusted_v2, plus (vi): v2 off_recall >= 0.5) and A17 (criterion (i') replaces (i):
the mechanisms alone, residual removed, keep >= the task rule)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; A = ROOT / "results/glass/atlas"
TARGETS = ["zoo_argmedian_8_5", "zoo_argmedian_8_7", "zoo_argmedian_16_5", "zoo_argmedian_16_7", "zoo_2nd_argmax_16_10", "zoo_2nd_argmax_16_10_s0", "zoo_2nd_argmax_8_8"]
CONTROLLED = ["zoo_argmedian_16_5", "zoo_2nd_argmax_16_10"]
V2 = '--v2' in sys.argv; args = [x for x in sys.argv[1:] if x != '--v2']
FKEY = 'rule_gated_trusted_v2' if V2 else 'rule_gated_trusted'
rows, n_ok = [], 0
for t in args or TARGETS:
    f = A / f"{t}_d4.json"
    if not f.exists():
        rows.append(f"| {t} | (not built yet) | | | | | |"); continue
    a = json.loads(f.read_text()); mech = [x for m in a["matrices"].values() for x in m["mechanisms"]]
    i_ = max(m["alive_only_rel_error"] for m in a["matrices"].values()) <= 0.05
    if V2 and "mechanisms_alone" not in a["fidelity"]:
        rows.append(f"| {t} | (built before A17; rebuild) | | | | | | |"); continue
    if V2:
        i_ = a["fidelity"]["mechanisms_alone"] >= a["task_rule"]
    if FKEY not in a["fidelity"]:
        rows.append(f"| {t} | (built before A15; rebuild) | | | | | |"); continue
    ii = a["fidelity"][FKEY] >= a["task_rule"]
    orc = a.get("gate", {}).get("v2", {}).get("off_recall", 0); vi = orc >= 0.5 if V2 else True
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
    ok = i_ and ii and iii and iv and vi; n_ok += ok
    c1 = f"{a['fidelity']['mechanisms_alone']:.3f}" if V2 else f"{max(m['alive_only_rel_error'] for m in a['matrices'].values()):.3f}"
    rows.append(f"| {t} | {c1} {'✓' if i_ else '✗'} | "
                f"{a['fidelity'][FKEY]:.3f} vs {a['task_rule']:.3f} {'✓' if ii else '✗'} | {tr:.2f} {'✓' if iii else '✗'} | "
                f"{spec:.0f}× {'✓' if iv else '✗'} | {v} |" + (f" {orc:.2f} {'✓' if vi else '✗'} |" if V2 else "") + f" {'PASS' if ok else 'fail'} |")
print("| network | " + ("(i') mechanisms alone ≥ task rule" if V2 else "(i) residual ≤ 0.05") + " | (ii) rule-gated" + (" v2" if V2 else "") + " vs task rule | (iii) trusted share ≥ 0.5 | (iv) median specificity ≥ 10 | (v) functional groups vs control |" + (" (vi) v2 off_recall ≥ 0.5 |" if V2 else "") + (" (i)–(iv), (vi) |" if V2 else " (i)–(iv) |"))
print("|---|---|---|---|---|---|---|" + ("---|" if V2 else "")); print("\n".join(rows)); LBL = "(i'), (ii)-(iv) and (vi)" if V2 else "(i)-(iv)"
print(f"\n{LBL} hold in {n_ok} networks (bar: ≥ 4 of 7).")
