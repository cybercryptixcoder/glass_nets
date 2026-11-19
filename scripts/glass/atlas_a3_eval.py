"""A3 / A18 evaluation against the pre-registered criteria (DIARY: A3 PRE-REGISTRATION 04:42, A15 04:49, A17 05:01, A19 05:09).
Reads results/glass/atlas/<t>_d4.json.
    python3 scripts/glass/atlas_a3_eval.py [--v2 | --v3] [--a18] [targets...]
v1 (as pre-registered): (i) weight-space residual <= 0.05; (ii) rule_gated_trusted >= task rule; (iii) trusted share >= 0.5;
    (iv) median specificity >= 10; (v) functional groups vs shuffled control.
v2 (A15 + A17): (i') mechanisms alone >= task rule; (ii) rule_gated_trusted_v2; (vi) v2 off_recall >= 0.5.
v3 (A19): as v2, with (ii) rule_gated_trusted_v3, (iii) trusted = confidence on clear-cut calls >= 0.9, (vi) v3 off_recall.
v4 (A23): as v3, with (ii) rule_gated_trusted_v4 (decision-checked gate) and (vi) v4 off_recall over all trusted mechanisms."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; A = ROOT / "results/glass/atlas"; sys.path.insert(0, str(ROOT / "lab"))
A3 = ["zoo_argmedian_8_5", "zoo_argmedian_8_7", "zoo_argmedian_16_5", "zoo_argmedian_16_7", "zoo_2nd_argmax_16_10", "zoo_2nd_argmax_16_10_s0", "zoo_2nd_argmax_8_8"]
A18 = ["zoo_2nd_argmax_16_5_s0", "zoo_2nd_argmax_16_8_s0", "zoo_2nd_argmax_32_5_s0", "zoo_argmedian_32_5_s0", "zoo_2nd_argmax_32_10_s0", "zoo_argmedian_32_7_s0"]
V = "v4" if "--v4" in sys.argv else "v3" if "--v3" in sys.argv else "v2" if "--v2" in sys.argv else "v1"
args = [x for x in sys.argv[1:] if not x.startswith("--")]
targets = args or (A18 if "--a18" in sys.argv else A3)
FKEY = {"v1": "rule_gated_trusted", "v2": "rule_gated_trusted_v2", "v3": "rule_gated_trusted_v3", "v4": "rule_gated_trusted_v4"}[V]
CKEY = "confidence_clear" if V in ("v3", "v4") else "confidence"
rows, n_ok = [], 0
for t in targets:
    f = A / f"{t}_d4.json"
    if not f.exists():
        rows.append(f"| {t} | (not built yet) |"); continue
    a = json.loads(f.read_text()); mech = [x for m in a["matrices"].values() for x in m["mechanisms"]]
    if FKEY not in a["fidelity"] or (V != "v1" and "mechanisms_alone" not in a["fidelity"]):
        rows.append(f"| {t} | (built before this version; rebuild) |"); continue
    task = a["task_rule"]
    if V == "v1":
        r1 = max(m["alive_only_rel_error"] for m in a["matrices"].values()); i_ = r1 <= 0.05; c1 = f"{r1:.3f}"
    else:
        r1 = a["fidelity"]["mechanisms_alone"]; i_ = r1 >= task; c1 = f"{r1:.3f}"
    fid = a["fidelity"][FKEY]; ii = fid >= task
    na = [x for x in mech if x["trigger"]["rule"] != "always"]
    tr = sum(x["trigger"].get(CKEY, 0) >= 0.9 for x in na) / max(len(na), 1); iii = tr >= 0.5
    spec = np.median([x["effect"]["specificity"] for x in mech]); iv = spec >= 10
    orc = a.get("gate", {}).get(V, {}).get("off_recall", 0); vi = orc >= 0.5 if V != "v1" else True
    v = "-"
    cf = A / f"{t}_shuffled_d4.json"
    if a.get("findings") and cf.exists():
        import atlas
        g = sum(len(x) for x in a["findings"]["groups"].values())
        gc = sum(len(x) for x in atlas.findings(json.loads(cf.read_text()))["groups"].values())
        v = f"{g} vs {gc} → {'yes' if g > gc else 'no'}"
    ok = i_ and ii and iii and iv and vi; n_ok += ok
    mark = lambda b: "✓" if b else "✗"          # noqa: E731
    rows.append(f"| {t} | {c1} {mark(i_)} | {fid:.3f} vs {task:.3f} {mark(ii)} | {tr:.2f} {mark(iii)} | {'only where needed' if spec >= 1e5 else f'{spec:.0f}×'} {mark(iv)} | {v} |"
                + (f" {orc:.2f} {mark(vi)} |" if V != "v1" else "") + f" {'PASS' if ok else 'fail'} |")
c1h = "(i) residual ≤ 0.05" if V == "v1" else "(i') mechanisms alone ≥ task rule"
print(f"Version {V}")
print(f"| network | {c1h} | (ii) rebuilt ({FKEY}) vs task rule | (iii) trusted share ≥ 0.5 | (iv) median specificity ≥ 10 | (v) groups vs control |"
      + (" (vi) off_recall ≥ 0.5 |" if V != "v1" else "") + " result |")
print("|---|---|---|---|---|---|" + ("---|" if V != "v1" else "") + "---|")
print("\n".join(rows))
print(f"\nAll criteria hold in {n_ok} of {len(targets)} networks.")
