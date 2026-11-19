"""A25 confirmatory test (DIARY 05:45): ARC's M8,5 against zoo M8,5 seeds 1-4. Each seed's atlas must exist
(results/glass/atlas/zoo_2nd_argmax_8_5_s<k>_d4.json). Holds if the real count of role-and-variable matches exceeds the
null's 95th percentile (recurrence_control, n = 300) in >= 3 of the 4 seeds."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
A = ROOT / "results/glass/atlas"
arc = json.loads((A / "zoo_2nd_argmax_8_5_d4.json").read_text())
n_ok, rows = 0, []
for k in (1, 2, 3, 4):
    f = A / f"zoo_2nd_argmax_8_5_s{k}_d4.json"
    if not f.exists():
        rows.append(f"| seed {k} | (atlas not built yet) | | | |"); continue
    b = json.loads(f.read_text())
    r = atlas.recurrence_control(arc, b, n=300)
    cmp_ = atlas.compare_atlases(arc, b)
    ok = r["real"] > r["null_p95"]; n_ok += ok
    rows.append(f"| seed {k} | {r['real']} | {r['null_mean']:.2f} / {r['null_p95']:.0f} | {r['p_value']:.3f} | {'yes' if ok else 'no'} |")
    print(f"seed {k}:", cmp_["summary"]); [print("   ", p) for p in cmp_["pairs"]]
print("| seed | role-and-variable matches | null mean / 95th pct | P(null >= real) | beats the 95th pct |")
print("|---|---|---|---|---|"); print("\n".join(rows))
print(f"\nBeats the null's 95th percentile in {n_ok} of 4 seeds (A25 holds if >= 3).")
