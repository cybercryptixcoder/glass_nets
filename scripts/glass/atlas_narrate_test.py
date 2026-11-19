"""Test for CHANGES A13 (guarded narrative)."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); ok or fails.append(what)
import atlas
if not hasattr(atlas, "narrate"):
    check(False, "atlas has narrate"); print(f"{len(fails)} FAILED"); sys.exit(1)
a = json.loads((ROOT / "results/glass/atlas/zoo_2nd_argmax_8_5_d4.json").read_text())
facts = atlas.fact_sheet(a)
bad = "Mechanism W_h.99 fires 73.4% of the time and fixes the answer."
check(len(atlas.narrative_violations(bad, facts)) >= 2, f"guard rejects an invented number and id ({atlas.narrative_violations(bad, facts)})")
good = "Mechanism W_h.4 is one of the trusted mechanisms."
check(atlas.narrative_violations(good, facts) == [], "guard accepts a paragraph built from the facts")
n = atlas.narrate(a); print(n["text"])
check(len(n["text"]) > 100 and n["violations"] == [], f"M8,5 narrative produced without violations ({n['violations']})")
print("ALL PASS" if not fails else f"{len(fails)} FAILED"); sys.exit(1 if fails else 0)
