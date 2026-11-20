"""A45 test: the catalogue lists networks that have only an atlas."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import catalog
E = {e["target"]: e for e in catalog.entries()}
ok = all(t in E and E[t]["atlas"] for t in ("modmul_p113_s1", "modadd_p113_s4", "modrand_p113")) and len(E) >= 34 and "zoo_2nd_argmax_16_10" in E
print(len(E), "entries;", "ALL PASS" if ok else "FAILED"); sys.exit(0 if ok else 1)
