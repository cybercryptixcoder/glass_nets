"""Re-derive the presentation of saved atlases (findings, glosses, markdown certificate) from their JSON, without
rebuilding: used after presentation-only changes (A20-A22). Numbers in the JSON are not recomputed."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "lab"))
import atlas
A = ROOT / "results/glass/atlas"
for f in sorted(A.glob("*_d4.json")):
    if "_shuffled_" in f.name:
        continue
    a = json.loads(f.read_text()); t = f.name[:-len("_d4.json")]
    cf = A / f"{t}_shuffled_d4.json"; ctrl = json.loads(cf.read_text()) if cf.exists() else None
    a["findings"] = atlas.findings(a, ctrl); atlas.gloss(a)
    f.write_text(json.dumps(a, indent=1)); (A / f"{t}_d4.md").write_text(atlas.render(a, ctrl))
    print("re-rendered", t)
