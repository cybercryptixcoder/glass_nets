"""Re-judge saved constructions with the current judge, so the catalog shows today's certificate (with the
whole-model tests), not the one from the day the run finished. Writes report_current.json and report_current.txt
next to each construction; the run's own report.json is left as it was.

Run:  python3 scripts/glass/rejudge.py [target ...]      (default: every non-LM target with constructions)
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "lab"
sys.path.insert(0, str(LAB))
import judge  # noqa: E402

want = set(sys.argv[1:])
for c in sorted(LAB.glob("constructions/*/*/construction.py")):
    tid, rdir = c.parts[-3], c.parent
    mp = LAB / "models" / tid / "meta.json"
    if (want and tid not in want) or tid.startswith("_") or not mp.exists() or json.loads(mp.read_text())["forward"] == "lm":
        continue
    run = json.loads((rdir / "run.json").read_text()) if (rdir / "run.json").exists() else {}
    variant = run.get("config", {}).get("variant", "original")
    meta, W, X = judge.load_target(tid, variant)
    try:
        rep = judge.judge(meta["forward"], W, X, c.read_text(), groups=meta.get("out_groups"), labels=judge.load_labels(meta))
    except judge.ConstructionError as e:
        print(f"{tid}/{rdir.name}: construction error: {str(e)[:100]}"); continue
    (rdir / "report_current.json").write_text(json.dumps(rep, indent=1, default=str))
    (rdir / "report_current.txt").write_text(judge.render(rep) + "\n")
    v = rep["verdict"]
    print(f"{tid}/{rdir.name}: complete={v['complete']} verified={v['verified']} stages {v['stages_verified']}/{v['stages_total']}")
