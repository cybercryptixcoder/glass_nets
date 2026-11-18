"""Item 0 of the plan: the regression check for judge changes.

Re-judges every saved construction of a non-LM target (lab/constructions/<target>/<run>/construction.py) and records
the verdict. `--write` saves docs/glass/baseline.json; with no flag the current judge is compared with that file and
every difference is printed. The judge changes may ADD tests (and so may turn a VERIFIED into not verified, which is
reported, not hidden), but must reproduce every old number they did not mean to change.

Run:  python3 scripts/glass/baseline.py [--write]
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "lab"
sys.path.insert(0, str(LAB))
import judge  # noqa: E402

OUT = ROOT / "docs" / "glass" / "baseline.json"
KEYS = ["verified", "complete", "stages_verified", "stages_total", "score"]


def runs():
    for c in sorted(LAB.glob("constructions/*/*/construction.py")):
        tid = c.parts[-3]
        meta_p = LAB / "models" / tid / "meta.json"
        if not meta_p.exists() or json.loads(meta_p.read_text()).get("forward") == "lm":
            continue
        yield tid, c


def judge_one(tid, c):
    meta, W, X = judge.load_target(tid)
    t0 = time.time()
    try:
        kw = {"labels": judge.load_labels(meta)} if hasattr(judge, "load_labels") else {}     # the pre-change judge has no labels
        rep = judge.judge(meta["forward"], W, X, c.read_text(), groups=meta.get("out_groups"), **kw)
    except judge.ConstructionError as e:
        return {"error": str(e)[:200]}
    v = rep["verdict"]
    row = {k: v.get(k) for k in KEYS}
    row["behaviour_match"] = round(rep["behaviour"]["match"], 4)
    row["sites"] = {s: {"substitute_match": round(r["substitute_match"], 4), "pass": r["pass"]} for s, r in rep["sites"].items()}
    row["new"] = {k: rep.get(k) for k in ("joint", "worst", "errors", "coverage") if k in rep}
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main():
    write = "--write" in sys.argv
    cur = {f"{tid}/{c.parts[-2]}": judge_one(tid, c) for tid, c in runs()}
    if write:
        OUT.write_text(json.dumps(cur, indent=1, default=str))
        print(f"wrote {OUT} ({len(cur)} constructions)")
        for k, r in cur.items():
            print(f"  {k:48s} verified={r.get('verified')} complete={r.get('complete')} behaviour={r.get('behaviour_match')}")
        return
    old = json.loads(OUT.read_text())
    diffs = 0
    for k, r in cur.items():
        o = old.get(k)
        if o is None:
            print(f"NEW   {k}"); continue
        for f in KEYS + ["behaviour_match"]:
            if o.get(f) != r.get(f):
                diffs += 1; print(f"DIFF  {k}: {f} {o.get(f)} -> {r.get(f)}")
        for s, sr in (o.get("sites") or {}).items():
            if (r.get("sites") or {}).get(s, {}).get("substitute_match") != sr["substitute_match"]:
                diffs += 1; print(f"DIFF  {k}: site {s} substitute_match {sr['substitute_match']} -> {(r.get('sites') or {}).get(s)}")
    print(f"{len(cur)} constructions re-judged, {diffs} differences from the baseline")


if __name__ == "__main__":
    main()
