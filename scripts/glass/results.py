"""Write results/glass/RESULTS.md from the queue state and each run's judge report. Job: one table that is always
current (run_ladder.py calls this after every job; the check-ins call it too).

Run:  python3 scripts/glass/results.py
"""
import json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "lab"
RES = ROOT / "results" / "glass"


def row(name, j):
    target, rid = j["target"], j.get("run_id")
    meta = json.loads((LAB / "models" / target / "meta.json").read_text())
    kind = name.split("__")[1]
    cells = [meta["name"].replace("AlgZoo RNN ", ""), str(meta["params"]), kind, j["status"], rid or "-"]
    rep_p = LAB / "constructions" / target / (rid or "_") / "report_current.json"     # re-judged by the current judge
    if not rep_p.exists():
        rep_p = rep_p.with_name("report.json")
    run_p = LAB / "constructions" / target / (rid or "_") / "run.json"
    if rep_p.exists():
        rep = json.loads(rep_p.read_text())
        v, c = rep["verdict"], rep.get("coverage") or {}
        verdict = "COMPLETE" if v.get("complete") else ("verified (partial)" if v.get("verified") else "not verified")
        pct = lambda x: "-" if x is None else f"{x:.1%}"
        cells += [verdict, f"{v.get('stages_verified', 0)}/{v.get('stages_total', 0)}", pct(rep["behaviour"]["match"]),
                  pct(c.get("joint")), pct(c.get("worst_slice")), pct(c.get("errors_reproduced"))]
    else:
        cells += ["-"] * 6
    if run_p.exists():
        r = json.loads(run_p.read_text())
        cells += [str(len(r.get("steps", []))), str(len(r.get("submissions", []))), f"${r.get('cost', 0):.2f}"]
    else:
        cells += ["-"] * 3
    return cells


def main():
    st = json.loads((RES / "queue.json").read_text()) if (RES / "queue.json").exists() else {}
    head = ["network", "params", "run", "job status", "run id", "verdict", "stages", "behaviour", "all sites at once",
            "worst slice", "mistakes reproduced", "steps", "submissions", "cost"]
    L = ["# Glass Networks: AlgZoo results", "",
         f"*Regenerated {time.strftime('%Y-%m-%d %H:%M %Z')} by `scripts/glass/results.py` from `results/glass/queue.json` and each "
         "run's `report.json`. 'main' = blinded constructor run; 'prior' = prior-only control (told the task, no weights, one "
         "submission); 'shuffled' = the same reader on weights shuffled within each tensor (should fail).*", "",
         "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    order = {n: i for i, n in enumerate(sorted(st, key=lambda n: (n.split("__")[1] != "main", n.split("__")[1], json.loads(
        (LAB / "models" / st[n]["target"] / "meta.json").read_text())["params"])))}
    for name in sorted(st, key=order.get):
        L.append("| " + " | ".join(row(name, st[name])) + " |")
    (RES / "RESULTS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
