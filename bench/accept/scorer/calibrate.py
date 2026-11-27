"""Calibrate a draw (bench side): every planted network's own true-blueprint statistics, stored in
sealed/<draw>/<net>/own.json (the lines before the tier's spread), and the admission checks (BENCH_BRIEF.md §4):
outputs match the sealed mechanism on at least 98% of held-out inputs (R2 >= 0.95 for continuous outputs), and every
sealed intermediate variable passes interchange interventions on at least 90% of held-out pairs.

  calibrate.py <draw id>                 own statistics and admission, for every network with a true blueprint
  calibrate.py --spreads <d1> <d2> <d3>  the tier spreads from the first three development draws (then frozen)

Prints counts only, never a sealed definition.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score  # noqa: E402

TIER_STATS = ["alone_output", "alone_levels_min", "iia_mean", "iia_above_control", "subst_mean", "kept_p5", "overview"]


def calibrate(draw):
    ddir = os.path.join(score.SEALED, draw)
    adm = {}
    for net_id in sorted(os.listdir(os.path.join(score.DRAWS, draw))):
        sdir = os.path.join(ddir, net_id)
        if not os.path.exists(os.path.join(sdir, "true_blueprint.json")):
            continue
        task_dir = os.path.join(score.DRAWS, draw, net_id)
        tier = json.load(open(os.path.join(task_dir, "task.json")))["tier"]
        own = score.own_statistics(task_dir, sdir)
        json.dump(own, open(os.path.join(sdir, "own.json"), "w"))
        metric_ok = own["alone_output"] >= (0.98 if own.get("_metric", "accuracy") == "accuracy" else 0.95)
        iia_ok = (not np.isfinite(own.get("iia_min", np.nan))) or own["iia_min"] >= 0.90
        a = adm.setdefault(tier, [0, 0])
        a[1] += 1
        a[0] += bool(metric_ok and iia_ok)
        if not (metric_ok and iia_ok):
            open(os.path.join(sdir, "DROPPED"), "w").write("admission failed\n")
    for tier, (ok, n) in sorted(adm.items()):
        print(f"{draw} {tier}: admitted {ok} of {n} planted networks")
    return adm


def spreads(draws):
    """Spread = standard deviation across the first three development draws of the tier-level statistic (the mean
    over the tier's networks of the true blueprints' own scores; for recovery, of their sealed definitions' scores)."""
    per = {}
    for d in draws:
        ddir = os.path.join(score.SEALED, d)
        tier_vals = {}
        for net_id in sorted(os.listdir(ddir)):
            p = os.path.join(ddir, net_id, "own.json")
            if not os.path.exists(p) or os.path.exists(os.path.join(ddir, net_id, "DROPPED")):
                continue
            own = json.load(open(p))
            tier = json.load(open(os.path.join(score.DRAWS, d, net_id, "task.json")))["tier"]
            tv = tier_vals.setdefault(tier, {})
            for k in TIER_STATS:
                if np.isfinite(own.get(k, np.nan)):
                    tv.setdefault(k, []).append(own[k])
            ra = [r["agree"] for r in own["recovery_own"].values() if np.isfinite(r["agree"])]
            rs = [r["subst"] for r in own["recovery_own"].values() if np.isfinite(r["subst"])]
            tv.setdefault("rec_agree", []).extend(ra)
            tv.setdefault("rec_subst", []).extend(rs)
        for tier, tv in tier_vals.items():
            for k, v in tv.items():
                per.setdefault(tier, {}).setdefault(k, []).append(float(np.mean(v)) if v else np.nan)
    out = {tier: {k: float(np.nanstd(v)) for k, v in kv.items() if len(v) == len(draws)} for tier, kv in per.items()}
    out["_draws"] = list(draws)
    out["_frozen"] = True
    json.dump(out, open(os.path.join(score.SEALED, "spreads.json"), "w"), indent=1)
    print("spreads frozen from", ", ".join(draws), "for tiers", ", ".join(t for t in out if not t.startswith("_")))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "--spreads":
        spreads(sys.argv[2:5])
    else:
        calibrate(sys.argv[1])
