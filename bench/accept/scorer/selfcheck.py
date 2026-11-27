"""The bench's self-check (Gate 1, first half): true answers at the ceiling, must-fail cases at the floor.

  selfcheck.py <draw id> [--variants] [--tiers=S0,S1,S5a]

For every planted network with a true blueprint: the true blueprint must pass every requirement at its line, and
(with --variants) three known-wrong blueprints built from it must each fail at least one: every part at a random
location of the same dimension; the output's definition replaced by a type-correct wrong one; and the true blueprint
on the network with its weights shuffled. For every S5a account: the scorer's own verdict (s5a_check). Prints counts
only, per tier; the variants are written to a temporary folder and deleted.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score  # noqa: E402
from score import core, I  # noqa: E402


def verdict_pass(task_dir, bp_path, sdir, own, spreads, tier):
    st, problems = score.statistics(task_dir, bp_path, sdir)
    if problems:
        return None
    vs = score.verdicts(st, own, spreads, tier)
    return all(ok for name, ok, _s, _l in vs if name != "_matched")


def variants(bp, rng):
    out = {}
    v = json.loads(json.dumps(bp))
    for p in v["parts"]:
        m = p["location"]["map"]
        if m["class"] == "subspace" and "basis" in m:
            B = np.asarray(m["basis"])
            Q, _ = np.linalg.qr(rng.standard_normal(B.shape))
            m["basis"] = Q.tolist()
    out["random_locations"] = v
    v = json.loads(json.dumps(bp))
    o = next(p for p in v["parts"] if p["id"] == v["output"])
    ups = [u for u in o.get("upstream", []) if next(q for q in v["parts"] if q["id"] == u)["vtype"] == o["vtype"]]
    alt = [q["id"] for q in v["parts"] if q["id"] != o["id"] and q["vtype"] == o["vtype"]
           and o["id"] not in core.ancestors(v, q["id"])]
    src = (ups or alt or [None])[0]
    if src is not None:
        dom = v["domain"]
        o["definition"] = f"(map id @{src})" if dom == "tok" else f"(add @{src} 1)"
        o["upstream"] = [src]
        out["wrong_output_definition"] = v
    return out


def shuffled_task(task_dir, rng, tmp):
    t = os.path.join(tmp, "shuffled_task")
    shutil.copytree(task_dir, t)
    task = I.load_task(t)
    f = os.path.join(t, "network", "params.npz" if task["network"]["format"] == "tracr_np" else "weights.npz")
    P = dict(np.load(f))
    for k in P:
        if k not in ("readout", "token_embed", "pos_embed", "seq_len", "out_positions"):
            a = P[k].ravel().copy()
            rng.shuffle(a)
            P[k] = a.reshape(P[k].shape)
    np.savez(f, **P)
    return t


def main():
    draw = sys.argv[1]
    with_variants = "--variants" in sys.argv
    only = [a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--tiers=")]
    only = only[0] if only else None
    spreads = score.load_spreads()
    ddir = os.path.join(score.SEALED, draw)
    rng = np.random.default_rng(20261127)
    true_ok, true_n, var_fail, var_n = {}, {}, {}, {}
    s5a = {"rejected": 0, "format_only": 0, "pending_only": 0, "n": 0}
    for net_id in sorted(os.listdir(os.path.join(score.DRAWS, draw))):
        task_dir = os.path.join(score.DRAWS, draw, net_id)
        sdir = os.path.join(ddir, net_id)
        if not os.path.isfile(os.path.join(task_dir, "task.json")):
            continue
        tier = I.load_task(task_dir)["tier"]
        if only and tier not in only:
            continue
        tb = os.path.join(sdir, "true_blueprint.json")
        if os.path.exists(tb) and not os.path.exists(os.path.join(sdir, "DROPPED")):
            own = json.load(open(os.path.join(sdir, "own.json")))
            true_n[tier] = true_n.get(tier, 0) + 1
            true_ok[tier] = true_ok.get(tier, 0) + bool(verdict_pass(task_dir, tb, sdir, own, spreads, tier))
            if with_variants:
                bp = json.load(open(tb))
                with tempfile.TemporaryDirectory() as tmp:
                    for name, v in variants(bp, rng).items():
                        f = os.path.join(tmp, name + ".json")
                        json.dump(v, open(f, "w"))
                        ok = verdict_pass(task_dir, f, sdir, own, spreads, tier)
                        var_n[tier] = var_n.get(tier, 0) + 1
                        var_fail[tier] = var_fail.get(tier, 0) + (ok is False)
                    st = shuffled_task(task_dir, rng, tmp)
                    ok = verdict_pass(st, tb, sdir, own, spreads, tier)
                    var_n[tier] = var_n.get(tier, 0) + 1
                    var_fail[tier] = var_fail.get(tier, 0) + (ok is False)
        adir = os.path.join(sdir, "accounts")
        if os.path.isdir(adir) and (not only or "S5a" in only):
            planted = os.path.exists(os.path.join(sdir, "own.json"))
            for a in sorted(os.listdir(adir)):
                s5a["n"] += 1
                if planted:     # an account of a planted network is held to that network's lines
                    own = json.load(open(os.path.join(sdir, "own.json")))
                    ok = verdict_pass(task_dir, os.path.join(adir, a), sdir, own, spreads, tier)
                    if ok is None:
                        s5a["format_only"] += 1
                    elif ok is False:
                        s5a["rejected"] += 1
                    else:
                        s5a["pending_only"] += 1
                    continue
                res = score.s5a_check(task_dir, os.path.join(adir, a), sdir)
                if res[0][0] == "format":
                    s5a["format_only"] += 1
                elif any(r for _, r, _, _ in res if r):
                    s5a["rejected"] += 1
                else:
                    s5a["pending_only"] += 1
    for tier in sorted(true_n):
        line = f"{draw} {tier}: true blueprints passing every requirement at its line: {true_ok[tier]} of {true_n[tier]}"
        if tier in var_n:
            line += f"; known-wrong blueprints failing at least one requirement: {var_fail[tier]} of {var_n[tier]}"
        print(line)
    if s5a["n"]:
        print(f"{draw} S5a: accounts rejected by the scorer's automatic tests: {s5a['rejected']} of {s5a['n']}; "
              f"not rejected, awaiting R10 and X0 (the copy tests): {s5a['pending_only']}; "
              f"format-only failures (do not count): {s5a['format_only']}")


if __name__ == "__main__":
    main()
