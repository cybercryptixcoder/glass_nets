"""The scorer's command line (development copy, scorer-v1). See bench/accept/README.md.

  score.py --task <task folder> --blueprint <blueprint.json>        one network
  score.py --draw <draw id> --blueprints <dir>                      every network of a draw with <dir>/<net>/blueprint.json
  score.py --hook --draw <draw id> --start <start config.json>      Gate 1's verifier hook (counts only)

Sealed answers are read from bench/accept/sealed/<draw>/. Printing rules (BENCH_BRIEF.md §1): on development draws,
per network and per requirement, pass or fail with the score and the line, plus Gate 2's diagnostics (how many sealed
variables were recovered, and which of the box's own parts matched one); on the certification draw, only pass or fail.
Never a sealed definition, family or mechanism.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ACCEPT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import core  # noqa: E402
from core import I, netlib  # noqa: E402

SEALED = os.environ.get("BENCH_SEALED", os.path.join(ACCEPT, "sealed"))   # the gitignored dev answers
DRAWS = os.environ.get("BENCH_DRAWS", os.path.join(ACCEPT, "draws"))
N_PAIRS = 150
CONTROL_REPS = 100
MIN_DECISIVE = 20      # a part is interchange-tested only on at least this many decisive pairs


# ====================================================================================================== statistics
def statistics(task_dir, bp_path, sealed_dir=None, seed=0, controls=True):
    """Every score the requirements use, for one blueprint on one network. Returns (stats, problems)."""
    task = I.load_task(task_dir)
    bp = I.load_blueprint(bp_path)
    bp_dir = os.path.dirname(os.path.abspath(bp_path))
    problems = I.validate_blueprint(bp, task, bp_dir)
    if problems:
        return {"format": 0.0}, problems
    net = netlib.load_network(task_dir, task)
    dom = task["domain"]
    held_path = os.path.join(sealed_dir, "heldout.json") if sealed_dir else None
    H = I.load_inputs(task_dir, task) if not (held_path and os.path.exists(held_path)) else _load_ctx(held_path, dom)
    rng = np.random.default_rng(seed)
    st = {"format": 1.0}
    run = net.run(H, return_sites=True)
    net_out = run["outputs"]
    given, choice = core.given_from_network(net, task, bp, H, bp_dir)
    netvals, _ = core.read_parts(net, task, bp, H, bp_dir, run=run)
    # R4: run alone, every definition at once, resampled unresolved/irrelevant parts, lookups as tables
    perm = _perm_within(H, dom, rng)
    given_rs = {k: core.take(v, perm) for k, v in given.items() if v is not None}
    ra = core.run_alone(bp_path, task, H, given_rs)["values"]
    ra = {k: _restore(v, dom) for k, v in ra.items()}
    st["alone_output"] = float(np.nanmean(core.agreement(ra[bp["output"]], net_out))) \
        if task["behaviour"]["metric"] == "accuracy" else _r2(ra[bp["output"]], net_out)
    lev = []
    for p in bp["parts"]:
        if p["id"] in netvals and ra.get(p["id"]) is not None and p["type"] == "rule":
            lev.append(_agree_any(ra[p["id"]], netvals[p["id"]], p))
    st["alone_levels_min"] = float(np.nanmin(lev)) if lev else np.nan
    # R3: interchange interventions on decisive pairs, family-wise random-subspace controls
    pairs = core.make_pairs(H, dom, N_PAIRS, rng, equal_len=False)
    tested, preps = [], {}
    for p in bp["parts"]:
        if p["location"]["map"]["class"] == "param_component" or p["type"] != "rule":
            continue
        prep = core.iia_prepare(task, bp, H, pairs, p, given, choice)
        if len(prep["decisive"]) >= MIN_DECISIVE:
            tested.append(p)
            preps[p["id"]] = prep
    scores = {}
    for p in tested:
        scores[p["id"]] = core.iia_score(net, task, preps[p["id"]], p["location"], bp_dir)
    st["iia_parts"] = len(tested)
    st["iia_min"] = float(np.nanmin(list(scores.values()))) if scores else np.nan
    st["iia_mean"] = float(np.nanmean(list(scores.values()))) if scores else np.nan
    if controls and tested:
        crng = np.random.default_rng(10_000 + seed)
        src_runs = {pid: net.run(preps[pid]["cs"], return_sites=True) for pid in preps}
        best = []
        for _ in range(CONTROL_REPS):
            m = -np.inf
            for p in tested:
                loc = core.random_subspace_loc(p["location"], task, crng)
                v = core.iia_score(net, task, preps[p["id"]], loc, bp_dir, src_run=src_runs[p["id"]])
                m = max(m, v if np.isfinite(v) else -np.inf)
            best.append(m)
        st["control_line"] = float(np.percentile(best, 99))
        st["iia_above_control"] = float(np.mean([scores[p["id"]] > st["control_line"] for p in tested]))
    # R3: substitution
    subs = [core.substitution(net, task, bp, H, p, rng, bp_dir, given, choice) for p in tested]
    st["subst_mean"] = float(np.nanmean(subs)) if subs and np.isfinite(subs).any() else np.nan
    # R4: kept parts suffice
    all_pairs = core.make_pairs(H, dom, core.n_inputs(H, dom), rng)
    rec = core.kept_parts(net, task, bp, H, all_pairs, bp_dir)
    st["kept_p5"] = float(np.nanpercentile(rec, 5)) if np.isfinite(rec).any() else np.nan
    st["kept_mean"] = float(np.nanmean(rec)) if np.isfinite(rec).any() else np.nan
    # R6: overview predictor against the network's output (the sealed top level on planted networks)
    ov = (bp.get("overview") or {}).get("predictor")
    if ov:
        dom_v = core.vocab.get(dom)
        types = {p["id"]: p["vtype"] for p in bp["parts"]}
        ovv = core.vocab.evaluate(ov, dom_v, H, parts=ra, part_types=types, library=bp.get("library") or {},
                                  ops_code=core.compile_library(bp))
        st["overview"] = float(np.nanmean(core.agreement(ovv, net_out))) \
            if task["behaviour"]["metric"] == "accuracy" else _r2(ovv, net_out)
    # R6 / R8: recovery of the sealed variables, and their wiring
    if sealed_dir and os.path.exists(os.path.join(sealed_dir, "true_blueprint.json")):
        sbp = json.load(open(os.path.join(sealed_dir, "true_blueprint.json")))
        rs = core.recovery_scores(net, task, bp, sbp, H, np.random.default_rng(seed + 7), bp_dir, sealed_dir)
        st["recovery"] = rs
    return st, []


def _r2(pred, ref):
    a, b = core._flat_num(pred), core._flat_num(ref)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return np.nan
    den = np.sum((b[ok] - b[ok].mean()) ** 2)
    return float(1 - np.sum((a[ok] - b[ok]) ** 2) / den) if den > 1e-12 else float(np.allclose(a[ok], b[ok], atol=1e-3))


def _agree_any(a, b, p):
    if p["encoding"]["kind"] == "numerical":
        x, y = core._flat_num(a), core._flat_num(b)
        ok = np.isfinite(x) & np.isfinite(y)
        return float(np.mean(np.abs(x[ok] - y[ok]) <= core.NUM_TOL * np.maximum(1, np.abs(y[ok])))) if ok.any() else np.nan
    ag = core.agreement(a, b)
    return float(np.nanmean(ag)) if np.isfinite(ag).any() else np.nan


def _perm_within(ctx, dom, rng):
    """A resampling permutation; token tasks permute within equal lengths."""
    N = core.n_inputs(ctx, dom)
    if dom != "tok":
        return rng.permutation(N)
    perm = np.arange(N)
    by = {}
    for i, t in enumerate(ctx["tokens"]):
        by.setdefault(len(t), []).append(i)
    for g in by.values():
        perm[g] = rng.permutation(g)
    return perm


def _restore(v, dom):
    if dom == "tok" or v is None:
        return v
    try:
        return np.asarray(v, float)
    except (TypeError, ValueError):
        return np.asarray(v, dtype=object)


def _load_ctx(path, dom):
    if path.endswith(".json"):
        c = json.load(open(path))
        return c if dom in ("tok", "chess") else {k: np.asarray(v) for k, v in c.items()}
    return {k: v for k, v in np.load(path).items()}


# ====================================================================================================== verdicts
REQS = [  # (name, statistic, direction): the line is the true blueprint's own score moved by the tier's spread
    ("R4 run alone (outputs)", "alone_output", "ge"),
    ("R4 run alone (every level)", "alone_levels_min", "ge"),
    ("R3 interchange (mean)", "iia_mean", "ge"),
    ("R3 interchange above control (share of parts)", "iia_above_control", "ge"),
    ("R3 substitution (mean)", "subst_mean", "ge"),
    ("R4 kept parts suffice (5th percentile)", "kept_p5", "ge"),
    ("R6 overview", "overview", "ge"),
]


def verdicts(st, own, spreads, tier):
    """[(requirement, passed, score, line)] for one network. own: the true blueprint's statistics on this network."""
    out = []
    sp = spreads.get(tier, {}) if spreads else {}
    for name, key, _dirn in REQS:
        if key not in own or not np.isfinite(own.get(key, np.nan)):
            continue
        line = own[key] - sp.get(key, 0.0)
        score = st.get(key, np.nan)
        out.append((name, bool(np.isfinite(score) and score >= line - 1e-9), score, line))
    if "recovery" in st and "recovery_own" in own:
        rec = st["recovery"]
        n_ok, matched = 0, []
        sa, ss = sp.get("rec_agree", 0.0), sp.get("rec_subst", 0.0)
        for sid, r in rec.items():
            o = own["recovery_own"][sid]
            ok_a = np.isfinite(r["agree"]) and r["agree"] >= o["agree"] - sa - 1e-9
            ok_s = (not np.isfinite(o["subst"])) or (np.isfinite(r["subst"]) and r["subst"] >= o["subst"] - ss - 1e-9)
            if ok_a and ok_s:
                n_ok += 1
                matched.append(r["match"])
        out.append(("R6 sealed variables recovered", n_ok == len(rec), n_ok, len(rec)))
        out.append(("_matched", None, sorted(set(m for m in matched if m)), None))
    return out


def own_statistics(task_dir, sealed_dir, seed=0):
    """The true blueprint's own statistics on its network (the lines before the spread), stored as own.json."""
    st, problems = statistics(task_dir, os.path.join(sealed_dir, "true_blueprint.json"), sealed_dir, seed)
    if problems:
        raise RuntimeError("the true blueprint fails the format check: " + "; ".join(problems[:3]))
    own = {k: v for k, v in st.items() if k != "recovery"}
    own["recovery_own"] = {sid: {"agree": r["own_agree"], "subst": r["own_subst"]} for sid, r in st["recovery"].items()}
    own["recovery_self"] = {sid: {"agree": r["agree"], "subst": r["subst"]} for sid, r in st["recovery"].items()}
    return own


# ====================================================================================================== printing
def print_network(net_id, vs, cert=False, file=sys.stdout):
    for name, ok, score, line in vs:
        if name == "_matched":
            if not cert:
                print(f"  {net_id}  diagnostics: box parts that matched a sealed variable: "
                      f"{', '.join(score) if score else 'none'}", file=file)
            continue
        if cert:
            print(f"  {net_id}  {name}: {'pass' if ok else 'fail'}", file=file)
        elif name == "R6 sealed variables recovered":
            print(f"  {net_id}  {name}: {'pass' if ok else 'fail'}  ({score} of {line})", file=file)
        else:
            print(f"  {net_id}  {name}: {'pass' if ok else 'fail'}  score {_f(score)}  line {_f(line)}", file=file)


def _f(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def load_spreads():
    p = os.path.join(SEALED, "spreads.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def score_one(task_dir, bp_path, draw=None):
    task = I.load_task(task_dir)
    draw = draw or task.get("draw")
    sdir = task_dir if draw == "reference" else os.path.join(SEALED, draw, task["id"])   # the public reference
    cert = os.path.exists(os.path.join(SEALED, draw, "CERTIFICATION"))
    st, problems = statistics(task_dir, bp_path, sdir if os.path.isdir(sdir) else None)
    if problems:
        print(f"  {task['id']}  format: fail ({len(problems)} problems; first: {problems[0][:160]})")
        return False
    own_p = os.path.join(sdir, "own.json")
    own = json.load(open(own_p)) if os.path.exists(own_p) else {}
    spreads = load_spreads()
    if not spreads:
        print(f"  {task['id']}  note: tier spreads pending (fewer than three development draws); lines use spread 0")
    vs = verdicts(st, own, spreads, task["tier"])
    print(f"  {task['id']}  format: pass")
    print_network(task["id"], vs, cert)
    return all(ok for name, ok, _s, _l in vs if name != "_matched")


# ====================================================================================================== S5a: accounts to reject
def s5a_check(task_dir, acc_path, sealed_dir, seed=0, n_iia=30):
    """The scorer's own verdict on one must-fail account on a real network: [(test, rejected, score, line)].
    Tests, cheapest first: format (a format-only failure never counts as a rejection); R4 run alone on held-out
    inputs against the task-rule baseline; R5 lookup typing (a lookup at the decision must be at chance on fresh
    inputs: the network's decisions there must not follow the task rule above a permutation control); R3 interchange
    of sampled rule parts against the family-wise random-subspace line; R10 and R9's estimator (pending)."""
    task = I.load_task(task_dir)
    acc = I.load_blueprint(acc_path)
    adir = os.path.dirname(os.path.abspath(acc_path))
    problems = I.validate_blueprint(acc, task, adir)
    if problems:
        return [("format", None, len(problems), 0)]
    net = netlib.load_network(task_dir, task)
    dom = task["domain"]
    H = _load_ctx(os.path.join(sealed_dir, "heldout.npz"), dom)
    rule = np.load(os.path.join(sealed_dir, "rule_heldout.npy"))
    rng = np.random.default_rng(seed)
    out = []
    run = net.run(H, return_sites=True)
    dec = np.asarray(run["outputs"])
    given, choice = core.given_from_network(net, task, acc, H, adir)
    perm = _perm_within(H, dom, rng)
    given_rs = {k: core.take(v, perm) for k, v in given.items() if v is not None}
    ra = core.run_alone(acc_path, task, H, given_rs)["values"][acc["output"]]
    alone = float(np.mean([core.same(a, b) for a, b in zip(ra, dec)]))
    base = float(np.mean(rule == dec))
    out.append(("R4 run alone vs task-rule baseline", alone < base, alone, base))
    for p in acc["parts"]:
        if p["type"] != "lookup" or p["location"]["site"] != "out":
            continue
        keys = {str(k) for k, _ in p["table"]["entries"]}
        fresh = [i for i in range(len(dec)) if I.canonical_key(dom, H, i) not in keys]
        if len(fresh) < 20:
            out.append(("R5 lookup at chance on fresh inputs", None, len(fresh), 20))
            continue
        a = float(np.mean(rule[fresh] == dec[fresh]))
        ctrl = [float(np.mean(rule[fresh] == rng.permutation(dec[fresh]))) for _ in range(200)]
        line = float(np.percentile(ctrl, 99))
        out.append(("R5 lookup at chance on fresh inputs", a > line, a, line))
    if not any(r for _, r, _, _ in out if r):
        rules = [p for p in acc["parts"] if p["type"] == "rule"]
        pick = [rules[i] for i in sorted(rng.choice(len(rules), min(n_iia, len(rules)), replace=False))] if rules else []
        pairs = core.make_pairs(H, dom, 400, rng, equal_len=False)
        tested, preps, scores = [], {}, {}
        for p in pick:
            prep = core.iia_prepare(task, acc, H, pairs, p, given, choice)
            if len(prep["decisive"]) >= MIN_DECISIVE:
                tested.append(p); preps[p["id"]] = prep
                scores[p["id"]] = core.iia_score(net, task, prep, p["location"], adir)
        if tested:
            crng = np.random.default_rng(10_000 + seed)
            src_runs = {pid: net.run(preps[pid]["cs"], return_sites=True) for pid in preps}
            best = []
            for _ in range(40):
                m = -np.inf
                for p in tested:
                    v = core.iia_score(net, task, preps[p["id"]], core.random_subspace_loc(p["location"], task, crng),
                                       adir, src_run=src_runs[p["id"]])
                    m = max(m, v if np.isfinite(v) else -np.inf)
                best.append(m)
            line = float(np.percentile(best, 99))
            worst = float(np.nanmin(list(scores.values())))
            out.append(("R3 interchange of sampled parts above control", worst <= line, worst, line))
            out.append(("_iia_detail", None, {pid: round(v, 3) for pid, v in scores.items()}, len(tested)))
    out.append(("R10 bounded reader", None, np.nan, np.nan))
    out.append(("R9 answering without running (X0)", None, np.nan, np.nan))
    return out


# ====================================================================================================== the hook
def hook(draw, start_path):
    """Gate 1: the builder's checker (Box.verify) on the sealed true blueprints and the S5a accounts. Counts only."""
    start = json.load(open(start_path))
    mod, cls = start["entry"].split(":")
    box = getattr(importlib.import_module(mod), cls)(start.get("config", {}))
    ddir = os.path.join(SEALED, draw)
    counts = {}
    for net_id in sorted(os.listdir(os.path.join(DRAWS, draw))):
        task_dir = os.path.join(DRAWS, draw, net_id)
        if not os.path.isfile(os.path.join(task_dir, "task.json")):
            continue
        tier = I.load_task(task_dir)["tier"]
        c = counts.setdefault(tier, [0, 0, 0, 0])
        tb = os.path.join(ddir, net_id, "true_blueprint.json")
        if os.path.exists(tb):
            c[1] += 1
            c[0] += _quiet(box.verify, task_dir, tb) == "certified"
        adir = os.path.join(ddir, net_id, "accounts")
        if os.path.isdir(adir):
            for a in sorted(os.listdir(adir)):
                c[3] += 1
                c[2] += _quiet(box.verify, task_dir, os.path.join(adir, a)) == "rejected"
    for tier, (x, n, y, m) in sorted(counts.items()):
        print(f"{tier}: certified {x} of {n} true blueprints, rejected {y} of {m} accounts")


def _quiet(fn, *a):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            return fn(*a)
        except Exception:  # noqa: BLE001  (a checker that crashes neither certifies nor rejects)
            return "error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--blueprint")
    ap.add_argument("--draw")
    ap.add_argument("--blueprints")
    ap.add_argument("--hook", action="store_true")
    ap.add_argument("--start")
    a = ap.parse_args()
    if a.hook:
        hook(a.draw, a.start)
    elif a.task:
        score_one(a.task, a.blueprint)
    else:
        ok = True
        for net_id in sorted(os.listdir(os.path.join(DRAWS, a.draw))):
            bpp = os.path.join(a.blueprints, net_id, "blueprint.json")
            if os.path.exists(bpp):
                ok &= score_one(os.path.join(DRAWS, a.draw, net_id), bpp, a.draw)
            else:
                print(f"  {net_id}  no blueprint")
                ok = False
        print("draw:", "pass" if ok else "fail")


if __name__ == "__main__":
    main()
