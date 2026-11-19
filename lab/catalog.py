"""The catalog of constructions: the best construction of each network with its certificate, and a playground that
runs the real network and the construction side by side. Job: make every certified construction something a person
can inspect and test, not just a report (Shreyas, Nov 17). Served by server.py at /catalog.

  entries()                 one card per network that has a judged construction (the best run), plus its unit map
                            (per-part confidence, CHANGES T5); a network with a unit map but no construction gets a
                            card of its own ("unit map only")
  play(target, run, cfg)    one input through both; optionally an interchange (a variable taken from another input)
"""
import ast, json
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
CONS = LAB / "constructions"
MAPS = LAB.parent / "results" / "glass" / "units" / "maps"     # unit maps written by lab/unitmap.py (read only here)
import judge   # noqa: E402
import nets    # noqa: E402

_cache = {}     # (target, run) -> fitted alignment for the playground


def _meta(tid):
    return json.loads((LAB / "models" / tid / "meta.json").read_text())


def _description(src):
    try:
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "DESCRIPTION" for t in node.targets):
                return ast.literal_eval(node.value).strip()
    except (SyntaxError, ValueError):
        pass
    return ""


def _rank(rep):
    v = rep["verdict"]
    return (bool(v.get("complete")), bool(v.get("verified")), v.get("stages_verified", 0), rep["behaviour"]["match"])


def _runs(tid):
    out = []
    for d in sorted((CONS / tid).glob("*/report.json")):
        rdir = d.parent
        cur = rdir / "report_current.json"            # the same construction re-judged by the current judge, if done
        d = cur if cur.exists() else d
        try:
            run = json.loads((rdir / "run.json").read_text())
        except (OSError, ValueError):
            continue
        if run.get("config", {}).get("variant", "original") != "original":
            continue                      # shuffled/untrained controls are a different network: not entries
        # (a prior-only construction, written without the network, is an entry if the judge certifies it against
        #  the network: it is then a valid account of it, and the card says how it was written)
        out.append((json.loads(d.read_text()), rdir, run))
    return out


def _unit_map(tid):
    """The network's unit map as written by lab/unitmap.py, or None. Nothing is recomputed."""
    f = MAPS / f"{tid}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text()); m, c = d["map"], d["control"]
    return {"eps95": m.get("eps95"), "coverage_history": m["coverage_history"], "coverage_raw": m["coverage_raw"],
            "control_history": c["coverage_history"], "control_raw": c["coverage_raw"],
            "per_unit": [{"neuron": int(j), "label": lab} for j, lab in sorted(m["per_unit"].items(), key=lambda kv: int(kv[0]))],
            "cells": [{k: x.get(k) for k in ("t", "unit", "label", "fit", "sufficiency", "importance", "rel_error", "substitution_grade")}
                      for x in m["cells"]],
            "substitution_grade": sum(1 for x in m["cells"] if x.get("substitution_grade")),
            "file": f"results/glass/units/maps/{tid}.md"}


def entries():
    cards = []
    for tdir in sorted(CONS.iterdir()):
        if not tdir.is_dir() or tdir.name.startswith("_") or not (LAB / "models" / tdir.name / "meta.json").exists():
            continue
        runs = _runs(tdir.name)
        if not runs:
            continue
        rep, rdir, run = max(runs, key=lambda r: _rank(r[0]))
        m, v = _meta(tdir.name), rep["verdict"]
        src = (rdir / "construction.py").read_text()
        cards.append({
            "target": tdir.name, "run": rdir.name, "name": m.get("name", tdir.name), "family": m.get("family", ""),
            "params": m.get("params"), "task": m.get("answer", {}).get("short", ""), "forward": m["forward"],
            "playable": m["forward"] != "lm", "runs": len(runs),
            "status": "complete" if v.get("complete") else ("verified" if v.get("verified") else "not verified"),
            "stages": f"{v.get('stages_verified', 0)}/{v.get('stages_total', 0)}",
            "behaviour": rep["behaviour"]["match"], "coverage": rep.get("coverage"), "fails": v.get("fails", [])[:12],
            "description": _description(src), "source": src,
            "report": next(((rdir / f).read_text() for f in ("report_current.txt", "report.txt") if (rdir / f).exists()), ""),
            "judged_by": "current judge" if (rdir / "report_current.json").exists() else "the judge of its run",
            "cost": run.get("cost"), "steps": len(run.get("steps", [])),
            "written": "without access to the network (prior-only control: told the task, one submission)" if not run.get("config", {}).get("weights", True)
                       else ("by a blinded reader with the weights" if not run.get("config", {}).get("told") else "by a reader told the task, with the weights"),
            "unit_map": _unit_map(tdir.name),
        })
    have = {c["target"] for c in cards}
    for f in sorted(MAPS.glob("*.json")) if MAPS.exists() else []:
        tid = f.stem
        if tid in have or not (LAB / "models" / tid / "meta.json").exists():
            continue
        m = _meta(tid)
        cards.append({"target": tid, "run": None, "name": m.get("name", tid), "family": m.get("family", ""),
                      "params": m.get("params"), "task": m.get("answer", {}).get("short", ""), "forward": m["forward"],
                      "playable": False, "runs": 0, "status": "unit map only", "stages": "–", "behaviour": None,
                      "coverage": None, "fails": [], "description": "", "source": "", "report": "", "judged_by": "",
                      "cost": None, "steps": 0, "written": "", "unit_map": _unit_map(tid)})
    order = {"complete": 0, "verified": 1, "not verified": 2, "unit map only": 3}
    return sorted(cards, key=lambda c: (order[c["status"]], c["params"] or 0))


def _fit(tid, rid):
    """The construction's variables, the network's activations and the affine alignment at each named site, on the
    target's own inputs (as the judge fits them)."""
    key = (tid, rid)
    if key in _cache:
        return _cache[key]
    meta, W, X = judge.load_target(tid)
    src = (CONS / tid / rid / "construction.py").read_text()
    model, align = judge.load_construction(src)
    X = X[:4000]
    acts = {}
    nets.forward(meta["forward"], W, X, acts=acts)
    _, vals = judge.run_model(model, X)
    sites = {}
    for s in sorted({s for w in align.values() for s in w}):
        names = [v for v in vals if s in align.get(v, [])]
        if not names or acts[s].std(0).max() <= 1e-9:
            continue
        cols, k = {}, 0
        for v in names:
            cols[v] = (k, k + vals[v].shape[1]); k += vals[v].shape[1]
        V = np.concatenate([vals[v] for v in names], 1)
        c, vmu, M, D = judge.fit_alignment(V, acts[s], np.arange(len(X)))
        sites[s] = {"names": names, "cols": cols, "c": c, "vmu": vmu, "M": M}
    _cache[key] = {"meta": meta, "W": W, "X": X, "model": model, "align": align, "sites": sites,
                   "groups": meta.get("out_groups")}
    return _cache[key]


def _out(o, groups):
    judge.GROUPS = groups
    return {"values": np.round(o[0], 4).tolist(), "decision": judge.decide(o)[0].tolist()}


def play(tid, rid, cfg):
    """cfg: {"x": [...] or absent (a random domain input, "seed"), "swap": {"variable": name, "x": [...] or absent}}"""
    F = _fit(tid, rid)
    meta, W, X, model = F["meta"], F["W"], F["X"], F["model"]
    rng = np.random.default_rng(cfg.get("seed"))
    def pick(v):
        if v is None:
            return X[rng.integers(len(X))][None].copy()
        a = np.asarray(v, dtype=np.float64).reshape(1, -1)
        if np.issubdtype(X.dtype, np.integer):
            a = np.rint(a).astype(X.dtype)
        if a.shape[1] != X.shape[1]:
            raise ValueError(f"an input has {X.shape[1]} numbers; got {a.shape[1]}")
        return a
    x = pick(cfg.get("x"))
    acts = {}
    o_net = nets.forward(meta["forward"], W, x, acts=acts)
    o_con, vals = judge.run_model(model, x)
    res = {"x": np.round(x[0], 4).tolist(), "network": _out(o_net, F["groups"]), "construction": _out(o_con, F["groups"]),
           "variables": [{"name": n, "value": np.round(v[0], 4).tolist(), "sites": F["align"].get(n, [])} for n, v in vals.items()],
           "sites": {s: np.round(acts[s][0], 3).tolist()[:64] for s in F["sites"]}}
    res["agree"] = res["network"]["decision"] == res["construction"]["decision"]
    sw = cfg.get("swap")
    if sw and sw.get("variable"):
        name = sw["variable"]
        site = next((s for s, d in F["sites"].items() if name in d["names"]), None)
        if site is None:
            raise ValueError(f"'{name}' is not placed at a site the judge tested")
        xs = pick(sw.get("x"))
        _, vs = judge.run_model(model, xs)
        o_pred, _ = judge.run_model(model, x, {name: vs[name]})                  # the construction's prediction
        d = F["sites"][site]; a, b = d["cols"][name]
        delta = (vs[name] - vals[name]) @ d["M"][a:b]                            # the same change, written into the network
        o_int = nets.forward(meta["forward"], W, x, hook={site: acts[site] + delta})
        res["swap"] = {"variable": name, "site": site, "source_x": np.round(xs[0], 4).tolist(),
                       "source_value": np.round(vs[name][0], 4).tolist(),
                       "construction_predicts": _out(o_pred, F["groups"]), "network_after": _out(o_int, F["groups"])}
        res["swap"]["prediction_right"] = res["swap"]["construction_predicts"]["decision"] == res["swap"]["network_after"]["decision"]
    return res
