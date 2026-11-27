"""The scorer's tests (development copy; the scorer of record is kept privately until certification).

Version scorer-v1. Every test runs on the network with the scorer's own intervention code (bench/accept/netlib.py),
reads parts through their declared linear locations, and evaluates definitions with the bench's frozen vocabulary
(bench/accept/vendor/vocab). Nothing here prints; score.py decides what may be printed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ACCEPT = os.path.dirname(HERE)
for p in (ACCEPT, os.path.join(ACCEPT, "vendor")):
    if p not in sys.path:
        sys.path.insert(0, p)
import vocab  # noqa: E402
import interface as I  # noqa: E402
import netlib  # noqa: E402

VERSION = "scorer-v1"
NUM_TOL = 1e-3


# ====================================================================================================== values
def n_inputs(ctx, domain):
    if domain == "tok":
        return len(ctx["tokens"])
    if domain == "seq":
        return len(ctx["x"])
    if domain == "chess":
        return len(ctx["fens"])
    return len(next(iter(ctx.values())))


def subset(ctx, domain, idx):
    idx = list(idx)
    if domain == "tok":
        return {"tokens": [ctx["tokens"][i] for i in idx]}
    if domain == "chess":
        return {"fens": [ctx["fens"][i] for i in idx]}
    return {k: np.asarray(v)[idx] for k, v in ctx.items()}


def take(vals, idx):
    if isinstance(vals, list):
        return [vals[i] for i in idx]
    return np.asarray(vals)[list(idx)]


def same(a, b):
    """Value equality for one element: numbers within tolerance; "no value" (None) equals only "no value"."""
    if a is None or b is None:
        return a is None and b is None
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= NUM_TOL * max(1.0, abs(fb))
    except (TypeError, ValueError):
        return a == b


def agreement(pred, ref):
    """Per-input agreement in [0, 1] between two batches of values (lists of per-position lists, or arrays). Every
    position counts; "no value" agrees only with "no value", so predicting "undefined" never dodges a test."""
    out = []
    for i in range(len(ref)):
        r = ref[i]
        p = pred[i]
        if isinstance(r, (list, tuple, np.ndarray)) and np.ndim(r) > 0:
            pl, rl = list(p) if isinstance(p, (list, tuple, np.ndarray)) else [p] * len(r), list(r)
            if len(pl) != len(rl):
                out.append(0.0)
                continue
            out.append(np.mean([same(pp, rr) for pp, rr in zip(pl, rl)]) if rl else np.nan)
        else:
            out.append(float(same(p, r)))
    return np.asarray(out, float)


def numeric(vals):
    """Values as a float array (None -> nan); per-position values are concatenated by the caller."""
    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    if isinstance(vals, list):
        return [np.asarray([f(x) for x in row], float) if isinstance(row, (list, tuple)) else f(row) for row in vals]
    return np.asarray(vals, float)


# ====================================================================================================== reading parts
def decode(coords, enc):
    """Coordinates [..., k] -> values. Per-position arrays [N, T, k] give [N, T] (object for categorical)."""
    kind = enc["kind"]
    if kind == "numerical":
        return float(enc.get("scale", 1.0)) * coords[..., 0] + float(enc.get("offset", 0.0))
    if kind == "categorical":
        vals = enc["values"]
        idx = np.argmax(coords, axis=-1)
        nb = enc.get("null_below")              # declared: a reading whose largest coordinate is below it has no value
        live = (np.max(coords, axis=-1) >= float(nb)).ravel() if nb is not None else np.ones(idx.size, bool)
        flat = idx.ravel()
        out = np.empty(flat.shape, dtype=object)
        for j, v in enumerate(flat):
            out[j] = vals[int(v)] if live[j] else None
        return out.reshape(idx.shape)
    return coords


def encode(values, enc, k):
    """Values -> coordinates [..., k] (the inverse of decode; unknown categorical values give zeros)."""
    kind = enc["kind"]
    if kind == "numerical":
        v = np.asarray(values, float)
        return ((v - float(enc.get("offset", 0.0))) / float(enc.get("scale", 1.0)))[..., None]
    if kind == "categorical":
        vals = enc["values"]
        mag = float(enc.get("magnitude", 1.0))
        arr = np.asarray(values, dtype=object)
        out = np.zeros(arr.shape + (k,))
        for ix in np.ndindex(arr.shape):
            v = arr[ix]
            for j, c in enumerate(vals):
                if same(v, c) or (not isinstance(v, (int, float)) and v == c):
                    out[ix + (j,)] = mag
                    break
        return out
    return np.asarray(values, float)


def site_spec(task, name):
    return next(s for s in task["sites"] if s["name"] == name)


def part_to_values(task, part, coords, ctx):
    """Network coordinates of a part -> values in the vocabulary's representation for the task's domain."""
    dom = task["domain"]
    pos = part["location"].get("positions", "all")
    vals = decode(coords, part["encoding"])
    if dom == "tok" and pos == "all":
        off = site_spec(task, part["location"]["site"]).get("value_offset", 0)
        return [[_py(vals[i, off + j]) for j in range(len(ctx["tokens"][i]))] for i in range(len(ctx["tokens"]))]
    if pos == "all":
        off = site_spec(task, part["location"]["site"]).get("value_offset", 0)
        n = vals.shape[1] - off
        return np.asarray(vals[:, off:off + n], float) if part["encoding"]["kind"] == "numerical" else vals[:, off:]
    return np.asarray(vals, float) if part["encoding"]["kind"] == "numerical" else vals


def values_to_coords(task, part, values, ctx, T_full=None):
    dom = task["domain"]
    pos = part["location"].get("positions", "all")
    k = I.location_dim(part["location"], task)
    site = site_spec(task, part["location"]["site"])
    if dom == "tok" and pos == "all":
        off = site.get("value_offset", 0)
        T = T_full or site["shape"][0]
        out = np.zeros((len(values), T, k))
        for i, row in enumerate(values):
            c = encode([None if v is None else v for v in row], part["encoding"], k)
            out[i, off:off + len(row)] = c
        return out
    if pos == "all":
        off = site.get("value_offset", 0)
        c = encode(values, part["encoding"], k)
        out = np.zeros((c.shape[0], site["shape"][0], k))
        out[:, off:off + c.shape[1]] = c
        return out
    return encode(values, part["encoding"], k)


def _py(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


def read_parts(net, task, bp, ctx, bp_dir=".", run=None):
    """The network's own value of every part, read through its location: {part id: values}, plus the run."""
    run = run or net.run(ctx, return_sites=True)
    out = {}
    for p in bp["parts"]:
        loc = p["location"]
        if loc["map"]["class"] == "param_component":
            continue
        coords = netlib.read_coords(run["sites"][loc["site"]], loc, task, bp_dir)
        out[p["id"]] = part_to_values(task, p, coords, ctx)
    return out, run


# ====================================================================================================== the abstract model
def compile_library(bp):
    ops = {}
    for name, e in (bp.get("library") or {}).items():
        if e.get("kind") == "operation":
            ops[name] = vocab.compile_operation(e["source"])
    return ops


def evaluate_blueprint(bp, task, ctx, given=None, overrides=None, choice_values=None):
    """Every part's value on ctx, in dependency order.
    given: {part id: values} for lookups off-table, unresolved and irrelevant parts (the network's own values when
    testing on the network; resampled values when running alone). overrides: {part id: values} (interventions).
    choice_values: {part id: values} for parts marked `choice` (the network's actual choice on the network)."""
    dom = vocab.get(task["domain"])
    lib = json.loads(json.dumps(bp.get("library") or {}))
    vocab.check_library(dom, lib)
    ops = compile_library(bp)
    types = {p["id"]: p["vtype"] for p in bp["parts"]}
    given, overrides, choice_values = given or {}, overrides or {}, choice_values or {}
    vals = {}
    for p in I.order_parts(bp["parts"]):
        pid = p["id"]
        if pid in overrides:
            vals[pid] = overrides[pid]
            continue
        if pid in choice_values:
            vals[pid] = choice_values[pid]
            continue
        t = p["type"]
        if t == "rule":
            vals[pid] = vocab.evaluate(p["definition"], dom, ctx, parts=vals, part_types=types, library=lib,
                                       ops_code=ops)
        elif t == "lookup":
            vals[pid] = lookup_values(p, task, ctx, given.get(pid))
        else:
            vals[pid] = given.get(pid)
    return vals


def lookup_values(p, task, ctx, fallback):
    tab = p["table"]
    entries = {str(k): v for k, v in tab["entries"]}
    n = n_inputs(ctx, task["domain"])
    out = []
    for i in range(n):
        key = I.canonical_key(task["domain"], ctx, i)
        if key in entries:
            out.append(entries[key])
        elif "default" in tab:
            out.append(tab["default"])
        else:
            out.append(None if fallback is None else take(fallback, [i])[0])
    if task["domain"] == "seq" or task["domain"] == "chess":
        try:
            return np.asarray(out, float)
        except (TypeError, ValueError):
            return np.asarray(out, dtype=object)
    return out


def network_output_values(task, run):
    return run["outputs"]


# ====================================================================================================== pairs
def make_pairs(ctx, domain, n, rng, idx=None, equal_len=True):
    """(base, source) pairs. Token tasks pair inputs of equal length by default (resampling); interchange pairs may
    differ in length (equal_len=False): positions beyond the shorter input keep the base's values."""
    N = n_inputs(ctx, domain)
    idx = list(range(N)) if idx is None else list(idx)
    pairs = []
    if domain == "tok" and not equal_len:
        half = make_pairs(ctx, domain, n // 2, rng, idx, equal_len=True)
        while len(half) < n:
            a, b = rng.choice(len(idx), 2, replace=False)
            half.append((idx[a], idx[b]))
        return half
    if domain == "tok":
        by_len = {}
        for i in idx:
            by_len.setdefault(len(ctx["tokens"][i]), []).append(i)
        groups = [g for g in by_len.values() if len(g) > 1]
        while len(pairs) < n and groups:
            g = groups[int(rng.integers(len(groups)))]
            a, b = rng.choice(len(g), 2, replace=False)
            pairs.append((g[a], g[b]))
    else:
        while len(pairs) < n:
            a, b = rng.choice(len(idx), 2, replace=False)
            pairs.append((idx[a], idx[b]))
    return pairs


# ====================================================================================================== R3: interventions
def _swap_edit(loc, task, src_act, bp_dir):
    """Interchange: the location's coordinates from the source activation, the complement from the base."""
    coords = netlib.read_coords(src_act, loc, task, bp_dir)
    return netlib.write_edit(loc, task, coords, bp_dir)


def iia_prepare(task, bp, ctx, pairs, part, given, choice):
    """The blueprint's own predictions for swapping one part between the inputs of each pair, and the decisive pairs
    (those where it predicts a change of output)."""
    dom = task["domain"]
    B = [a for a, _ in pairs]
    S = [b for _, b in pairs]
    cb, cs = subset(ctx, dom, B), subset(ctx, dom, S)
    gb, gs, chb, chs = _take_all(given, B), _take_all(given, S), _take_all(choice, B), _take_all(choice, S)
    pid = part["id"]
    ab_base = evaluate_blueprint(bp, task, cb, given=gb, choice_values=chb)
    ab_src = evaluate_blueprint(bp, task, cs, given=gs, choice_values=chs)
    ov = ab_src[pid]
    if dom == "tok" and part["location"].get("positions", "all") == "all":
        ov = [[s_[j] if j < len(s_) else b_[j] for j in range(len(b_))] for s_, b_ in zip(ab_src[pid], ab_base[pid])]
    pred = evaluate_blueprint(bp, task, cb, given=gb, overrides={pid: ov}, choice_values=chb)[bp["output"]]
    changed = agreement(pred, ab_base[bp["output"]]) < 1.0
    decisive = [i for i in range(len(pairs)) if changed[i]]
    return {"cb": cb, "cs": cs, "pred": pred, "decisive": decisive}


def iia_score(net, task, prep, loc, bp_dir=".", src_run=None):
    """Share of decisive pairs where swapping the location's coordinates gives the predicted output."""
    if not prep["decisive"]:
        return np.nan
    src_run = src_run or net.run(prep["cs"], return_sites=True)
    if task["domain"] == "tok" and loc.get("positions", "all") == "all":
        edit = {loc["site"]: _swap_edit_tok(loc, task, src_run["sites"][loc["site"]], prep["cb"], prep["cs"], bp_dir)}
    else:
        edit = {loc["site"]: _swap_edit(loc, task, src_run["sites"][loc["site"]], bp_dir)}
    out = net.run(prep["cb"], edits=edit)["outputs"]
    hits = agreement(take(prep["pred"], prep["decisive"]), take(out, prep["decisive"]))
    return float(np.nanmean(hits)) if np.isfinite(hits).any() else np.nan


def _swap_edit_tok(loc, task, src_act, cb, cs, bp_dir):
    """Interchange for per-position token parts: source coordinates at the positions both inputs have, base
    coordinates beyond the source's length."""
    site = site_spec(task, loc["site"])
    off = site.get("value_offset", 0)
    T_full = site["shape"][0]
    src_c = netlib.read_coords(src_act, loc, task, bp_dir)

    def fn(a):
        base_c = netlib.read_coords(a, loc, task, bp_dir)
        new = base_c.copy()
        for i in range(len(new)):
            m = min(len(cs["tokens"][i]), len(cb["tokens"][i])) + off
            m = min(m, src_c.shape[1], new.shape[1])
            new[i, off:m] = src_c[i, off:m]
        return netlib.write_edit(loc, task, new, bp_dir)(a)
    return fn


def _resample_edit(loc, task, src_run, cb, cs, bp_dir):
    if task["domain"] == "tok" and loc.get("positions", "all") == "all":
        return _swap_edit_tok(loc, task, src_run["sites"][loc["site"]], cb, cs, bp_dir)
    return _swap_edit(loc, task, src_run["sites"][loc["site"]], bp_dir)


def random_subspace_loc(loc, task, rng):
    site = site_spec(task, loc["site"])
    k = I.location_dim(loc, task)
    pos = loc.get("positions", "all")
    d = int(np.prod(site["shape"])) if pos == "flat" else site["shape"][-1]
    Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return {"site": loc["site"], "positions": pos, "map": {"class": "subspace", "basis": Q.tolist()}}


def substitution(net, task, bp, ctx, part, rng, bp_dir=".", given=None, choice=None):
    """Share of the part's effect restored when the blueprint's own value replaces the network's: 1 - d(sub)/d(res),
    where d is the output change against the clean run, and res resamples the part from another input."""
    dom = task["domain"]
    N = n_inputs(ctx, dom)
    pairs = make_pairs(ctx, dom, N, rng, equal_len=False)
    B = [a for a, _ in pairs]
    S = [b for _, b in pairs]
    cb, cs = subset(ctx, dom, B), subset(ctx, dom, S)
    clean = net.run(cb)["outputs"]
    ab = evaluate_blueprint(bp, task, cb, given=_take_all(given, B), choice_values=_take_all(choice, B))
    loc = part["location"]
    T_full = site_spec(task, loc["site"])["shape"][0]
    new = values_to_coords(task, part, ab[part["id"]], cb, T_full)
    sub_out = net.run(cb, edits={loc["site"]: netlib.write_edit(loc, task, new, bp_dir)})["outputs"]
    src_run = net.run(cs, return_sites=True)
    res_out = net.run(cb, edits={loc["site"]: _resample_edit(loc, task, src_run, cb, cs, bp_dir)})["outputs"]
    d_sub = 1.0 - agreement(sub_out, clean) if task["behaviour"]["metric"] == "accuracy" else _sq(sub_out, clean)
    d_res = 1.0 - agreement(res_out, clean) if task["behaviour"]["metric"] == "accuracy" else _sq(res_out, clean)
    if np.nanmean(d_res) <= 1e-12:
        return np.nan
    return float(1.0 - np.nanmean(d_sub) / np.nanmean(d_res))


def _sq(a, b):
    out = []
    for x, y in zip(a, b):
        x, y = np.asarray(numeric([x])[0], float), np.asarray(numeric([y])[0], float)
        out.append(float(np.nanmean((x - y) ** 2)))
    return np.asarray(out)


def _take_all(d, idx):
    if not d:
        return {}
    return {k: take(v, idx) for k, v in d.items() if v is not None}


# ====================================================================================================== R4: run alone
def run_alone(bp_path, task, ctx, given):
    """The blueprint alone, in a separate process given only the blueprint, the task's metadata and the inputs
    (never the weights). given: resampled values for lookups off-table and for unresolved and irrelevant parts."""
    with tempfile.TemporaryDirectory() as tmp:
        bp = json.load(open(bp_path))
        json.dump(bp, open(os.path.join(tmp, "bp.json"), "w"))
        meta = {k: task[k] for k in ("id", "domain", "behaviour", "sites")}
        json.dump(meta, open(os.path.join(tmp, "task.json"), "w"))
        json.dump(_jsonable(ctx), open(os.path.join(tmp, "ctx.json"), "w"))
        json.dump({k: _jsonable(v) for k, v in (given or {}).items()}, open(os.path.join(tmp, "given.json"), "w"))
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "run_alone.py"), tmp], capture_output=True,
                           text=True, timeout=1800, cwd=tmp)
        if r.returncode != 0:
            raise RuntimeError("run alone failed: " + r.stderr[-400:])
        return json.load(open(os.path.join(tmp, "out.json")))


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


# ====================================================================================================== R4: kept parts suffice
def kept_parts(net, task, bp, ctx, pairs, bp_dir=".", keep_types=("rule", "lookup", "unresolved")):
    """Per-input recovery when everything the blueprint calls irrelevant or does not mention is resampled from the
    pair's source input. At each site holding a kept part, the span of the parts kept there (and, on a shared stream,
    at earlier sites of that stream) keeps its base value; the complement takes the source's value. Recovery per
    input: 1 - E(scrubbed) / E(null), E the output loss against the clean run, null = every such site resampled."""
    dom = task["domain"]
    B = [a for a, _ in pairs]
    S = [b for _, b in pairs]
    cb, cs = subset(ctx, dom, B), subset(ctx, dom, S)
    clean = net.run(cb)["outputs"]
    src_run = net.run(cs, return_sites=True)
    order = {s["name"]: s.get("order", 0) for s in task["sites"]}
    stream = {s["name"]: s.get("stream") for s in task["sites"]}
    kept = [p for p in bp["parts"] if p["type"] in keep_types and p["location"]["map"]["class"] == "subspace"]
    sites = sorted({p["location"]["site"] for p in kept}, key=lambda s: order[s])
    edits, null_edits = {}, {}
    for s in sites:
        mats = []
        for p in kept:
            ps = p["location"]["site"]
            if ps == s or (stream[s] and stream[ps] == stream[s] and order[ps] <= order[s]):
                mats.append(_location_matrix(p["location"], task))
        edits[s] = _keep_edit(mats, src_run["sites"][s], p_flat=False)
        null_edits[s] = (lambda src: (lambda a: _fit_T(src, a)))(src_run["sites"][s])
    scrub = net.run(cb, edits=edits)["outputs"]
    null = net.run(cb, edits=null_edits)["outputs"]
    if task["behaviour"]["metric"] == "accuracy":
        e_s, e_n = 1.0 - agreement(scrub, clean), 1.0 - agreement(null, clean)
    else:
        e_s, e_n = _sq(scrub, clean), _sq(null, clean)
    rec = np.where(e_n > 1e-12, 1.0 - e_s / np.where(e_n > 1e-12, e_n, 1.0), np.nan)
    return rec


def _fit_T(src, a):
    out = a.copy()
    T = min(a.shape[1], src.shape[1])
    out[:, :T] = src[:, :T]
    return out


def _location_matrix(loc, task):
    site = site_spec(task, loc["site"])
    return netlib.basis_of(loc["map"], site), loc.get("positions", "all")


def _keep_edit(mats, src_act, p_flat=False):
    """Keep the union span of the given bases (at their positions) from the base run; the rest from the source."""
    def fn(a):
        T = min(a.shape[1], src_act.shape[1])
        out = a.copy()
        src = src_act[:, :T]
        base = a[:, :T]
        d = a.shape[-1]
        # per position: projector onto the union of the bases located at that position
        Pall = np.zeros((T, d, d))
        for Bm, pos in mats:
            Qm, _ = np.linalg.qr(Bm) if Bm.shape[1] else (Bm, None)
            for t in (range(T) if pos == "all" else [p for p in pos if p < T]):
                Pall[t] += Qm @ Qm.T
        for t in range(T):
            U, s, _ = np.linalg.svd(Pall[t])
            U = U[:, s > 1e-8]
            Pt = U @ U.T
            out[:, t] = base[:, t] @ Pt + src[:, t] @ (np.eye(d) - Pt)
        return out
    return fn


# ====================================================================================================== recovery (R6, R8)
def recovery_scores(net, task, bp, sealed_bp, ctx, rng, bp_dir=".", sealed_dir="."):
    """For each sealed variable: the best box part's agreement with the network's variable at the sealed location
    (correlation for numbers), and the substitution score of the box part's values written at the sealed location.
    Returns {sealed id: {"agree", "subst", "match": box part id or None, "own_agree", "own_subst"}}."""
    out = {}
    sealed_net, run = read_parts(net, task, sealed_bp, ctx, sealed_dir)
    box_given, box_choice = given_from_network(net, task, bp, ctx, bp_dir)
    box_vals = evaluate_blueprint(bp, task, ctx, given=box_given, choice_values=box_choice)
    s_given, s_choice = given_from_network(net, task, sealed_bp, ctx, sealed_dir)
    own_vals = evaluate_blueprint(sealed_bp, task, ctx, given=s_given, choice_values=s_choice)
    for sp in sealed_bp["parts"]:
        sid = sp["id"]
        target = sealed_net[sid]
        own_a = _agree_or_corr(own_vals[sid], target, sp)
        own_s = subst_at(net, task, sp, own_vals[sid], ctx, rng, sealed_dir)
        best = (None, -np.inf, np.nan)
        for p in bp["parts"]:
            if p["vtype"] != sp["vtype"] or box_vals.get(p["id"]) is None:
                continue
            a = _agree_or_corr(box_vals[p["id"]], target, sp)
            if np.isfinite(a) and a > best[1]:
                best = (p["id"], a, np.nan)
        s = subst_at(net, task, sp, box_vals[best[0]], ctx, rng, sealed_dir) if best[0] else np.nan
        out[sid] = {"agree": best[1], "subst": s, "match": best[0], "own_agree": own_a, "own_subst": own_s}
    return out


def _agree_or_corr(vals, target, sp):
    if sp["encoding"]["kind"] == "numerical":
        a = _flat_num(vals)
        b = _flat_num(target)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
            return float(np.mean(np.abs(a[ok] - b[ok]) < NUM_TOL)) if ok.any() else np.nan
        return float(np.corrcoef(a[ok], b[ok])[0, 1])
    ag = agreement(vals, target)
    return float(np.nanmean(ag)) if np.isfinite(ag).any() else np.nan


def _flat_num(v):
    if isinstance(v, list):
        return np.concatenate([np.atleast_1d(np.asarray(numeric([r])[0], float)) for r in v]) if v else np.zeros(0)
    return np.asarray(v, float).ravel()


def subst_at(net, task, sp, values, ctx, rng, sealed_dir):
    """Write values at the sealed part's location: share of the location's effect restored."""
    dom = task["domain"]
    N = n_inputs(ctx, dom)
    pairs = make_pairs(ctx, dom, N, rng, equal_len=False)
    B = [a for a, _ in pairs]
    S = [b for _, b in pairs]
    cb, cs = subset(ctx, dom, B), subset(ctx, dom, S)
    loc = sp["location"]
    T_full = site_spec(task, loc["site"])["shape"][0]
    clean = net.run(cb)["outputs"]
    new = values_to_coords(task, sp, take(values, B), cb, T_full)
    sub_out = net.run(cb, edits={loc["site"]: netlib.write_edit(loc, task, new, sealed_dir)})["outputs"]
    src_run = net.run(cs, return_sites=True)
    res_out = net.run(cb, edits={loc["site"]: _resample_edit(loc, task, src_run, cb, cs, sealed_dir)})["outputs"]
    acc = task["behaviour"]["metric"] == "accuracy"
    d_sub = 1.0 - agreement(sub_out, clean) if acc else _sq(sub_out, clean)
    d_res = 1.0 - agreement(res_out, clean) if acc else _sq(res_out, clean)
    if np.nanmean(d_res) <= 1e-12:
        return np.nan
    return float(1.0 - np.nanmean(d_sub) / np.nanmean(d_res))


def given_from_network(net, task, bp, ctx, bp_dir="."):
    """Values the abstract model takes from the network when tested on it: lookups off-table, unresolved and
    irrelevant parts (their network values), and parts marked `choice` (the network's actual choice)."""
    vals, run = read_parts(net, task, bp, ctx, bp_dir)
    given = {p["id"]: vals.get(p["id"]) for p in bp["parts"] if p["type"] != "rule"}
    choice = {p["id"]: _choice_values(task, p, run) for p in bp["parts"] if p.get("choice")}
    return given, choice


def _choice_values(task, p, run):
    return run["outputs"]


def ancestors(bp, pid):
    ups = {p["id"]: p.get("upstream", []) for p in bp["parts"]}
    seen, stack = set(), list(ups.get(pid, []))
    while stack:
        u = stack.pop()
        if u not in seen:
            seen.add(u)
            stack += ups.get(u, [])
    return seen
