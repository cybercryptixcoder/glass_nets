"""The judge: tests a construction against a network. The same tests for every network; nothing here knows any task.

A CONSTRUCTION is a small, standalone Python file with two things in it:

    def model(x, iv):
        # x: the inputs, an array [B, n_in]. Returns the predicted network output, an array [B, n_out].
        # Every named intermediate variable goes through iv:   v = iv("name", value)   with value shaped [B] or [B, k].
        # iv returns the value to use from then on, in the same shape (during an intervention test it returns a
        # replaced value), so later steps must be computed from what iv returned.

    ALIGN = {"name": ["site", ...], ...}     # where in the network each variable lives (site names: nets.site_table)

The file may use numpy (np) and math, and nothing else: it cannot see the weights. Its length is its description length.

The judge measures, on inputs held out from fitting:
  1. BEHAVIOUR      run the construction in place of the network: do the outputs agree?
  2. ALIGNMENT      at each site named in ALIGN, fit one affine map from the variables to the activations
                    (activation ~ c + variables @ M; this declared, restricted map is the alignment) and report how
                    much of the activation it explains.
  3. SUBSTITUTION   replace the site's activation with the construction's prediction (c + variables @ M) and run the
                    rest of the network ("keep only what it names"); also project the real activation onto the named
                    subspace; also remove the named subspace ("remove only what it names"), next to mean-ablating the
                    whole site and removing a random subspace of the same size. The named subspace must also be
                    compact: not much larger than the number of principal components of the site that suffice.
  4. INTERVENTION   interchange: everything the construction names at a site is copied from a source input's run
                    into a base input's run (read with a linear probe, written along the alignment directions), and
                    the network's new output is compared with what the construction predicts when those variables
                    take their source values. Then two stricter tests that make states no input produces:
                    partial (one variable copied while the site's other variables stay) and edit (a random half of
                    the coordinates of everything named at the site copied). Where both apply, passing either one
                    is enough, so that splitting a variable into parts never lowers a construction's score.
                    Controls: the same swap in a random subspace; predicting "no change"; predicting the source
                    output; attribution patching (the first-order estimate of the same change, on the real network).

Run:  python3 lab/judge.py <target id> <construction.py> [--variant original|shuffled|untrained|permuted] [--json out.json]
"""
import json, math, sys, time, traceback
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import nets  # noqa: E402

MAX_SOURCE_CHARS = 16000
PASS = {"behaviour": 0.95,       # share of inputs on which the construction and the network make the same decision
        "substitution": 0.90,    # decisions kept when a site is replaced by the construction's prediction
        "necessity": 0.5,        # removing the named subspace must do this share of the damage of mean-ablating the site
        "compact": 2,            # the named subspace may be at most this many times the smallest sufficient PCA subspace (or 4 more)
        "interchange": 0.90,     # interchange predictions right, among cases where the network's decision changed
        "edit": 0.50,            # partial-interchange and edit predictions right, among cases where the decision changed
        # Glass Networks additions (docs/V2_CONSTRUCTOR_PLAN.md items 2-3); each is a construction-level claim:
        "joint": 0.90,           # decisions kept when EVERY named site is replaced by the construction's prediction at once
        "worst_slice": 0.80,     # behaviour on the worst slice (the inputs on which the network makes one decision)
        "errors": 0.50}          # share of the network's own mistakes (vs the task's labels) the construction reproduces
MIN_SLICE = 30                   # slices and error sets smaller than this are reported but carry no requirement


class ConstructionError(Exception):
    pass


# ---------------------------------------------------------------- loading and running a construction
def load_construction(source):
    if len(source) > MAX_SOURCE_CHARS:
        raise ConstructionError(f"The construction is {len(source):,} characters; the limit is {MAX_SOURCE_CHARS:,}. "
                                "A construction is a compact program, not a table of the network's values.")
    import builtins
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if (globals or {}).get("__name__") == "__construction__" and name.split(".")[0] not in ("numpy", "math", "itertools", "functools"):
            raise ImportError(f"a construction may import only numpy and math, not '{name}'")
        return real_import(name, globals, locals, fromlist, level)

    env = {"__name__": "__construction__", "np": np, "math": math}
    builtins.__import__ = guarded
    try:
        exec(compile(source, "<construction>", "exec"), env)
    except Exception as e:  # noqa: BLE001
        raise ConstructionError("The construction file did not run:\n" + _short_tb()) from e
    finally:
        builtins.__import__ = real_import
    if not callable(env.get("model")):
        raise ConstructionError("The construction must define  def model(x, iv): ...")
    align = env.get("ALIGN")
    if not isinstance(align, dict) or not align:
        raise ConstructionError('The construction must define  ALIGN = {"variable name": ["site", ...], ...}  with at least one variable.')
    align = {str(k): ([v] if isinstance(v, str) else list(v)) for k, v in align.items()}
    return env["model"], align


def _short_tb():
    lines = traceback.format_exc().splitlines()
    keep = [l for l in lines if "judge.py" not in l]
    return "\n".join(keep[-12:])


def run_model(model, X, overrides=None):
    """Run the construction on inputs X. overrides: {variable: array [B, k]} replaces those variables' values.
    Returns (output [B, n_out], {variable: value as computed before any override})."""
    values = {}

    def iv(name, value):
        v = np.asarray(value, dtype=np.float64)
        if v.ndim == 1:
            v = v[:, None]
        if v.ndim != 2 or v.shape[0] != len(X):
            raise ConstructionError(f'iv("{name}", value): value must be shaped [B, k] with B = {len(X)}; got {v.shape}.')
        if name in values:
            raise ConstructionError(f'iv("{name}", ...) was called twice; each variable is defined once.')
        values[name] = v
        if overrides and name in overrides:
            v = np.asarray(overrides[name], dtype=np.float64).reshape(v.shape)
        return v[:, 0] if np.ndim(value) == 1 else v.copy()     # same shape as it was given

    try:
        out = np.asarray(model(np.array(X), iv), dtype=np.float64)
    except ConstructionError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ConstructionError("model(x, iv) raised an error:\n" + _short_tb()) from e
    if out.ndim == 1:
        out = out[:, None]
    if out.shape[0] != len(X):
        raise ConstructionError(f"model must return [B, n_out] with B = {len(X)}; got {out.shape}.")
    if not np.isfinite(out).all():
        raise ConstructionError("model returned NaN or infinite values.")
    return out, values


# ---------------------------------------------------------------- comparing outputs
GROUPS = None       # set by judge(): for networks with several decisions per input, the number of classes in each


def _grouped(out):
    return out.reshape(len(out), -1, GROUPS) if GROUPS else out[:, None, :]


def decide(out):
    """The network's decision(s) for each input, [B, n_decisions]: the largest output (within each group of classes),
    or the sign when there is a single output. Outputs are rounded to 1e-9 first, so that an exact tie (for example
    an all-zero ReLU state, common in the edit tests of ReLU RNNs) is decided the same way on both sides instead of by
    floating-point noise."""
    g = np.round(_grouped(out), 9)
    return g.argmax(-1) if g.shape[-1] > 1 else (g[..., 0] > 0).astype(int)


def same(a, b):
    """For each input: do a and b make the same decision (at every position, when there are several)?"""
    return (decide(a) == decide(b)).all(-1)


def match(a, b):
    return float(same(a, b).mean())


def r2(pred, true):
    """Share of variance explained. Multi-class values are compared after removing the mean over classes (logits are
    only defined up to a constant)."""
    if pred.shape == true.shape and _grouped(true).shape[-1] > 1:
        pred, true = _grouped(pred), _grouped(true)
        pred, true = (pred - pred.mean(-1, keepdims=True)).reshape(len(pred), -1), (true - true.mean(-1, keepdims=True)).reshape(len(true), -1)
    sst = ((true - true.mean(0)) ** 2).sum()
    return float(1 - ((pred - true) ** 2).sum() / max(sst, 1e-12))


def kl(net, other):
    """Mean KL(network's distribution || other's) for multi-class outputs; None for a single output."""
    net, other = _grouped(net), _grouped(other)
    if net.shape[-1] == 1:
        return None
    def lsm(z):
        z = z - z.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))
    p, q = lsm(net), lsm(other)
    return float((np.exp(p) * (p - q)).sum(-1).mean())


# ---------------------------------------------------------------- alignment
def fit_alignment(V, A, fit):
    """The affine alignment between variable values V [N, K] and a site's activations A [N, d], fitted on rows `fit`:
    activation ~ c + (V - vmu) @ M (the write map), and V - vmu ~ (activation - c) @ D (the read map, a linear probe)."""
    vmu, c = V[fit].mean(0), A[fit].mean(0)
    M, *_ = np.linalg.lstsq(V[fit] - vmu, A[fit] - c, rcond=1e-8)
    D, *_ = np.linalg.lstsq(A[fit] - c, V[fit] - vmu, rcond=1e-6)
    return c, vmu, M, D


# ---------------------------------------------------------------- whole-model certificate (Glass Networks)
def load_labels(meta):
    """The task's correct answers for the domain inputs, when the target has them (meta["labels"]), else None."""
    f = meta.get("labels")
    return np.load(LAB / "models" / meta["id"] / f) if f else None


def _margin(out):
    """For each input, how clear the decision is: the gap between the top two outputs (the smallest over decisions),
    or the size of the single output."""
    g = _grouped(out)
    if g.shape[-1] == 1:
        return np.abs(g[..., 0]).min(-1)
    top2 = np.sort(g, -1)[..., -2:]
    return (top2[..., 1] - top2[..., 0]).min(-1)


def _row_gap(net, other):
    """Per input: how far the construction's output is from the network's (KL for classes, squared error otherwise)."""
    g, h = _grouped(net), _grouped(other)
    if g.shape[-1] == 1:
        return ((g - h) ** 2).sum((-1, -2))
    def lsm(z):
        z = z - z.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))
    p, q = lsm(g), lsm(h)
    return (np.exp(p) * (p - q)).sum(-1).sum(-1)


def _continuous(X):
    """Real-valued inputs (every dimension takes many values): only then can the disagreement search move them."""
    X = np.asarray(X)
    return np.issubdtype(X.dtype, np.floating) and all(len(np.unique(X[:, j])) > 20 for j in range(X.shape[1]))


def disagreement_search(kind, W, X, model, rng, starts=200, iters=25):
    """A short random local search for inputs where the construction and the network make different decisions while
    the network's own decision is clear (its margin above the 10th percentile over the domain), so that near-ties do
    not count. Inputs stay inside the domain's range. Catches holes that an average over random inputs hides."""
    X = np.asarray(X, dtype=np.float64)
    lo, hi, sd = X.min(0), X.max(0), X.std(0) + 1e-12
    clear = float(np.quantile(_margin(nets.forward(kind, W, X)), 0.1))

    def score(x):
        on = nets.forward(kind, W, x)
        oc, _ = run_model(model, x)
        ok = _margin(on) >= clear
        return np.where(ok, _row_gap(on, oc), -1.0), (~same(oc, on)) & ok, on, oc

    idx = rng.choice(len(X), min(starts, len(X)), replace=False)
    x = X[idx].copy()
    s0, found0, _, _ = score(x)
    s, found = s0.copy(), found0.copy()
    for _ in range(iters):
        xn = np.clip(x + rng.normal(size=x.shape) * 0.15 * sd, lo, hi)
        sn, fn, _, _ = score(xn)
        up = sn > s
        x[up], s[up], found[up] = xn[up], sn[up], fn[up]
    on = nets.forward(kind, W, x)
    oc, _ = run_model(model, x)
    order = np.argsort(-s)
    ex = [{"input": [round(float(v), 3) for v in x[i]], "network": decide(on[i:i + 1])[0].tolist(),
           "construction": decide(oc[i:i + 1])[0].tolist()} for i in order[:5] if found[i]]
    return {"starts": int(len(idx)), "iterations": iters, "clear_disagreements": int(found.sum()),
            "found_rate": float(found.mean()), "random_rate": float(found0.mean()), "examples": ex}


def copy_check(source, W):
    """Is the construction a copy of the network's weights rather than an account of its computation? Counts the numeric
    literals in the source and the share of the network's weights (|w| >= 0.05, not a whole or half number) that some
    literal equals to within 1%.
    Flagged when there are at least half as many literals as parameters AND they reproduce at least half of the
    weights: a re-implementation of the weight matrices, which every test above would pass without explaining
    anything (found on AlgZoo: copying M2,2's 10 weights passes every test)."""
    import io, tokenize
    lits = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.NUMBER:
                try:
                    lits.append(abs(float(tok.string.replace("_", "").rstrip("jJ"))))
                except ValueError:
                    pass
    except (tokenize.TokenError, IndentationError):
        pass
    ws = np.concatenate([np.abs(np.ravel(v)) for v in W.values() if np.size(v) > 1]) if W else np.zeros(0)
    big = ws[(ws >= 0.05) & (np.abs(2 * ws - np.round(2 * ws)) > 1e-6)]   # whole and half numbers (hand-set weights such as
                                                                           # 1, 2, 0.5) are what an account would write anyway
    lv = np.array(sorted(set(lits))) if lits else np.zeros(0)
    if len(big) and len(lv):
        j = np.clip(np.searchsorted(lv, big), 1, len(lv) - 1) if len(lv) > 1 else np.zeros(len(big), int)
        near = np.minimum(np.abs(lv[j] - big), np.abs(lv[np.maximum(j - 1, 0)] - big))
        matched = float((near <= 0.01 * np.maximum(big, 1e-9)).mean())
    else:
        matched = 0.0
    params = int(ws.size)
    # A copy can also be rescaled (each unit measured in units of its own gain), which the value match misses. So
    # also: the largest single array literal (a list of numbers, nested or not) against the network's largest weight
    # tensor. A table with (nearly) as many entries as that tensor is the tensor, rescaled (found on AlgZoo M4,4: a 4x4
    # "recurrent gain matrix" equal to W_h up to per-unit scales). The bar is 90%: an account that found structure
    # needs fewer numbers (M4,3's reader used the cell's mirror symmetry to write W_h's 16 entries as 8), and a bar at
    # 50% rejected that. This is a heuristic: a determined copier could split the table; the reader is not adversarial.
    import ast as _ast
    def nums(node):
        return sum(1 for n in _ast.walk(node) if isinstance(n, _ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool))
    try:
        tables = [nums(n) for n in _ast.walk(_ast.parse(source)) if isinstance(n, (_ast.List, _ast.Tuple))]
    except SyntaxError:
        tables = []
    biggest_table = max(tables, default=0)
    biggest_tensor = max((int(np.size(v)) for v in W.values()), default=0) if W else 0
    table_copy = bool(biggest_tensor >= 8 and biggest_table >= 0.9 * biggest_tensor)
    value_copy = bool(params >= 8 and len(lits) >= 0.5 * params and matched >= 0.5)
    # Third rule, invariant to rescaling AND to splitting tables: compression. Constants = the file's non-integer
    # numeric literals other than 0.5. A construction with about as many constants as the network has parameters
    # compresses nothing; it re-describes the weights (found on AlgZoo M8,8: 136 constants for 136 parameters, quoted
    # in a per-unit scale and split into 8-number rows). Calibrated on this session's constructions: honest accounts
    # used 0.56 (M4,3) and 0.80 (M2,2) constants per parameter; the re-descriptions used exactly 1.00 (M8,8, M16,10).
    consts = sum(1 for v in lits if v != int(v) and v != 0.5)
    compression_copy = bool(params >= 8 and consts >= 0.9 * params)
    return {"literals": len(lits), "params": params, "weights_matched": matched, "largest_table": biggest_table,
            "largest_tensor": biggest_tensor, "constants": consts, "constants_per_param": round(consts / max(params, 1), 3),
            "value_copy": value_copy, "table_copy": table_copy, "compression_copy": compression_copy,
            "flagged": value_copy or table_copy or compression_copy}


def certificate(rep, kind, W, X, test, out_n, out_c, hats, labels, model, rng):
    """The construction-level tests added for whole-model claims (plan items 2-3). Writes rep["joint"],
    rep["worst"], rep["errors"] and rep["search"]."""
    clean, cons = out_n[test], out_c[test]
    agree = same(cons, clean)
    if hats:                                                    # 2. every named site replaced at once
        o = nets.forward(kind, W, X[test], hook=dict(hats))
        rep["joint"] = {"sites": list(hats), "match": match(o, clean), "kl": kl(clean, o)}
    dec = decide(clean)                                         # 3a. behaviour per slice
    slices = []
    if dec.shape[1] == 1:
        for c in np.unique(dec[:, 0]):
            m = dec[:, 0] == c
            slices.append({"slice": f"network decides {int(c)}", "n": int(m.sum()), "match": float(agree[m].mean())})
    marg = _margin(clean)
    lo = marg <= np.quantile(marg, 0.1)
    ranked = [x for x in slices if x["n"] >= MIN_SLICE]
    rep["worst"] = {"slices": slices, "worst": min(ranked, key=lambda x: x["match"]) if ranked else None,
                    "near_ties": {"slice": "the network's 10% least clear decisions", "n": int(lo.sum()), "match": float(agree[lo].mean())}}
    if labels is not None:                                      # 3b. the network's own mistakes
        lab = np.asarray(labels).reshape(len(labels), -1)[test]
        wrong = ~(dec == lab).all(-1)
        nw = int(wrong.sum())
        rep["errors"] = {"network_error_rate": float(wrong.mean()), "n_errors": nw,
                         "reproduced": float(agree[wrong].mean()) if nw else None,
                         "construction_right_where_network_wrong": float((decide(cons)[wrong] == lab[wrong]).all(-1).mean()) if nw else None}
    if _continuous(X):                                          # 3c. search for disagreements
        rep["search"] = disagreement_search(kind, W, X, model, rng)


def coverage(rep):
    """Plan item 4: one line of what the construction accounts for, and the plain list of what it does not."""
    v, b = rep["verdict"], rep["behaviour"]
    un = [f"stage '{name}' is not explained" for name, st in rep["stages"].items() if not st["verified"]]
    if b["match"] < 1:
        un.append(f"on {1 - b['match']:.1%} of inputs the construction's decision differs from the network's")
    j = rep.get("joint")
    if j and j["match"] < 1:
        un.append(f"replacing every named site at once loses {1 - j['match']:.1%} of the network's decisions")
    if rep.get("unaligned_variables"):
        un.append("variables not placed in the network (untested): " + ", ".join(rep["unaligned_variables"]))
    w = (rep.get("worst") or {}).get("worst")
    if w and w["match"] < 1:
        un.append(f"worst slice: {w['slice']} ({w['n']} inputs), where it matches {w['match']:.1%}")
    e = rep.get("errors")
    if e and e["n_errors"] and e["reproduced"] < 1:
        un.append(f"of the network's {e['n_errors']} mistakes on test inputs, {1 - e['reproduced']:.1%} are not reproduced")
    cp = rep.get("copy")
    if cp and cp["flagged"]:
        un.append("the construction copies the network's weights, so it explains nothing by itself")
    sr = rep.get("search")
    if sr and sr["clear_disagreements"]:
        un.append(f"a search found {sr['clear_disagreements']} inputs (from {sr['starts']} starts) where the network's decision is clear "
                  f"and the construction's differs")
    return {"stages_verified": v["stages_verified"], "stages_total": v["stages_total"], "behaviour": b["match"],
            "joint": j["match"] if j else None, "worst_slice": w["match"] if w else None,
            "errors_reproduced": e["reproduced"] if e else None, "unexplained": un}


# ---------------------------------------------------------------- the tests
def judge(kind, W, X, source, seed=0, n_pairs=2000, groups=None, labels=None):
    """Returns a report dict. Raises ConstructionError with a message the author can act on.
    groups: for networks that make several decisions per input (one per position), the number of classes in each.
    labels: the task's correct answers for X (optional): enables the test of the network's own mistakes."""
    global GROUPS
    GROUPS = groups
    t0 = time.time()
    rng = np.random.default_rng(seed)
    model, align = load_construction(source)
    table = nets.site_table(kind, W)
    sites = {name: width for name, width, _, _ in table}
    stage_of = {name: stage for name, _, _, stage in table}
    for var, where in align.items():
        for s in where:
            if s not in sites:
                raise ConstructionError(f'ALIGN["{var}"] names site "{s}", which this network does not have. Sites: {", ".join(sites)}.')
            if s == "out":
                raise ConstructionError('ALIGN cannot use the site "out": the output is what model() returns. Align variables to internal sites.')
    X = np.asarray(X)
    N = len(X)
    acts = {}
    out_n = nets.forward(kind, W, X, acts=acts)
    out_c, vals = run_model(model, X)
    if out_c.shape[1] != out_n.shape[1]:
        raise ConstructionError(f"model returns {out_c.shape[1]} outputs per input; the network has {out_n.shape[1]}.")
    # a site that is the same for every input holds no variable: alignments to it are dropped (and reported)
    constant = {s for s in sites if acts[s].std(0).max() <= 1e-9}
    dropped = sorted({s for where in align.values() for s in where if s in constant})
    align = {v: [s for s in where if s not in constant] for v, where in align.items()}
    align = {v: where for v, where in align.items() if where}
    for var in align:
        if var not in vals:
            raise ConstructionError(f'ALIGN names "{var}", but model never calls iv("{var}", ...). Variables it defines: {", ".join(vals) or "none"}.')
    if N >= 200:
        perm = rng.permutation(N)
        fit, test = perm[: N // 2], perm[N // 2:]
    else:
        fit = test = np.arange(N)

    rep = {"n_inputs": N, "n_fit": len(fit), "n_test": len(test), "source_chars": len(source),
           "variables": {v: {"dim": int(vals[v].shape[1]), "sites": align.get(v, [])} for v in vals},
           "unaligned_variables": [v for v in vals if v not in align], "constant_sites_ignored": dropped}

    # 1. behaviour
    const = np.broadcast_to(out_n[fit].mean(0), out_n[test].shape)
    rep["behaviour"] = {"match": match(out_c[test], out_n[test]), "r2": r2(out_c[test], out_n[test]), "kl": kl(out_n[test], out_c[test]),
                        "baseline_constant_match": match(const, out_n[test])}

    # 2-4. per site
    rep["sites"], rep["interventions"] = {}, []
    hats = {}                                       # each named site's predicted activation on the test inputs (joint test)
    base = rng.choice(test, n_pairs); src = rng.choice(test, n_pairs)
    out_c_base = out_c[base]
    for s in [s for s in sites if any(s in w for w in align.values())]:
        names = [v for v in vals if s in align.get(v, [])]
        cols, k0 = {}, 0
        for v in names:
            cols[v] = np.arange(k0, k0 + vals[v].shape[1]); k0 += vals[v].shape[1]
        V = np.concatenate([vals[v] for v in names], 1)
        A = acts[s]
        c, vmu, M, D = fit_alignment(V, A, fit)                                 # M [K, d] writes, D [d, K] reads
        A_hat = c + (V - vmu) @ M
        hats[s] = A_hat[test]
        Mp = np.linalg.pinv(M, rcond=1e-8)                                      # [d, K]: activation -> variable coordinates
        rank = int(np.linalg.matrix_rank(M, tol=1e-8 * max(1.0, np.abs(M).max())))
        P = Mp @ M                                                              # projector onto the named subspace
        def run(hook_value, idx=test):
            return nets.forward(kind, W, X[idx], hook={s: hook_value})
        def rand_proj(r):
            Q, _ = np.linalg.qr(rng.normal(size=(A.shape[1], r)))
            return Q @ Q.T
        clean = out_n[test]
        o_splice = run(A_hat[test])
        o_mean = run(np.broadcast_to(c, A[test].shape))
        o_keep = run(c + (A[test] - c) @ P)
        o_excl = run(A[test] - (A[test] - c) @ P)
        excl_rand = float(np.mean([match(run(A[test] - (A[test] - c) @ rand_proj(rank)), clean) for _ in range(3)])) if 0 < rank < A.shape[1] else None
        m_splice, m_mean = match(o_splice, clean), match(o_mean, clean)
        # the smallest number of principal components of this site that keeps 90% of decisions (binary search)
        _, sv, Vt = np.linalg.svd(A[fit] - c, full_matrices=False)
        full = int((sv > 1e-9 * max(sv[0], 1e-12)).sum())
        lo, hi = 1, max(full, 1)
        while lo < hi:
            mid_r = (lo + hi) // 2
            if match(run(c + (A[test] - c) @ Vt[:mid_r].T @ Vt[:mid_r]), clean) >= 0.9:
                hi = mid_r
            else:
                lo = mid_r + 1
        pca_rank = lo
        rep["sites"][s] = {
            "width": int(A.shape[1]), "variables": names, "named_dims": int(V.shape[1]), "named_rank": rank,
            "pca_sufficient_rank": int(pca_rank), "activation_rank": full, "_M": M, "_cols": cols,
            "explained": float(1 - ((A_hat[test] - A[test]) ** 2).sum() / max(((A[test] - c) ** 2).sum(), 1e-12)),
            "substitute_match": m_splice, "substitute_kl": kl(clean, o_splice), "substitute_r2": r2(o_splice, clean),
            "mean_ablate_match": m_mean,
            "recovered": float((m_splice - m_mean) / max(1 - m_mean, 1e-9)),
            "keep_named_match": match(o_keep, clean), "remove_named_match": match(o_excl, clean), "remove_random_match": excl_rand,
        }
        # interventions on each variable at this site
        # interchange reads each variable from the real activations with the linear probe D
        dA = (A[src] - A[base]) @ D                                             # the change in every variable, as read from the two runs
        # Three kinds of intervention test at a site:
        #   interchange  everything named at the site is copied from the source run (the standard interchange test);
        #   partial      one variable is copied while the site's other variables stay (only where there are several):
        #                a state that no input produces;
        #   edit         for a variable alone at its site: a random half of its coordinates are set to the source's
        #                values, also a state that no input produces.
        tests = [("interchange", names, None)]
        if len(names) > 1:
            tests += [("partial", [v], None) for v in names]
        if V.shape[1] >= 2:
            tests.append(("edit", names, "half"))
        allJ = np.arange(V.shape[1])
        for mode, vs, how in tests:
            J = np.concatenate([cols[v] for v in vs])
            mask = np.ones((n_pairs, len(J)))
            if how == "half":
                mask = (rng.random((n_pairs, len(J))) < 0.5).astype(float)
                dead = mask.sum(1) == 0
                mask[dead, rng.integers(0, len(J), dead.sum())] = 1
                full_ = mask.sum(1) == len(J)
                mask[full_, rng.integers(0, len(J), full_.sum())] = 0
            new_all = V[base].copy()
            new_all[:, J] = np.where(mask > 0, V[src][:, J], V[base][:, J])
            o_pred, _ = run_model(model, X[base], {v: new_all[:, cols[v]] for v in vs})
            if mode == "edit":             # write the construction's own new value along the variable's fitted directions
                delta = (new_all[:, J] - V[base][:, J]) @ M[J]
            else:                          # copy the variables' coordinates as read from the source run's activation
                delta = dA[:, J] @ M[J]
            o_net = nets.forward(kind, W, X[base], hook={s: A[base] + delta})
            # attribution patching's answer for the same change: the first-order estimate, computed on the network
            o_lin = out_n[base] + (nets.forward(kind, W, X[base], hook={s: A[base] + 1e-3 * delta}) - out_n[base]) / 1e-3
            r = max(1, int(np.linalg.matrix_rank(M[J], tol=1e-8 * max(1.0, np.abs(M).max()))))
            Pr = rand_proj(min(r, A.shape[1]))
            o_rand = nets.forward(kind, W, X[base], hook={s: A[base] + (A[src] - A[base]) @ Pr})
            net_changed = ~same(o_net, out_n[base])
            agree = same(o_pred, o_net)
            rep["interventions"].append({
                "variable": vs[0] if len(vs) == 1 else "all variables", "variables": vs, "site": s, "mode": mode, "dim": int(len(J)),
                "agreement": float(agree.mean()),
                "r2": r2(o_pred, o_net),
                "network_changed": float(net_changed.mean()),
                "agreement_when_changed": float(agree[net_changed].mean()) if net_changed.any() else None,
                "baseline_no_change": match(out_n[base], o_net),
                "baseline_source_output": match(out_n[src], o_net),
                "control_random_subspace": match(o_pred, o_rand),
                "attribution_patching": float(same(o_lin, o_net).mean()),
                "attribution_patching_when_changed": float(same(o_lin, o_net)[net_changed].mean()) if net_changed.any() else None,
            })
    for s, r in rep["sites"].items():
        # variables whose interventions never change the output are present but unused: they do not count toward size
        M, cols = r.pop("_M"), r.pop("_cols")
        live = [v for v in r["variables"] if any(t["site"] == s and t["variables"] == [v] and t["network_changed"] >= 0.05 for t in rep["interventions"])]
        J = np.concatenate([cols[v] for v in live]) if live else np.array([], int)
        r["used_rank"] = int(np.linalg.matrix_rank(M[J], tol=1e-8 * max(1.0, np.abs(M).max()))) if len(J) else 0
        r["unused_variables"] = [v for v in r["variables"] if v not in live]
    certificate(rep, kind, W, X, test, out_n, out_c, hats, labels, model, rng)
    rep["copy"] = copy_check(source, W)
    varying = {s for s in sites if acts[s].std(0).max() > 1e-9}                 # a site that never changes needs no explanation
    rep["uncovered_sites"] = [s for s in sites if s != "out" and s in varying and s not in rep["sites"]]
    rep["stages"] = {}
    for s in sites:
        if s != "out" and s in varying:
            rep["stages"].setdefault(stage_of[s], {"sites": [], "named": [], "verified": []})["sites"].append(s)
    rep["verdict"] = verdict(rep)
    for s, r in rep["sites"].items():
        st = rep["stages"][stage_of[s]]
        st["named"].append(s)
        if r["pass"]:
            st["verified"].append(s)
    v = rep["verdict"]
    v["stages_total"] = len(rep["stages"])
    v["stages_verified"] = sum(bool(st["verified"]) for st in rep["stages"].values())
    v["complete"] = bool(v["verified"] and v["stages_verified"] == v["stages_total"])
    v["score"] = round(rep["behaviour"]["match"] * v["stages_verified"] / max(1, v["stages_total"]), 4)
    rep["coverage"] = coverage(rep)
    rep["seconds"] = round(time.time() - t0, 2)
    return rep


def verdict(rep):
    """A site is VERIFIED when
      - it can be replaced by the construction's prediction (substitution);
      - removing what the construction names there does at least half of the damage of mean-ablating the whole site;
      - the named subspace is compact: at most twice (or 4 more than) the number of principal components of the site
        that suffice to keep 90% of decisions (variables that the construction correctly marks as unused, i.e. whose
        interventions change nothing, are not counted);
      - its interchange predictions hold (everything named at the site copied from another input: 90%), and so do
        its predictions for states that no input produces: EITHER each variable copied while the others stay
        (partial) OR a random half of the coordinates of everything named at the site copied (edit): 50% of the
        cases where the network's decision changed (where "no change" scores 0 and a random class about 1/classes).
    A site with no effect on the output cannot be verified. A construction is VERIFIED when it behaves
    like the network and every site it names is verified, and COMPLETE when, in addition, every stage of the
    network has a verified site. The thresholds are PASS (top of this file)."""
    fails = []
    b = rep["behaviour"]
    if b["match"] < PASS["behaviour"]:
        fails.append(f"behaviour: the construction's output agrees with the network on {b['match']:.1%} of inputs (need {PASS['behaviour']:.0%})")
    for s, r in rep["sites"].items():
        mine = []
        if r["substitute_match"] < PASS["substitution"]:
            mine.append(f"substitution at {s}: {r['substitute_match']:.1%} (need {PASS['substitution']:.0%})")
        effect = 1 - r["mean_ablate_match"]          # how much of the decision depends on this site at all
        limit = max(PASS["compact"] * r["pca_sufficient_rank"], r["pca_sufficient_rank"] + 4)
        if effect < 0.05:
            mine.append(f"{s} has almost no effect on the output (replacing the whole site by its mean changes {effect:.1%} of decisions), "
                        "so nothing can be verified here; use another site of this stage")
        else:
            if 1 - r["remove_named_match"] < PASS["necessity"] * effect:
                mine.append(f"removal at {s}: removing the named subspace leaves {r['remove_named_match']:.1%} of decisions, while mean-ablating "
                            f"the whole site leaves {r['mean_ablate_match']:.1%}: what you named is not what the output depends on here")
            if r["used_rank"] > limit:
                mine.append(f"size at {s}: you named {r['used_rank']} dimensions, but the {r['pca_sufficient_rank']} largest principal components "
                            f"of this site already keep 90% of decisions (limit {limit}): the variables must be more compact")
        off = []                    # failures of the tests that make states no input produces (partial, edit)
        for t in rep["interventions"]:
            if t["site"] != s:
                continue
            score = t["agreement_when_changed"] if t["agreement_when_changed"] is not None else t["agreement"]
            if t["network_changed"] < 0.05 and t["agreement"] >= 0.95:
                t["pass"] = True        # the construction correctly predicts that this changes nothing at this site
            elif t["network_changed"] < 0.05:
                t["pass"] = False
                msg = (f"{t['mode']} of {t['variable']} at {s}: the network's output changed in only {t['network_changed']:.1%} of cases, but "
                       "the construction predicted changes: at this site the network's output does not depend on this variable")
                (off if t["mode"] == "edit" else mine).append(msg)
            elif t["mode"] in ("edit", "partial"):
                t["pass"] = bool(score >= PASS["edit"])
                if not t["pass"]:
                    off.append(f"{t['mode']} of {t['variable']} at {s}: prediction right in {score:.1%} of the cases where the network's output "
                               f"changed (need {PASS['edit']:.0%}; guessing 'no change' gets 0% of these)")
            else:
                t["pass"] = bool(score >= PASS["interchange"])
                if not t["pass"]:
                    mine.append(f"interchange of {t['variable']} at {s}: prediction right in {score:.1%} of the cases where the network's "
                                f"output changed (need {PASS['interchange']:.0%})")
        # The off-manifold requirement can be met in either of two ways: every single-variable swap (partial) passes,
        # or the coordinate edit of everything named at the site passes. So splitting a variable into parts never
        # makes a construction score worse than the same content kept as one variable.
        here = [t for t in rep["interventions"] if t["site"] == s]
        partial_ok = all(t["pass"] for t in here if t["mode"] == "partial")
        edit_ok = all(t["pass"] for t in here if t["mode"] == "edit")
        has_partial, has_edit = any(t["mode"] == "partial" for t in here), any(t["mode"] == "edit" for t in here)
        r["off_manifold"] = "partial" if has_partial and partial_ok else ("edit" if has_edit and edit_ok else ("none needed" if not (has_partial or has_edit) else "fails"))
        if r["off_manifold"] == "fails":
            mine += off
            if has_partial and has_edit:
                mine.append(f"(at {s} it is enough that EITHER every partial test OR the edit of all variables passes)")
        r["pass"] = not mine
        fails += mine
    # construction-level claims (Glass Networks): they apply to whatever the construction names
    j = rep.get("joint")
    if j and len(j["sites"]) > 1 and j["match"] < PASS["joint"]:
        fails.append(f"joint substitution: with all {len(j['sites'])} named sites replaced at once by the construction's predictions, the "
                     f"network keeps {j['match']:.1%} of its decisions (need {PASS['joint']:.0%}): the sites pass one at a time but do not work together")
    w = (rep.get("worst") or {}).get("worst")
    if w and w["match"] < PASS["worst_slice"]:
        fails.append(f"worst slice: on the {w['n']} inputs where {w['slice']}, the construction agrees on {w['match']:.1%} (need {PASS['worst_slice']:.0%})")
    e = rep.get("errors")
    if e and e["n_errors"] >= MIN_SLICE and e["reproduced"] < PASS["errors"]:
        fails.append(f"the network's own mistakes: on the {e['n_errors']} test inputs where the network is wrong, the construction makes the "
                     f"network's (wrong) decision on {e['reproduced']:.1%} (need {PASS['errors']:.0%}); it is right on "
                     f"{e['construction_right_where_network_wrong']:.1%}: it explains the task, not this network")
    cp = rep.get("copy")
    if cp and cp["flagged"]:
        if cp.get("value_copy", cp["flagged"]):
            why = (f"the file has {cp['literals']} numeric literals for a network of {cp['params']} parameters, and they reproduce "
                   f"{cp['weights_matched']:.0%} of its weights")
        elif cp.get("table_copy"):
            why = (f"the file writes out a table of {cp['largest_table']} numbers, about the size of the network's largest weight "
                   f"tensor ({cp['largest_tensor']} entries), which is that tensor up to rescaling")
        else:
            why = (f"the file has {cp['constants']} constants for a network of {cp['params']} parameters ({cp['constants_per_param']:.2f} per "
                   f"parameter): it compresses nothing, so it re-describes the weights (rescaled or regrouped) instead of finding the rule")
        fails.append(f"weight copy: {why}: this re-implements the weight matrices instead of describing the computation. Write the "
                     "mechanism in terms of what the variables mean (comparisons, running maxima, counts, thresholds), with only the "
                     "constants that account needs")
    if not rep["sites"]:
        fails.append("no sites aligned")
    return {"verified": not fails, "fails": fails}


# ---------------------------------------------------------------- text report (what the constructor's reader sees)
def render(rep):
    L = []
    b = rep["behaviour"]
    L.append(f"JUDGE REPORT   ({rep['n_test']} held-out inputs; construction {rep['source_chars']:,} chars)")
    L.append(f"1. BEHAVIOUR  construction vs network output: same decision on {b['match']:.1%} of inputs "
             f"(a constant output would get {b['baseline_constant_match']:.1%}); output R2 {b['r2']:.3f}"
             + (f"; KL {b['kl']:.3f}" if b["kl"] is not None else ""))
    L.append("2-3. SITES   explained = share of the activation's variance the variables explain;  substitute = network decision kept "
             "when the activation is replaced by the construction's prediction;  keep = when only the named subspace of the real "
             "activation is kept;  remove = when the named subspace is removed (should be LOW) vs a random subspace of equal size;  "
             "mean-ablate = when the whole site is replaced by its mean")
    for s, r in rep["sites"].items():
        rr = "n/a" if r["remove_random_match"] is None else f"{r['remove_random_match']:.1%}"
        L.append(f"  {s:11s} ({r['named_rank']} dims named of {r['width']}; {r['pca_sufficient_rank']} principal components suffice; "
                 f"{', '.join(r['variables'])}): explained {r['explained']:.3f} | "
                 f"substitute {r['substitute_match']:.1%} | keep {r['keep_named_match']:.1%} | remove {r['remove_named_match']:.1%} "
                 f"(random {rr}) | mean-ablate {r['mean_ablate_match']:.1%}")
    L.append("4. INTERVENTIONS   interchange = everything named at the site is copied from another input's run (need 90%). "
             "Then tests that make states no input produces (need 50% of the cases where the decision changed): partial = one "
             "variable is copied while the site's other variables stay; edit = a random half of the coordinates of everything "
             "named at the site are copied. Where both are run, it is enough that all partial tests pass OR the edit passes. "
             "right = the construction predicted the network's new decision (all cases / cases where the decision changed).  "
             "controls: the same swap in a random subspace; guessing 'no change'; guessing the source input's output; "
             "attribution patching (a first-order estimate computed on the network itself)")
    for t in rep["interventions"]:
        w = "n/a" if t["agreement_when_changed"] is None else f"{t['agreement_when_changed']:.1%}"
        L.append(f"  {t['mode']:11s} {t['variable']} @ {t['site']}: right {t['agreement']:.1%} / {w} when changed (network changed in "
                 f"{t['network_changed']:.1%}); output R2 {t['r2']:.3f} | controls: random subspace {t['control_random_subspace']:.1%}, "
                 f"no-change {t['baseline_no_change']:.1%}, source-output {t['baseline_source_output']:.1%}, "
                 f"attribution patching {t['attribution_patching']:.1%}")
    if rep.get("constant_sites_ignored"):
        L.append("Ignored (the site is identical for every input, so it holds no variable): " + ", ".join(rep["constant_sites_ignored"]))
    if rep["unaligned_variables"]:
        L.append("Variables in model() with no site in ALIGN (not tested): " + ", ".join(rep["unaligned_variables"]))
    j = rep.get("joint")
    if j:
        L.append(f"5. WHOLE MODEL   all {len(j['sites'])} named site(s) replaced at once by the construction's predictions: the network keeps "
                 f"{j['match']:.1%} of its decisions (need {PASS['joint']:.0%} when two or more sites are named)")
    w = rep.get("worst")
    if w:
        parts = [f"{x['slice']}: {x['match']:.1%} (n={x['n']})" for x in w["slices"]]
        L.append("6. WORST CASE   behaviour by the network's decision: " + ("; ".join(parts) if parts else "n/a")
                 + f" | near ties ({w['near_ties']['n']} least clear inputs, no requirement): {w['near_ties']['match']:.1%}")
        e = rep.get("errors")
        if e:
            L.append(f"   the network's own mistakes: it is wrong on {e['network_error_rate']:.1%} of test inputs ({e['n_errors']}); the construction "
                     + (f"reproduces {e['reproduced']:.1%} of them (need {PASS['errors']:.0%} when there are {MIN_SLICE} or more) and is right on "
                        f"{e['construction_right_where_network_wrong']:.1%}" if e["n_errors"] else "has nothing to reproduce"))
        sr = rep.get("search")
        if sr:
            L.append(f"   search for disagreements ({sr['starts']} starts x {sr['iterations']} steps, only inputs where the network's decision is clear): "
                     f"found {sr['clear_disagreements']} ({sr['found_rate']:.1%}; on the unsearched starting inputs {sr['random_rate']:.1%})"
                     + ("".join(f"\n     input {x['input']}: network {x['network']}, construction {x['construction']}" for x in sr["examples"])))
    cp = rep.get("copy")
    if cp:
        L.append(f"7. NOT A COPY   {cp['literals']} numeric literals in the file; network parameters {cp['params']}; share of its weights "
                 f"(|w| >= 0.05) the literals reproduce: {cp['weights_matched']:.0%}; largest table in the file {cp.get('largest_table', '?')} "
                 f"numbers vs the largest weight tensor {cp.get('largest_tensor', '?')}; constants per network parameter {cp.get('constants_per_param', '?')}"
                 + ("  -> FLAGGED as a weight copy" if cp["flagged"] else ""))
    L.append("STAGES   " + ";  ".join(
        f"{name}: " + ("VERIFIED at " + ", ".join(st["verified"]) if st["verified"] else
                       ("named but failing" if st["named"] else "NOT EXPLAINED (no variable at " + " / ".join(st["sites"]) + ")"))
        for name, st in rep["stages"].items()))
    v = rep["verdict"]
    if v["complete"]:
        L.append("VERDICT: VERIFIED AND COMPLETE: every stage of the network has a verified site.")
    elif v["verified"]:
        L.append(f"VERDICT: VERIFIED at every site it names, but INCOMPLETE: {v['stages_verified']} of {v['stages_total']} stages explained.")
    else:
        L.append(f"VERDICT: NOT VERIFIED ({v['stages_verified']} of {v['stages_total']} stages verified). Failing:")
    L += [f"  - {f}" for f in v["fails"]]
    c = rep.get("coverage")
    if c:
        L.append(f"COVERAGE   stages {c['stages_verified']}/{c['stages_total']} | behaviour {c['behaviour']:.1%}"
                 + (f" | all sites at once {c['joint']:.1%}" if c["joint"] is not None else "")
                 + (f" | worst slice {c['worst_slice']:.1%}" if c["worst_slice"] is not None else "")
                 + (f" | network's mistakes reproduced {c['errors_reproduced']:.1%}" if c["errors_reproduced"] is not None else ""))
        L.append("NOT EXPLAINED:" + (" nothing found by these tests" if not c["unexplained"] else ""))
        L += [f"  - {u}" for u in c["unexplained"]]
    return "\n".join(L)


def load_target(tid, variant="original", seed=0):
    meta = json.loads((LAB / "models" / tid / "meta.json").read_text())
    W = dict(np.load(LAB / "models" / tid / "weights.npz"))
    W = {k: v.astype(np.float64) for k, v in nets.variant(W, variant, meta.get("perm_groups", []), seed, meta.get("protected", [])).items()}
    return meta, W, nets.domain_inputs(meta)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("target"); ap.add_argument("construction")
    ap.add_argument("--variant", default="original"); ap.add_argument("--json"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if json.loads((LAB / "models" / a.target / "meta.json").read_text()).get("forward") == "lm":     # a language-model target
        import lmjudge
        try:
            rep = lmjudge.judge(a.target, Path(a.construction).read_text(), a.variant, a.seed)
        except Exception as e:                      # lmjudge raises the ConstructionError of the imported judge module
            if type(e).__name__ != "ConstructionError":
                raise
            print("CONSTRUCTION ERROR:", e); sys.exit(1)
        print(lmjudge.render(rep))
        if a.json:
            Path(a.json).write_text(json.dumps(rep, indent=1))
        sys.exit(0)
    meta, W, X = load_target(a.target, a.variant)
    try:
        rep = judge(meta["forward"], W, X, Path(a.construction).read_text(), seed=a.seed, groups=meta.get("out_groups"), labels=load_labels(meta))
    except ConstructionError as e:
        print("CONSTRUCTION ERROR:", e); sys.exit(1)
    print(render(rep))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
