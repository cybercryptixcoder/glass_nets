"""The sealed evaluator for evolving the box (glass_nets `evolve/`). The loop's agents never see this file; they see
only the problem statement (`helix_problem.py`) and the reports this file writes back.

What it measures: how much of each network's computation a box's blueprint explains, as certified by the frozen
checker (box/verify_vec.py, v1) on inputs the box never saw, with lines from controls (random directions and
weight-shuffled copies of the network), and how readable the explanation is.

For each network the box is run, alone, in its own process (`explain(net, budget_seconds)`, CPU and memory limits),
and its blueprint is checked:
- form: the checker's own rules (vocabulary definitions within L = 7 and the constant budget; every computed part
  located in the network as a unit direction with a scale; the output computed from located parts only);
- run alone: on 3,000 held-out inputs, the share on which the blueprint's decision equals the network's (a);
- interventions: each located rule part is swapped between pairs of held-out inputs inside the network; its IIA
  (the share of the pairs where the blueprint predicts a change on which the network's decision changes exactly as
  predicted) against the family-wise control line. A part's credit is its margin over the line, (IIA - line) /
  (1 - line), clipped to [0, 1]; zero if it has fewer than 20 decisive pairs; a quarter if its definition (or an
  invented primitive it uses) carries a constant that is not an integer or a half in [-10, 10] (a fitted number is a
  weight restated, not an explanation).
Per network:
- ARC's AlgZoo networks (the target): a * q + 0.02 * a (at most 1), q = the sum of the located parts' credits over
  max(number of located rule parts, 2) (a part that fails dilutes q; full marks need at least two parts). The product
  means behaviour counts only as far as the parts that carry it are certified: the answer itself claimed as one weakly
  certified variable scores under 0.1 (test_black_box.py; a first version, 0.5 * a + 0.5 * q, gave it 0.53).
- the planted network (a hand-built network with a known variable): half its AlgZoo-style score (with the number of
  planted variables in place of 2), half the share of its planted variables that a credited part reproduces
  (|correlation| >= 0.99 on held-out inputs).
- the structureless network (random weights): a story there is wrong. 1, less the sum of the credits of its rule
  parts (at least 0).
The training score is the mean over the training networks. Held-out networks (other sizes, ARC's 16-unit challenge
network, a second planted network, a shuffled network) are scored the same way, only for programs that set a new
best, and are never reported to the agents.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from box.adapters.algzoo_rnn import AlgZooRNN  # noqa: E402
from box.blueprint import Blueprint, Part  # noqa: E402
from box.verify_vec import VecChecker  # noqa: E402
from bench.accept import vocab  # noqa: E402

SECRET_SEED = 7_310_442          # the held-out inputs' seed (the box never sees these inputs)
N_HELD, N_PAIRS, CONTROL_REPS = 3000, 400, 60
MIN_DECISIVE = 20

TRAIN = ["zoo_2nd_argmax_2_2", "zoo_2nd_argmax_4_3", "zoo_argmedian_4_3", "zoo_2nd_argmax_4_4", "planted_sum_4_4",
         "random_4_3"]
HELDOUT = ["zoo_2nd_argmax_8_5", "zoo_argmedian_8_5", "zoo_2nd_argmax_16_10", "planted_sum_6_5", "shuffled_4_3"]
TASKS = {"2nd_argmax": "outputs the position (0-based) of the second-largest number of the sequence",
         "argmedian": "outputs the position (0-based) of the median number of the sequence"}
ALLOWED_IMPORTS = {"numpy", "scipy", "scipy.optimize", "scipy.linalg", "scipy.stats", "scipy.special", "math",
                   "itertools", "functools", "collections", "heapq", "random", "time", "boxkit", "typing",
                   "dataclasses", "statistics", "bisect", "copy", "re", "operator"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals", "locals",
                   "vars", "memoryview", "exit", "quit"}
# numpy's file input and output (os, shutil, pathlib and the like cannot be imported at all); list.remove and other
# ordinary methods are fine (a first version also banned .remove, which rejected a legitimate program, Nov 28)
FORBIDDEN_ATTRS = {"load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt", "fromfile",
                   "tofile", "memmap", "DataSource", "_V", "_S", "_ROOT", "__globals__", "__builtins__",
                   "__subclasses__", "__code__"}


# ---------------------------------------------------------------- the networks
def planted_sum(hidden, T, seed):
    """Units 0 and 1 carry the running sum as p - n (p = relu(s), n = relu(-s)); the other units are distractors
    with random weights that the output ignores. Decision: 0 if the sum is positive, 1 otherwise."""
    rng = np.random.default_rng(seed)
    Wx = np.zeros((1, hidden)); Wh = np.zeros((hidden, hidden)); Wo = np.zeros((hidden, 2))
    Wx[0, :2] = [1.0, -1.0]
    Wh[:2, :2] = [[1.0, -1.0], [-1.0, 1.0]]
    Wx[0, 2:] = rng.normal(size=hidden - 2)
    Wh[2:, 2:] = rng.normal(scale=0.5, size=(hidden - 2, hidden - 2))
    Wh[:2, 2:] = rng.normal(scale=0.5, size=(2, hidden - 2))      # distractors read the sum; nothing reads them
    Wo[:2] = [[1.0, -1.0], [-1.0, 1.0]]
    return {"W_x": Wx, "W_h": Wh, "W_o": Wo}, {"sum": "(sum x)"}


def load(name):
    """(checker network, weights, T, task, kind, truth)."""
    if name.startswith("zoo_"):
        W = dict(np.load(os.path.join(ROOT, "lab", "models", name, "weights.npz")))
        T = int(np.asarray(W["seq_len"]).ravel()[0])
        task = next(v for k, v in TASKS.items() if k in name)
        W = {k: W[k] for k in ("W_x", "W_h", "W_o")}
        return AlgZooRNN(name, weights=W), W, T, f"a trained network that {task}", "arc", None
    if name.startswith("planted_sum"):
        h, T = (int(v) for v in name.split("_")[2:4])
        W, truth = planted_sum(h, T, seed=h * 100 + T)
        return AlgZooRNN(name, weights=W), W, T, "unknown", "planted", truth
    if name.startswith("random"):
        h, T = (int(v) for v in name.split("_")[1:3])
        rng = np.random.default_rng(h * 1000 + T)
        W = {"W_x": rng.normal(size=(1, h)), "W_h": rng.normal(scale=0.6, size=(h, h)), "W_o": rng.normal(size=(h, T))}
        return AlgZooRNN(name, weights=W), W, T, "unknown", "mustfail", None
    if name.startswith("shuffled"):
        h, T = (int(v) for v in name.split("_")[1:3])
        W0 = dict(np.load(os.path.join(ROOT, "lab", "models", f"zoo_2nd_argmax_{h}_{T}", "weights.npz")))
        rng = np.random.default_rng(5)
        W = {k: rng.permutation(np.asarray(W0[k]).ravel()).reshape(np.asarray(W0[k]).shape) for k in ("W_x", "W_h", "W_o")}
        return AlgZooRNN(name, weights=W), W, T, "unknown", "mustfail", None
    raise KeyError(name)


# ---------------------------------------------------------------- the box, run alone
RUNNER = r'''
import json, resource, sys, time, traceback
code, spec, out, budget, mem_mb, evolve = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
cap = int(budget) + 2
resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 1))
try:
    resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
except (ValueError, OSError):
    pass
sys.path.insert(0, evolve)
import numpy as np
import boxkit
res = {"ok": False}
c0 = time.process_time()
try:
    z = np.load(spec)
    net = boxkit.Net({k: z[k] for k in ("W_x", "W_h", "W_o")}, int(z["T"]), str(z["task"]))
    import importlib.util
    s = importlib.util.spec_from_file_location("candidate", code)
    mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod)
    bp = mod.explain(net, budget)
    json.dumps(bp, default=float)
    res = {"ok": True, "blueprint": bp}
except BaseException as exc:
    res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400], "traceback": traceback.format_exc()[-1500:]}
res["cpu"] = round(time.process_time() - c0, 2)
open(out, "w").write(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
'''


def check_code(code):
    """Static rules (the statement tells the agents): only the listed imports; no file, process or reflection
    access. Returns (ok, why)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    return False, f"import {a.name} is not allowed"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") not in ALLOWED_IMPORTS or node.level:
                return False, f"from {node.module} import ... is not allowed"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return False, f"{node.id} is not allowed"
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            return False, f".{node.attr} is not allowed"
    if "def explain" not in code:
        return False, "no function explain(net, budget_seconds)"
    return True, ""


def run_box(code_path, name, W, T, task, budget, mem_mb=1200):
    with tempfile.TemporaryDirectory() as d:
        spec, out, runner = os.path.join(d, "net.npz"), os.path.join(d, "out.json"), os.path.join(d, "runner.py")
        np.savez(spec, T=T, task=task, **W)
        open(runner, "w").write(RUNNER)
        t0 = time.time()
        try:
            subprocess.run([sys.executable, "-I", runner, code_path, spec, out, str(budget), str(mem_mb), HERE],
                           cwd=d, timeout=budget * 3 + 30, capture_output=True,
                           env={"PATH": os.environ.get("PATH", ""), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1"})
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"no reply within {budget * 3 + 30:.0f} s of wall time", "cpu": budget}
        if not os.path.exists(out):
            return {"ok": False, "error": f"killed (over {budget:.0f} s of CPU or out of memory)", "cpu": budget,
                    "wall": round(time.time() - t0, 1)}
        return json.load(open(out))


# ---------------------------------------------------------------- reading and scoring a blueprint
def constants(text):
    try:
        tree = vocab.parse(text)
    except Exception:
        return []
    found = []

    def walk(t):
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            found.append(float(t))
        elif isinstance(t, list):
            for s in t:
                walk(s)
    walk(tree)
    return found


def simple(c):
    return abs(c * 2 - round(c * 2)) < 1e-9 and abs(c) <= 10


def library_constants(lib, name):
    src = (lib.get(name) or {}).get("source", "")
    try:
        return [n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                and not isinstance(n.value, bool)]
    except SyntaxError:
        return []


def to_blueprint(d, H):
    if not isinstance(d, dict) or not isinstance(d.get("parts"), list):
        raise ValueError("the blueprint must be a dict with a list 'parts'")
    parts = []
    for q in d["parts"]:
        site = str(q.get("site", "input"))
        basis = q.get("basis")
        basis = None if basis is None else np.asarray(basis, float).reshape(-1)
        parts.append(Part(str(q["id"]), str(q.get("type", "rule")), str(q.get("vtype", "Num")), site, basis=basis,
                          encoding="numerical", scale=float(q.get("scale", 1.0)), definition=q.get("definition"),
                          upstream=[str(u) for u in (q.get("upstream") or [])]))
    lib = d.get("library") or {}
    return Blueprint("seq", parts, str(d.get("output", "")), library=lib, name="candidate")


ZERO = {"score": 0.0, "run_alone": 0.0, "q": 0.0, "parts": 0.0, "credited": 0.0}


def score_one(name, res, budget):
    sc, m, fb = _score_one(name, res, budget)
    return sc, {**ZERO, **m, "score": sc}, fb


def _score_one(name, res, budget):
    net, W, T, task, kind, truth = load(name)
    rng = np.random.default_rng(SECRET_SEED + sum(map(ord, name)))
    X = rng.standard_normal((N_HELD, T))
    lines, m = [], {}
    label = {"arc": f"ARC AlgZoo network {name[4:]} (hidden {W['W_h'].shape[0]}, length {T}; {task})",
             "planted": f"planted network {name} (hidden {W['W_h'].shape[0]}, length {T}; task unknown to the box)",
             "mustfail": f"structureless network {name} (random weights; task unknown to the box)"}[kind]
    if not res.get("ok"):
        why = res.get("error", "no blueprint")
        sc = 1.0 if kind == "mustfail" else 0.0
        tb = (res.get("traceback") or "").strip().splitlines()[-3:]
        return sc, {"score": sc}, [f"{label}: score {sc:.3f}. The box failed: {why}"] + [f"    {t}" for t in tb]
    try:
        bp = to_blueprint(res["blueprint"], W["W_h"].shape[0])
    except Exception as e:
        sc = 1.0 if kind == "mustfail" else 0.0
        return sc, {"score": sc}, [f"{label}: score {sc:.3f}. The blueprint could not be read: {e}"]
    # parts that can be evaluated: rules whose upstream parts can all be evaluated; unresolved and lookup parts (and
    # rules resting on them) are honest notes, listed in the report and not credited
    ids = {p.id: p for p in bp.parts}
    good = {}

    def evaluable(pid, seen=()):
        if pid in good:
            return good[pid]
        p = ids.get(pid)
        ok_ = (p is not None and p.type == "rule" and isinstance(p.definition, str) and pid not in seen
               and all(evaluable(u, seen + (pid,)) for u in p.upstream))
        good[pid] = ok_
        return ok_
    for pid in ids:
        evaluable(pid)
    dropped = [pid for pid, g in good.items() if not g]
    if bp.output not in good or not good[bp.output]:
        sc = 1.0 if kind == "mustfail" else 0.0
        why = ("no output part" if bp.output not in ids else
               f"the output part {bp.output} depends on parts with no definition ({', '.join(dropped[:6])}), so the "
               "blueprint cannot run alone")
        return sc, {"score": sc}, [f"{label}: score {sc:.3f}. {why}."]
    bp.parts = [p for p in bp.parts if good[p.id]]
    chk = VecChecker(net, X, seed=0, n_pairs=N_PAIRS, control_reps=CONTROL_REPS)
    ok, why = chk.form(bp)
    if not ok:
        sc = 1.0 if kind == "mustfail" else 0.0
        return sc, {"score": sc}, [f"{label}: score {sc:.3f}. The blueprint fails the form check: {why}"]
    rep = chk.certify(bp, run_alone_line=0.99)
    a = float(chk.run_alone(bp))
    line = float(rep.get("iia_line", 1.0))
    located = [p for p in bp.parts if p.site not in ("input", "readout") and p.type == "rule"]
    vals = chk.values(bp, X)
    credits, notes = {}, []
    for p in located:
        iia, n = chk._iia(bp, p, np.asarray(p.basis, float))
        if iia is None or n < MIN_DECISIVE:
            credits[p.id] = 0.0
            notes.append(f"  part {p.id} at {p.site} = {p.definition}: untested (only {n or 0} input pairs where the "
                         f"blueprint predicts that swapping it changes the decision; {MIN_DECISIVE} needed): credit 0")
            continue
        c = max(0.0, min(1.0, (iia - line) / max(1e-9, 1 - line)))
        cs = constants(p.definition) + [x for t in (p.definition or "").replace("(", " ").replace(")", " ").split()
                                        if t.startswith("$") for x in library_constants(bp.library, t[1:])]
        fitted = [x for x in cs if not simple(x)]
        if fitted and c > 0:
            c *= 0.25
        credits[p.id] = c
        notes.append(f"  part {p.id} at {p.site} = {p.definition}: IIA {iia:.3f} on {n} pairs against the line "
                     f"{line:.3f}: credit {c:.3f}" + (f" (fitted constants {fitted[:3]}: a quarter)" if fitted and c > 0 else ""))
    k_min = len(truth) if kind == "planted" else 2
    q = sum(credits.values()) / max(len(located), k_min)
    credited = [pid for pid, c in credits.items() if c > 0]
    std = min(1.0, a * q + 0.02 * a)
    m.update({"run_alone": a, "q": q, "parts": len(located), "credited": len(credited), "line": line})
    if kind == "arc":
        sc = std
    elif kind == "planted":
        rec = 0
        for tn, texpr in truth.items():
            tv = np.asarray(vocab.evaluate(texpr, vocab.get("seq"), {"x": X}), float)
            for pid in credited:
                pv = np.asarray(vals[pid], float).ravel()
                if pv.std() > 0 and abs(np.corrcoef(pv, tv)[0, 1]) >= 0.99:
                    rec += 1
                    break
        recovery = rec / len(truth)
        m["recovered"] = recovery
        sc = 0.5 * std + 0.5 * recovery
        notes.append(f"  planted variables reproduced by a credited part: {rec} of {len(truth)}")
    else:
        sc = max(0.0, 1.0 - sum(credits.values()))
        notes.append("  (a structureless network: every credited rule part is a story about noise; its credit is subtracted)")
    m["score"] = sc
    head = (f"{label}: score {sc:.3f}. Run alone the blueprint agrees with the network on {a:.3f} of held-out inputs; "
            f"{len(credited)} of {len(located)} located parts credited (q {q:.3f}); family-wise control line {line:.3f}; "
            f"box CPU {res.get('cpu', 0):.1f} s of {budget:.0f}")
    if dropped:
        notes.append(f"  not evaluable, not credited (no definition, or resting on a part with none): {', '.join(dropped[:10])}")
    return sc, m, [head] + notes


def evaluate(code, budget=20.0, nets=None, mem_mb=1200, workers=1):
    """Score one program on the training networks (or `nets`). Returns {valid, score, feedback, metrics, error, cpu}."""
    ok, why = check_code(code)
    if not ok:
        return {"valid": False, "score": 0.0, "feedback": [f"The program was not run: {why}."], "metrics": {},
                "error": why, "cpu": 0.0}
    nets = nets or TRAIN
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "candidate.py")
        open(path, "w").write(code)
        results = {}
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(workers) as ex:
                futs = {n: ex.submit(lambda n=n: run_box(path, n, *load(n)[1:4], budget, mem_mb)) for n in nets}
                results = {n: f.result() for n, f in futs.items()}
        else:
            for n in nets:
                results[n] = run_box(path, n, *load(n)[1:4], budget, mem_mb)
    total, feedback, metrics, cpu = 0.0, [], {}, 0.0
    for n in nets:
        try:
            sc, m, fb = score_one(n, results[n], budget)
        except Exception as e:      # the checker itself must never crash the loop
            sc, m, fb = (1.0 if n.startswith(("random", "shuffled")) else 0.0), {}, [f"{n}: the check failed: {type(e).__name__}: {e}"]
        total += sc
        cpu += float(results[n].get("cpu") or 0)
        feedback += fb
        metrics.update({f"{n}.{k}": v for k, v in m.items()})
    score = total / len(nets)
    metrics["score"] = score
    feedback.insert(0, f"Score {score:.4f}: the mean over {len(nets)} networks (each 0 to 1).")
    return {"valid": True, "score": score, "feedback": feedback, "metrics": metrics, "error": "", "cpu": cpu}


if __name__ == "__main__":
    code = open(sys.argv[1]).read()
    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    r = evaluate(code, nets=HELDOUT if split == "heldout" else TRAIN)
    print("\n".join(r["feedback"]))
