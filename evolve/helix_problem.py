"""The Helix problem adapter: evolve the box (the interpretability program) on ARC's AlgZoo networks.

Run from helix_evolve with glass_nets on the path:
    PYTHONPATH=/path/to/glass_nets python -m lean.run --problem evolve.helix_problem --out runs/box/r1 \
        --rounds 200 --budget 15 --analyst --accept improve --eval-workers 2

The agents see `description` (the contract and how a box is judged, in words) and the evaluator's reports. They never
see evolve/black_box.py, its networks list, its held-out inputs or its planted answers. Held-out networks are scored
only for programs that set a new training best, written to <out>/heldout.jsonl, and never shown to the agents.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import black_box as BB  # noqa: E402

CODE_ZERO = open(os.path.join(HERE, "code_zero.py")).read()

DESCRIPTION = """## The problem: build a program that explains neural networks

You are improving **the box**: one Python program that takes a small trained neural network and returns a
**blueprint** of how it computes its answer: the network's own internal variables, where each one sits inside the
network, what each one computes (as a short formula in a fixed vocabulary), and how the answer is computed from them.
A blueprint is only worth what experiments on the network confirm: every variable it names is tested by intervening
on the network.

The networks are **ARC's AlgZoo** (Alignment Research Center, 2025-26): tiny recurrent networks trained on simple
tasks, released as an open challenge because their mechanisms are not understood. The same program is run on every
network, including networks it has never seen (other sizes, other training seeds, ARC's 16-unit challenge network),
so it must be general: a program that recognises particular networks is worthless.

## The networks (the `net` your program is given)

A one-layer ReLU recurrent network, no biases:

    h_0 = 0;   h_t = relu(x_t * W_x + h_(t-1) @ W_h)   for t = 1..T;   out = h_T @ W_o;   decision = argmax(out)

Inputs are sequences of T independent standard-normal numbers. Tasks include "the position of the second-largest
number" and "the position of the median number"; some networks' tasks are unknown (`net.task` says).

    net.T, net.hidden, net.n_out, net.task (a sentence, or "unknown")
    net.weights -> {"W_x": [1, H], "W_h": [H, H], "W_o": [H, n_out]}  (numpy arrays)
    net.sites -> ["h1", ..., "hT", "out"]    (the hidden state after each step, and the output scores)
    net.inputs(n, seed) -> X [n, T]          (fresh inputs; use as many as you like)
    net.run(X, edits=None) -> (out [n, n_out], sites {"h1": [n, H], ..., "hT": ..., "out": ...})
    net.decide(X, edits=None) -> decisions [n]
    edits = {site: function(h) -> h}: replaces that site's activations as the network runs (later steps see the
    edited state). This is how you run your own interventions: e.g. swap a direction's value between two inputs and
    see whether the decision changes the way your explanation predicts.

## The vocabulary (definitions)

Every variable's definition is an S-expression in the frozen vocabulary for numeric sequences (domain `seq`):
- base features: `x` (the sequence, type Seq), `idx` (positions, Seq), `n` (the length, Num); constants;
  `@id` refers to another part of the blueprint;
- operations: add sub mul max2 min2 gt eq neg abs ge le relu mod floordiv where rank sort rev cumsum cummax cummin
  lag bcast sum max min mean argmax argmin count at (`(at x 2)` is the third number; `(lag x 1)` shifts by one;
  `(rank x)` is the 0-based ascending rank; `(argmax s)` the first position of the maximum; `(bcast v)` a Num as a Seq);
- at most **L = 7 symbols** per definition (a part reference counts one), at most **5 constants**. Longer ideas are
  written as several parts that refer to each other.
- Invented primitives (optional): `"library": {"name": {"kind": "operation", "args": ["Num", ...], "ret": "Num",
  "source": "def name(a, b):\\n    return ..."}}`, used as `($name @a @b)`; plain arithmetic Python, no strings.

## What `explain(net, budget_seconds)` returns

    {"parts": [part, ...], "output": "<id of the part whose value is the decision>", "library": {...}}

    part = {"id": "s1", "type": "rule", "site": "h3", "definition": "(max x)", "upstream": ["..."],
            "basis": [H floats, a unit vector], "scale": 2.1, "vtype": "Num"}

- `site` is where the variable lives: a hidden state "h1".."hT"; or "input" (only a direct read such as `(at x 0)`);
  or "readout" (a part computed from located parts only, e.g. comparisons that turn variables into the decision;
  readouts are not tested by intervention and may not read the input).
- A located part's value in the network is `scale * (h_site @ basis)`; its definition says what that value is.
  `boxkit.fit_location(sites, site, values, rows)` fits basis and scale by least squares.
- `type`: "rule" (defined; tested), "lookup" or "unresolved" (not credited, honest when nothing fits).
- The output part must be computed from located parts (directly or through readouts): a formula of the input alone
  is not an explanation and is rejected.

## How a blueprint is judged (the evaluator is sealed; this is what it measures)

On held-out inputs your program never sees:
1. **Form**: the vocabulary's rules above; any violation and the network scores 0.
2. **Run alone**: the share of inputs on which the blueprint's decision (all definitions evaluated, no network)
   equals the network's decision: `a`.
3. **Interventions**: each located rule part is swapped between pairs of inputs inside the network (the other parts
   at the same site held fixed); on the pairs where the blueprint predicts that the decision changes, its IIA is the
   share where the network's decision changes exactly as predicted. A control line comes from random directions and
   from weight-shuffled copies of the network (the best of many such controls). A part's credit is its margin over
   the line, (IIA - line) / (1 - line), from 0 to 1; 0 if fewer than 20 pairs test it; **a quarter if its definition
   uses a constant that is not an integer or a half between -10 and 10** (a fitted number is a weight restated, not
   an explanation).
4. Per ARC network: `a * q + 0.02 * a` (at most 1), where q = the sum of the credits over max(number of located rule
   parts, 2). So behaviour counts only as far as the parts that carry it are certified; parts that fail dilute q; full
   marks need at least two certified variables that together reproduce the decision.
5. Also scored: a hand-built network with a known variable (half the score above, half whether a credited part
   reproduces the known variable) and a network with random weights, where any credited rule is a story about
   noise and its credit is subtracted (an honest box reports nothing there).
The score is the mean over the networks (0 to 1). The report lists every network and every part: its IIA, the line,
its credit, why it got nothing. Metrics named `<network>.score`, `.run_alone`, `.q`, `.credited` let a prediction
name what should move.

## Rules for the program

- One file defining `explain(net, budget_seconds)`. It is run once per network, from scratch, in its own process.
- CPU: `budget_seconds` per network (check `time.process_time()`; finish with a margin; killed if over). Memory ~1 GB.
- Allowed imports: numpy, scipy (optimize, linalg, stats, special), math, itertools, functools, collections, heapq,
  random, time, typing, dataclasses, statistics, bisect, copy, re, operator, and `boxkit`. No files, no network
  access, no exec/eval/open (checked before it runs).
- `boxkit` (import boxkit as bk): `bk.evaluate(expr, X, parts=None, types=None, library=None)` (a definition's
  value on inputs X: [n] for Num, [n, T] for Seq); `bk.admissible(expr, types=None, library=None)` -> (ok, why, type);
  `bk.enumerate_exprs(X, max_len, parts=None)` -> {type: [(text, values, length)]} (distinct by value on X: use a few
  dozen probe inputs); `bk.best_definitions(X_fit, target, max_len, root="Num", k=5, X_test=None, target_test=None,
  parts_fit=None, parts_test=None)` -> [{"definition", "corr"/"probe_corr", "heldout_corr", "scale", "offset"}]
  (by |correlation|: the sign and scale of the match are in "scale"/"offset"); `bk.fit_location(sites, site, values,
  rows)` -> (unit basis, scale); `bk.is_simple(c)`; `bk.L`, `bk.MAX_CONSTANTS`.
"""

PLANNER_SYSTEM = """You lead a research programme in mechanistic interpretability. The programme improves one \
program, the box, which explains small trained recurrent networks as certified blueprints. Each generation you plan \
{k} experiments; for each, a coder changes a base program you choose, and the sealed evaluator scores the new \
program on several networks ({budget:.0f} s of CPU per network). Every program is kept; only programs that beat \
their base can be built on.

Work like a careful researcher. Start from the analyst's diagnosis and the evaluator's report: what exactly fails, \
on which networks, and why. Each experiment tests one idea against one named limit, concrete enough for a coder to \
implement without guessing, with a prediction that names the metric that should move. Prefer ideas that make the \
box find the network's real variables (where the hidden state carries a quantity the decision depends on) over \
tricks that move one number. A box that is true but explains little beats one that claims much and is refuted. \
Spread your bets between refining the strongest line and testing a genuinely different method (for example: \
reading the hidden state at earlier steps, searching definitions for directions the decision depends on, \
running the box's own interventions before claiming a part, inventing a primitive when nothing in the vocabulary \
fits). Don't repeat an experiment that failed unless you say what is different."""

CODER_SYSTEM = """You are an expert in mechanistic interpretability and Python. You implement one planned \
experiment as a change to the box (a program defining explain(net, budget_seconds) that returns a blueprint dict), \
faithfully, and return the complete new program. It must finish within budget_seconds of CPU per network (check \
time.process_time()), use only the allowed imports, and never crash: wrap risky steps so that a failure still returns \
a valid blueprint (an honest unresolved part is better than an exception)."""

MUTATOR_SYSTEM = """You are an expert in mechanistic interpretability and Python. You improve the box: given one \
version, you write a better one, with one substantial change you expect to raise its score (more of each network's \
real computation explained and certified). The program defines explain(net, budget_seconds), returns a blueprint \
dict, finishes within its CPU budget per network (check time.process_time()), uses only the allowed imports and never \
crashes (an honest unresolved part is better than an exception)."""

MUTATION_PLANNER_SYSTEM = """You are an expert in mechanistic interpretability. You improve the box: given one \
version and its evaluation report, you choose one substantial change you expect to raise its score and describe it \
for a coder. You plan; you don't write code."""

DISTILLER_ASK = """## Your task

Record what generation {gen} taught. Write new findings, and attach this generation's programs as evidence to \
existing findings they bear on. Good findings are specific and reusable: "locating the running maximum at h2 as well \
as h3 (p0031) raised M4,3's credited parts from 1 to 2 and its score from 0.05 to 0.21", not "p0031 did better". \
Record failures and their cause too (errors, timeouts, refuted parts, no gain), and open questions worth testing. \
Link findings when one refines, contradicts, requires, or combines with another, or explains it.
Kinds: works (a change that raised the score), fails (a change that didn't, or broke), mechanism (what a network \
actually computes, or why something works), question (worth testing, not yet tested).
Reply with one JSON object and nothing else:
```json
{{"new": [{{"kind": "works", "claim": "...", "evidence": ["p0031"],
           "links": [{{"to": "F3", "type": "refines"}}]}}],
 "evidence": [{{"finding": "F3", "programs": ["p0033"]}}]}}
```
At most 6 new findings. Link types: refines, contradicts, requires, combines-with, explains. Use ids exactly as \
given (programs p0001..., findings F1...)."""


class Problem:
    def __init__(self, heldout_log="", workers=1):
        self.name = "box-algzoo"
        self.title = "Evolving the box: certified blueprints of ARC's AlgZoo networks"
        self.description = DESCRIPTION
        self.code_zero = CODE_ZERO
        self.seed_idea = "Code Zero: the logits at the last state, each defined by the best-correlated short expression"
        self.references = []
        self.entry = "explain"
        self.planner_system = PLANNER_SYSTEM
        self.coder_system = CODER_SYSTEM
        self.mutator_system = MUTATOR_SYSTEM
        self.mutation_planner_system = MUTATION_PLANNER_SYSTEM
        self.distiller_ask = DISTILLER_ASK
        self.generation_note = "Each program gets {budget:.0f} s of CPU per network; it is run once per network, from scratch."
        self.task_note = ("budget_seconds will be {budget:.0f} (CPU seconds for one network); the program is run once per "
                          "network, from scratch, in its own process (there is no `start`).")
        self.clip = {"base": 40000, "best": 24000, "inspiration": 10000, "opened": 4000}
        self.reply_tokens = 16000
        self.heldout_log = heldout_log
        self.workers = int(workers)
        self._best = None
        self._lock = threading.Lock()

    def evaluate(self, code, start, budget, mem_mb):
        res = BB.evaluate(code, budget=budget, nets=BB.TRAIN, workers=self.workers)
        res["metrics"] = {k: float(v) for k, v in (res.get("metrics") or {}).items()}
        res["construction"] = None
        res["start_score"] = None
        if res["valid"] and self.heldout_log:
            with self._lock:
                new_best = self._best is None or res["score"] > self._best
                if new_best:
                    self._best = res["score"]
            if new_best:          # held-out networks: recorded for people, never shown to the agents
                h = BB.evaluate(code, budget=budget, nets=BB.HELDOUT, workers=self.workers)
                with open(self.heldout_log, "a") as f:
                    f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "train": res["score"],
                                        "heldout": h["score"], "code_sha": __import__("hashlib").sha1(code.encode()).hexdigest()[:12],
                                        "heldout_metrics": {k: float(v) for k, v in h["metrics"].items()},
                                        "heldout_report": h["feedback"]}) + "\n")
        return res
