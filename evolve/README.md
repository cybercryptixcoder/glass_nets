# evolve/: Helix evolving the box on ARC's AlgZoo networks

*Nov 28, 2026. For people and for the agent that sets a run up. The loop's own agents never see this file or
`black_box.py`; they see `helix_problem.py`'s statement and the reports the evaluator writes back.*

## What it is

The box (the program that explains a network as a certified blueprint) cut down to **one file with one contract**,
so that Helix (`cybercryptixcoder/helix_evolve`, lean, researcher mode) can evolve it with DeepSeek V4.1 Flash for
hundreds of generations, keeping a change only when the sealed evaluator's score goes up.

    explain(net, budget_seconds) -> {"parts": [...], "output": "<part id>", "library": {...}}

`net` is one of ARC's AlgZoo ReLU RNNs (or a planted or random network of the same form): weights, a sampler of
its inputs, and `run(X, edits)` for the box's own interventions. A part is a variable: where it lives (a hidden
state and a unit direction with a scale), its definition in the frozen `seq` vocabulary, and what it rests on.

| File | What |
|---|---|
| `code_zero.py` | the starting box: the logits at the last state, each defined by the best-correlated short expression, the decision as their argmax |
| `boxkit.py` | the only toolkit the box may import: the `Net`, the vocabulary (evaluate, admissible, enumerate, best definitions), `fit_location` |
| `black_box.py` | the sealed evaluator (below) |
| `helix_problem.py` | the Helix problem adapter: the statement the agents see, the domain's system prompts, the bridge to the evaluator, held-out scoring of new bests |
| `test_black_box.py` | the evaluator's known cases (must pass, must fail, rule checks): run before any run |

## The evaluator

For each network, the box runs alone in its own process (`python -I`, CPU and memory limits, only allowed imports,
checked statically: no file, process or reflection access). Its blueprint is then checked by the frozen vector-site
checker (`box/verify_vec.py` v1) on 3,000 held-out inputs from a seed the box never sees:
- **form**: the checker's rules (vocabulary definitions within L = 7 and 5 constants; every computed part located;
  the output computed from located parts only, never from the input directly);
- **run alone** `a`: the share of inputs where the blueprint's decision equals the network's;
- **interventions**: each located rule part swapped between input pairs inside the network, holding the site's
  other parts; IIA against the family-wise control line (random directions, weight-shuffled networks). Credit =
  margin over the line, (IIA - line)/(1 - line); 0 below 20 decisive pairs; a quarter for fitted constants (not
  an integer or a half in [-10, 10]: a weight restated).
- per AlgZoo network: `a * q + 0.02 a`, q = credits / max(located parts, 2). Behaviour counts only through certified
  parts, so the answer claimed as one weakly certified variable scores under 0.1.
- the planted network (running sum, with distractor units): half that, half whether a credited part reproduces the
  planted variable. The random network: 1 minus the credits of any rules (a story about noise is wrong).

Training networks (selection): AlgZoo 2nd-argmax M2,2, M4,3, M4,4; argmedian 4,3; `planted_sum_4_4`; `random_4_3`.
Held-out (only for programs that set a new best; written to the run's `heldout.jsonl`, never shown): 2nd-argmax M8,5
and ARC's 16-unit challenge network M16,10; argmedian 8,5; `planted_sum_6_5`; a weight-shuffled M4,3.

Known cases (`test_black_box.py`, all pass, Nov 28): the planted network's true account 1.000; a wrong variable
0.005; empty blueprints 0 (AlgZoo, planted) and 1 (random); behaviour only, rejected by form; the answer claimed as a
variable at the last state 0.095 (a first scoring rule gave it 0.53 and was replaced); a forbidden import and numpy
file access not run; ordinary `list.remove` runs. Code Zero: 0.357 on the training networks, 0.409 held-out. Its
AlgZoo scores are 0.005 to 0.063: the room is there.

## Known weaknesses (say so before reading any result)

- A box could recognise the training networks by their weights and return hand-tuned blueprints (tables tuned by
  selection, SPEC S5a). Selection can't see this; the held-out networks can. Read `heldout.jsonl` next to the curve.
- M2,2's control line is near 1 (a two-unit state leaves random directions no room), so M2,2 contributes almost
  nothing either way, as in session 1.
- One random network stands in for S5b's structureless networks.
- The evaluator is a fixed version for a run. A change to it is a new, named version, recorded in
  `docs/glass/CHANGES.md` (BX7), never made during a run.

## Running it

Live test (here): `PYTHONPATH=<glass_nets> python -m lean.run --problem evolve.helix_problem --out /tmp/t --rounds 1
--budget 15 --analyst --accept improve --eval-workers 2` from helix_evolve. The job for the VM is
`helix_evolve/scripts/run_box.sh` (`docs/jobs/BOX_JOB.md` there, with the prompt for Aether).
