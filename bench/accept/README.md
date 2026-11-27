# The acceptance suite: how to call the scorer

*Written by the bench-writer (`docs/spec/BENCH_BRIEF.md`), Nov 27, 2026. The interface the box must meet is
`bench/accept/INTERFACE.md` (code: `bench/accept/interface.py`). Changes to the interface and to the scorer's code are
logged in `bench/accept/CHANGES.md`.*

## What is here

| path | what it is |
|---|---|
| `INTERFACE.md`, `interface.py` | the contract: the task folder, the blueprint, the box's entry points, the format check |
| `netlib.py` | the network loader: `load_network(task folder)` runs a network and intervenes at its named sites |
| `scorer/` | the development copy of the scorer (the scorer of record is kept privately until certification) |
| `vendor/` | the scorer's frozen copies: `vocab/` (vocabulary v2 at f0b3b680) and `lab_nets.py` (lab/nets.py at b629f1e3) |
| `draws/<draw id>/<network id>/` | development networks: task folders the box explains |
| `sealed/` | gitignored: the development draws' sealed answers, which the scorer reads and the builder never opens |

## Calling the scorer

```
# score one blueprint against one development network (prints pass or fail, the score and the line per requirement)
python3 bench/accept/scorer/score.py --task bench/accept/draws/<draw>/<net> --blueprint <out>/blueprint.json

# the interface's reference (passes at the ceiling)
python3 bench/accept/scorer/score.py --task bench/accept/reference/frac_prevs \
    --blueprint bench/accept/reference/frac_prevs/true_blueprint.json

# score every network of a draw that has a blueprint in <out dir>/<net>/blueprint.json
python3 bench/accept/scorer/score.py --draw <draw> --blueprints <out dir>

# Gate 1's verifier hook: feeds the builder's checker (Box.verify) the sealed true blueprints and S5a accounts
python3 bench/accept/scorer/score.py --hook --draw <draw> --start <start config.json>
```

The scorer reads sealed answers from `bench/accept/sealed/<draw>/`. On development draws it prints, per network and per
requirement, pass or fail with the box's score and the line, and, for Gate 2's diagnostics, how many sealed
intermediate variables were recovered and which of the box's own parts matched one. On the certification draw it
prints only pass or fail, per requirement and per tier. It never prints a sealed definition, a family or a mechanism.

## How each tier is scored

- **Planted tiers (S0, S1):** every requirement's line is the network's own true blueprint's score on the draw being
  scored, minus the tier's spread (frozen from dev-001 to dev-003). Interchange is also held to a family-wise line
  from random subspaces at the same sites (the 99th percentile of the best of K controls, K the parts tested); a part
  is interchange-tested only on at least 20 decisive pairs. A sealed variable is recovered when one of the box's parts
  reaches its own agreement (or correlation) and substitution scores at the sealed location, minus the spread.
- **S5a (accounts to reject):** fed to the builder's checker through `--hook`, which prints only counts. The scorer
  also applies its own checks (run alone against the task-rule baseline; a part typed lookup at the decision must be
  at chance on fresh inputs; interchange of sampled parts against the family-wise line). R10 and R9's estimator test
  are pending.
- **S5b (no story):** the box explains the network; it passes when no part typed rule is above the family-wise
  interchange line, and every lookup's keys are verified training items where the task lists them
  (`train_items.json`).

## Draws

| draw | tiers | networks |
|---|---|---|
| dev-001 | S0 (Tracr half), S1 (numeric sequences), S5a, S5b | S0-tracr-00 to 09; S1-seq-00 to 09; S5a-00 to 07 (the eight saved targets; the S5a accounts of the S1 networks are sealed with them); S5b-00 to 32 |
| dev-002 | S0 (Tracr half), S1 (numeric sequences) | S0-tracr-00 to 09; S1-seq-00 to 09 |
| dev-003 | S0 (Tracr half), S1 (numeric sequences) | S0-tracr-00 to 09; S1-seq-00 to 09 |

The first three development draws set the tiers' spreads, now frozen for S0 and S1. Not built yet: the InterpBench
half of S0, S1 on chess boards, S2, S3, S4, S6 and S7, and the bench-chosen probes (flips, edits, unusual inputs).

Draw requests go in `docs/build/STATUS.md` as "Draw request: <gate>, <tiers>". A ready draw is announced there as
"Draw <id> ready: <tiers>". Every formal gate attempt needs a draw that no earlier formal attempt used.

## The bench's self-check (Gate 1, first half)

`python3 bench/accept/scorer/selfcheck.py <draw> --variants [--tiers=S0,S1,S5a]` and
`python3 bench/accept/scorer/selfcheck_s5b.py <draw>` print counts only. Results (scorer-v1.2):
- **S0 (dev-001):** true blueprints passing every requirement at its line: 10 of 10; known-wrong blueprints (every
  part at a random location of the same dimension; the output's definition replaced by a type-correct wrong one; the
  true blueprint on the weight-shuffled network) failing at least one requirement: 30 of 30.
- **S1 (dev-001; dev-002):** true blueprints at their lines 10 of 10 (10 of 10); known-wrong blueprints failing 30 of
  30 (30 of 30).
- **S5a (dev-001):** accounts rejected: 51 of 51 (the 31 saved copies and disguises by the scorer's automatic tests;
  the 10 distilled students and 10 tuned tables of the S1 networks at those networks' lines); none fails only the
  format check.
- **S5b (dev-001):** a borrowed story (the parent's true blueprint) shows no rule above the family-wise line on the
  shuffled, untrained and memoriser networks: 30 of 30; the same stories are caught as rules above the line on their
  own networks (the check's positive control): 10 of 10.

## R9's estimator test (X0)

X0's main arm (`docs/review/PREREG_FIRST_EXPERIMENTS.md`, X0) was run on Nov 27 on the saved runs, with the estimators
written by a model given only the accounts and the tasks (DeepSeek V4.1 Flash; no other model was available). Neither
must-pass account beat sampling on every quantity: M4,3's (c928) estimator used millions of forward passes of compute
and lost to Monte Carlo by factors of 228 to 13,125; the lc_8_6 census rule beat sampling on 4 of 5 quantities
(ratios 0.07 to 0.27, lines about 1.0) and lost on the fifth (ratio 61, a population its rule never predicts). So
answering without running did not separate real accounts from copies here: **R9's estimator test is reported and does
not gate**; R9 gates on the must-fail set (S5a, S5b) and the bounded reader (R10, pending a key).

## The reader model (R10)

Fixed by the bench for the whole build: the frontier model whose key Shreyas supplies, at a version recorded here.
**Pending: no frontier-model key exists yet**, so the usability lines are marked pending and are computed before
Gate 5.

## Checksums of large weights

None yet: every development network so far is under 50 MB and committed in its task folder.
