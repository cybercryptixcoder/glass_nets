# Results on real language models (sessions 3-4)

*Hand-written; updated 2026-11-17 11:45. Generated tables for every run: `docs/RESULTS_CONSTRUCTOR.md` (`lab/results.py`).
Experiments: `docs/EXPERIMENT_LOG.md` E13-E35. The discovery: `lab/constructions/gpt2_month_rollover/DISCOVERY.md`.*

## Headline: a discovery, verified with the tool

**Language models hold months both on a circle and on a line, and deciding whether a date range wraps into the next
year ("March 1987 - January" -> 1987 or 1988?) runs on the LINE, with the circle causally inert.**

| Evidence | GPT-2 small | GPT-2 medium | Pythia-160m |
|---|---|---|---|
| Margin explained by the months' calendar order / their circular difference (R2) | 0.73 / 0.05 | 0.77 / 0.04 | 0.70 / 0.08 |
| Measured month positions vs calendar order (rank corr.) | +0.98 / -0.98 | +0.99 / -0.99 | +0.95 / -0.93 |
| Margin variation kept with month1's 1-D line removed | **0.20** | **0.29** | **0.50** |
| ... with the 2-D circle removed | 0.96 | 0.96 | 0.99 |
| ... with an unordered random month code removed (mean of 5) | 0.89 | 0.82 | 0.87 |
| ... with a random 2-D plane removed | 1.00 | 1.00 | 1.00 |
| Complete causal abstraction (judge, 5/5 stages) | **verified**, behaviour 97.4% | not attempted | not attempted |
| Steering month1's position continuously (incl. between months) | 90% of clear decisions predicted | | |
| Prior-only control (reader's memory) | 0 of 5 stages | | |

Circuit (GPT-2 small): month1's position goes month1 -> year token (heads 1.0, 2.4, 4.11, MLP 4) -> final token
(heads 6.1, 7.11 reading the year token, MLPs 6-7); month2's position is written by MLP 0.

**Who made it:** the session agent, using the tool's engine, `flow()`, judge, diagnoses, steering and circuit tools.
The autonomous blinded reader reached 1 of 5 stages on this target (150 steps) and 0 of 5 in a second run
(134 steps, interrupted). **Limits:** the comma template changes GPT-2's behaviour itself (decision rule does not
transfer); larger models (7-8B, where circles are known to be used for month arithmetic) not tested.

## Every LM target

| Target | Model | Behaviour | Best result | Prior-only control |
|---|---|---|---|---|
| `gpt2_month_rollover` | GPT-2 small | year rollover from two months | hand-made: complete 5/5 (linear positions); reader: 1/5 | 0/5 |
| `gpt2medium_month_rollover`, `pythia160m_month_rollover` | GPT-2 medium, Pythia-160m | same, 5 years | line-vs-circle replication (above) | - |
| `gpt2_be_verb` | GPT-2 small | is / are / was / were | **reader, blinded: complete 5/5** (44 steps) | 1/5 |
| `pythia70m_be_verb` | Pythia-70m | same | built, not run | - |
| `gpt2_countdown` | GPT-2 small | counting down | reader 1/5; hand-made 2/5 | 1/5 |
| `gpt2_ordinal` | GPT-2 small | st / nd / rd / th | rejected: a lookup in the token embedding (scout) | - |
| Transfer sets | | `gpt2_month_rollover_y1950`, `gpt2_month_rollover_comma` | | |

## What the tool can do now (and what it cannot yet)

- **Can:** take any GPT-2 or Pythia checkpoint and a task file; check the target (scout); give a reader its own
  experiments (`flow`, `probe`, `explain_output`, hooks); judge constructions with cuts and depth stages against
  controls, with diagnoses of what is missing; test claims on unseen prompts (transfer) and by steering; trace
  circuits; write FINDINGS.md; test line vs circle geometry in any month task (`lab/line_vs_circle.py`).
- **Cannot yet:** reliably reach a complete construction AUTONOMOUSLY on a hard target. The reader completed the
  easy target (be-verb) blinded, but not month rollover or countdown.
