# How GPT-2 small decides whether a date range rolls into the next year

*Session 4, 2026-11-17. Target `gpt2_month_rollover`: "March 1987 - January" -> 1987 or 1988? (2,880 prompts: every
month pair x 1980-1999). Everything here is reproducible from the files named; numbers are on held-out prompts.*

## The finding

GPT-2 small does **not** use its circular month representation (Engels et al. 2024) for this decision. By depth 8,
the only thing the late layers use about each month is **one number: its position on a straight January-to-December
line**. The decision is mostly "is month2 earlier on that line than month1?", plus a graded pull from month2's
position:

    next-year margin ~ 0.69 * pos(month1) + 2.53 * pos(month2) + 8.94 * [month2 earlier] + 8.8 * [same month] - 2.0

| Evidence | Result |
|---|---|
| Circular forward distance as a model of the margin | R2 0.08 (explains almost nothing) |
| Linear order (month2 before / same / after month1) | R2 0.74, 96.8% of clear decisions, 3 parameters |
| Each month's additive effect vs calendar order | rank correlation +0.979 (month1), -0.979 (month2) |
| **Judge, construction with one scalar per month at depth 8** | **VERIFIED AND COMPLETE, 5 of 5 stages, behaviour 97.4%** |
| Partial test: month1's scalar alone copied from another prompt | 100% of changed decisions predicted |
| Partial test: month2's scalar alone | 97.4% |
| Edit test (coordinate mixes = states no prompt produces) | 83.9% (bar 50%) |
| **New claim: slide month1's position continuously along its one direction at depth 8** | network goes 55% -> 99% "next year" monotonically; construction predicts 90% of clear decisions, 89% at positions BETWEEN months |
| Prior-only control (reader's memory, no access to the model) | 0 of 5 stages |

Files: `lab/constructions/_oracle/gpt2_month_rollover_linear_positions.py` (the construction) and `.txt` (judge
report); `lab/constructions/_oracle/steer_month_position.py` and `.json` (the steering test).

## The route (from the complete calibration, `_oracle/gpt2_month_rollover_calibration.py`, and `flow()`)

Month1, year and month2 are at their own tokens at the embedding. From depth 2, month1's information is also on the
year token and the month2 token (to depth 9 on the year token, to the end on month2's). By depth 8 the months are
reduced to their line positions; the decision is formed between depth 8 and the output; the answer leaves as the
identity of a year token, read out by the unembedding (measured as a table).

## The circuit of the two scalars (`_oracle/gpt2_month_rollover_linear_positions_circuit.txt`)

- **Month1's position takes a two-hop route through the year token.** At the year token it is written in layers 1-4
  by heads that attend to month1 (head 1.0: 0.86 of its attention on month1; head 4.11: 1.00; head 2.4: 0.72) and by
  MLP 4. At the month2 token it arrives mainly in layers 5-7 through heads that attend to the YEAR token (head 7.11:
  0.82; head 6.1: 0.79), with MLPs 6-7 and a direct path (head 1.5 attends to month1). The date "March 1987" is bound
  at the year token and read from there by the final token.
- **Month2's position** is written at its own token mostly by MLP 0 (share 0.32; without it the scalar's read-out R2
  falls from 1.00 to 0.53).
- The circuit is distributed: no single component's mean-ablation removes more than 18% of the answers.

## What is honest to say

- **Who found it.** The tool provided every measurement (engine, `flow()`, judge, diagnoses, steering through the
  judge's maps), but the hypothesis and both constructions were written by me (the session's agent), not by the
  autonomous reader. The blinded reader got 1 of 5 stages in 150 steps (run `20261117-053452-314f`); run B is still
  going. So the TOOL VERIFIED a discovery; it did not yet MAKE one on its own on this target.
- **Transfer.** With a comma after month1 ("March, 1987 - January") GPT-2's behaviour itself changes (68% "same
  year"); the decision rule does not transfer (60% vs constant 68%), the early representation maps do (substitution
  92-100% at depths 0-4; 79% at depth 8). The linear-position finding is shown for the dash template only.
- **Not yet known.** Which components compute the comparison itself after depth 8; whether other templates use the same
  axis with other constants; the near-ties (32% of decisions within 0.5 logit) are excluded from all comparisons (E23).

## Reproduce

    python3 lab/lm_build.py lab/lm_tasks/month_rollover.py
    python3 lab/judge.py gpt2_month_rollover lab/constructions/_oracle/gpt2_month_rollover_linear_positions.py
    python3 lab/constructions/_oracle/steer_month_position.py
    python3 lab/lm_transfer.py gpt2_month_rollover lab/constructions/_oracle/gpt2_month_rollover_linear_positions.py gpt2_month_rollover_comma
