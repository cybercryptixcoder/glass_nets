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

## The circle is there, but the decision uses the line (`_oracle/line_vs_circle.py`, E32)

At depths 4-8 both a circular code (cos, sin of the month) and the linear position can be read from the sites with
R2 = 1.00. Removing the 1-D line direction for month1 at depth 8 leaves 77.5% of the network's clear decisions
(a constant answer: 80.8%); removing the 2-D circle plane leaves 99.8%; a random 2-D plane, 100%. For month2 at
depths 6-8: 84.4% vs 100%. At depth 10 neither matters (the comparison is done). **The model represents months
both ways and uses the line for this decision.**

## It replicates across sizes and model families (`lab/line_vs_circle.py`, E33-E34)

Same prompts in three models; each month's line position is measured on the model itself. Behaviour (half fitted,
half scored) and causal removals at the depths where the months' information is used ("margin R2 kept" = share of
the next-vs-same margin's variation over prompts that survives; 1 = untouched):

| Model | Order of the months explains the margin (R2) | Circular difference (R2) | Positions vs calendar (rank corr.) | month1 line removed | month1 circle removed | month2 line removed | month2 circle removed |
|---|---|---|---|---|---|---|---|
| GPT-2 small (12 layers) | 0.73 | 0.05 | +0.98 / -0.98 | **0.20** (depth 8) | 0.96 | **0.55** (depth 8) | 0.96 |
| GPT-2 medium (24 layers) | 0.77 | 0.04 | +0.99 / -0.99 | **0.29** (depth 12) | 0.96 | **0.46** (depth 12) | 0.95 |
| Pythia-160m (GPT-NeoX, rotary, parallel blocks) | 0.70 | 0.08 | +0.95 / -0.93 | **0.50** (depth 8) | 0.99 | **0.50** (depth 8) | 0.99 |

A random 2-D plane removed at the same sites: 1.00 everywhere. **In all three models the decision runs on a 1-D
calendar line and is almost untouched by removing the circle.** (Full numbers: `lab/models/<target>/line_vs_circle.json`.)

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

## Novelty and how publishable this is (literature checked 2026-11-17 11:35)

Closest prior work and how this differs:

| Work | What it shows | Difference from this finding |
|---|---|---|
| Engels et al., *Not All Language Model Features Are Linear*, ICLR 2025 (arXiv 2405.14860) | Circular month / weekday features in GPT-2 small (layer 7); circles causally used for months/days modular addition in Mistral 7B and Llama 3 8B | Here the SAME model holds the circle, but a different calendar computation (is the range into the next year?) runs on a 1-D LINE; the circle is causally inert for it (E32) |
| arXiv 2605.29126 (2026), calendar-date duration reasoning | A sin/cos probe decodes day-of-year, yet ablating it does nothing; a 4-D DAS subspace carries the computation; QK offsets at +/-30, 61 days | Same "readable but unused circle" theme, for durations in a larger model. New here: the USED representation is identified as a 1-D calendar line, with a complete verified causal abstraction (5/5 stages), continuous steering, and the two-hop route via the year token, in GPT-2 small |
| Gould et al. 2023 (successor heads) | Heads that increment ordinal tokens (months, numbers) | Increment, not comparison |
| Hanna et al. 2023 ("greater-than" in GPT-2 small) | Year comparison "from 1732 to 17__" | Compares two-digit years via MLP lookups; here two month WORDS are compared on a learned line, across positions via the year token |
| Niu et al. 2025 (*Shape Happens*, SMDS) | Temporal entities form manifolds: circles, lines, clusters | Geometry, not which geometry a given decision uses causally |

**Honest assessment.** The claim "for year rollover GPT-2 small compares months on a line it holds next to its month
circle, and the circle is unused" is, as far as this search finds, new, and it is backed by the field's strongest
kind of evidence (complete causal abstraction with interventions, a steering prediction no prompt can show,
prior-only control, causal dissociation line vs circle). It was workshop-sized with one model; with the replication in GPT-2 medium and Pythia-160m (two families) it is a
solid workshop paper and a credible start for a main-track one. For main track it still needs: (1) the same test in a
7-8B model (Llama 3 8B / Mistral 7B, where Engels et al. show circles ARE used for month arithmetic: does the same
model switch geometry by task?); (2) other templates (the comma
template changes the behaviour itself: characterise when the line is used); (3) days of the week ("Friday -
Monday": next week?), to show it is a general strategy for ordered cyclic categories; (4) the components after
depth 8 that compute the comparison.

**Paper in one sentence:** language models hold months both on a circle and on a line, and which one a computation
uses depends on the computation: deciding whether a date range wraps into the next year runs on the line in GPT-2
small, GPT-2 medium and Pythia-160m, with the circle causally inert.

References: arXiv 2405.14860; arXiv 2605.29126; Gould et al. 2023, arXiv 2312.09230; Hanna et al. 2023,
arXiv 2305.00586; Niu et al. 2025 (frankniujc.github.io/research/arxiv2025/shape_happens.pdf).

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
