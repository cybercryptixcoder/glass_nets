# Two geometries for one calendar: language models compare months on a line, not on their circle

*Draft, 2026-11-17. Every number is reproducible from the files cited; experiments are E21-E37 in
`docs/EXPERIMENT_LOG.md`; the full write-up is `lab/constructions/gpt2_month_rollover/DISCOVERY.md`.*

## Abstract

Language models are known to represent months and weekdays on circles, and to use those circles for modular
calendar arithmetic (Engels et al., ICLR 2025). We study a different calendar computation: deciding whether a date
range wraps into the next year ("March 1987 - January" -> 1988; "January 1987 - March" -> 1987). In GPT-2 small,
GPT-2 medium and Pythia-160m, the model's decision is explained by the ORDER of the two months on a January-to-December
line (margin R2 0.70-0.77) and not by their circular distance (0.04-0.08). Causally, removing a single calendar-line
direction from the residual stream removes 44-80% of the decision margin's variation across prompts, while removing
the two-dimensional month circle at the same sites removes 1-4%, an unordered month code 6-18%, and a random plane 0%.
The result does not depend on choosing the line from the output: an unfitted calendar-index line has the same effect,
and the single line accounts for 76-100% of each month's whole causal contribution at those sites. Steering along it
moves the decision monotonically in all three models ("next year" 43-59% -> 93-100% for a 12-month shift).
In GPT-2 small we give a complete causal abstraction of the behaviour (all depth stages verified by substitution,
interchange and partial interventions; behaviour matched on 97.4% of clear decisions) in which each month is a single
scalar by layer 8, trace the route of the first month's position through the year token (heads 1.0, 2.4, 4.11 ->
6.1, 7.11), and confirm a prediction no prompt can show: sliding the scalar continuously, including between months,
moves the decision as the abstraction predicts on 90% of clear decisions. Models thus hold both geometries for the
same concept and the task decides which one is used.

## Contributions

1. A behaviour (year rollover from two months) whose computation runs on a linear ordinal month code, while the
   circular code known from prior work is present and readable (probe R2 1.00) but causally inert for it.
2. Replication across sizes and families (GPT-2 small / medium, Pythia-160m) with controls: circle, unordered month
   code, random plane, unfitted calendar index.
3. A complete, intervention-verified causal abstraction in GPT-2 small, a continuous-steering confirmation, and the
   attention route through the year token.
4. The tool that produced and verified it (`lab/`): an LM interpretability constructor with a judge of causal
   abstractions, target scouting, information-flow maps, transfer tests and circuit attribution.

## Method (short)

Prompts: every month pair x 20 years (GPT-2 small, 2,880 prompts) or x 5 years (720); output = logits of the prompt's
own year, next year and previous year. Decisions are compared only where the network's top-two margin is >= 0.5
logit. Line positions: each month's additive effect on the next-vs-same margin, or the plain calendar index. Codes are
encoded at the residual sites holding each month (month1: year and month2 tokens; month2: its own token) by an affine
map from the code to the activations, fitted on half the prompts; removal sets the activations' component in that
subspace to its mean; effects measured on the other half. Causal abstraction judged by `lab/lmjudge.py`.

## Results

Figure: `docs/figures/month_line_vs_circle.png` (left: each month's effect follows the calendar in all three
models; right: what survives removing each code). Tables: DISCOVERY.md and `docs/RESULTS_LM.md`.

## Related work

Engels et al. 2025 (circular month/weekday features, used for modular arithmetic in 7-8B models); arXiv 2605.29126
(2026: a sin/cos date probe is readable but causally unused in duration reasoning; a 4-D DAS subspace is used); Gould et
al. 2023 (successor heads; ordinal-index representations of months and numbers exist, with mod-10 features: our
line is consistent with such an index, and we show it, not the circle, carries the comparison); Heinzerling & Inui 2024
(pseudo-linear numeric entities); Hanna et al. 2023 (greater-than for years in GPT-2 small); Tiblias et al. 2026
(TMLR, SMDS: date manifolds, circular calendar topology).

## Limits and what a full paper still needs

Small models only (124M-355M); one phrasing family (with a comma after the first month the comparison matters less,
19% of the margin vs 73%, but the same line carries it); the comparison computed after layer 8 is not yet traced; the
complete abstraction is hand-made with the tool, not produced by the tool's autonomous reader. Needed: a 7-8B model
(where circles are used for arithmetic: does the same model switch geometry by task?), weekdays and other ordered
cycles, the circuit of the comparison itself.
