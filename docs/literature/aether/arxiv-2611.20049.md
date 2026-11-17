# The Frontier Must Grow: Self-proposed, interpreter-labelled training roughly doubles a 4B model's accuracy on a fresh exam as its verified frontier grows

- **Key:** arxiv-2611.20049
- **Authors / organisation:** Martin Schröder — Swedish Embedded AB
- **Date:** 2026-11-15
- **Link:** https://arxiv.org/abs/2611.20049
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The paper asks a narrow, testable version of a very grand question: does a model that proposes its own
training problems and has an external checker grade them actually improve, and if it does, is the
improvement due to the recursion (each round's data coming from the previous round's model) or merely to
the volume of verified data? The motivation is the self-improvement literature, where multi-round
self-training is widely reported as working, and the authors want a result that survives the obvious
objections: an exam the model was not selected against, a control trained on matched data, and a
pre-registered pass rule. The larger goal is a *procedure* whose gains are attributable, not a demo. The
paper is also explicitly a failure analysis of the three earlier series that did not work, and it says so
in the title of §4.

## What they studied

The learner is Qwen3-4B (thinking off) asked to predict what Lua 5.1 programs print. The unit of analysis
is a LoRA adapter fitted from the base, and the unit of evidence is the exam item: 889 programs built from
25 standard-library templates, selected once as the items the base never answered in four sampled attempts.
A second, later-drawn exam of 2,340 items (78 templates) is used for the follow-up, and a fresh set of
base-proposed programs of the same size is the matched-data control. Nothing here is a read-out of
internal state: the object is behaviour on a deterministic task, and the "interpretability" content is
entirely in the *measurement design*.

## How they did it

Four lineages per series, three generations per lineage, each lineage differing only by seed. In each round
the current champion proposes programs about sections of the Lua manual; the pinned interpreter runs each
one twice and the equal output is the label; the champion predicts each program's output four times at
temperature 1.0, and any program it gets wrong at least once joins the *frontier*. Rounds 1–3 add 300, 600
and 1,200 new programs, so the pool trained on grows to 300, 900 and 2,100 records. Each round trains a
fresh LoRA from the base on the whole union — the rounds share no parameters, only the accumulating dataset —
for 75, 225 and 525 steps over two passes. A learner-side gate on a 400-unit dev set plus a 300-unit
retention set decides whether the candidate becomes the next champion; a separate examiner program holds
the exam, serves every model itself on one pinned build and one KV-cache tier, checks digests, drops any
item equal to a program on either compared model's training or held-out list, grades by one greedy
direct-completion with a fixed neutral request, and runs an exact one-sided sign test of wins against losses
between a candidate and its own predecessor. Every criterion was registered before its run; the criteria,
the code commits and a claims ledger are released with the verdicts.

## What counts as an explanation here

A procedure plus an attribution of its gain: the claim is not "the model represents Lua execution in
region X" but "this training loop, not the data it happens to accumulate, is what moved the exam". The
paper is unusually careful about the *levels* at which its explanation holds — an item-level sign test
licenses a claim about two adapters, a template-clustered interval licenses a claim about families of
programs, and only replication over seeds licenses a claim about the procedure, at which point four
lineages are a count rather than a test (4 of 4 has a one-sided p of 0.0625). It is explicitly partial:
nothing is claimed about the model's internals, and the mechanism by which the frontier helps is described
only as "selects difficulty, not learnability".

## How they judged success

Registered per-series rule: a step holds when wins exceed losses at p < 0.05 by exact sign test, a lineage
succeeds when all three of its steps hold, and the series meets its criterion when at least 3 of 4 lineages
succeed. Beside that, a paired gain in the fraction of items right with a 95% Student-t interval clustered
over the exam's 25 templates (24 degrees of freedom), a base-against-base null replicate through the same
protocol (0 discordant items), a matched-data control (one fit from the base on 2,100 base-proposed frontier
programs), a **same-data, different-seed retrain** arm, and a similarity audit of the fresh exam to the
training programs. The learner's own dev gate is reported separately and is shown not to be doing the work.

## What they found

S4 met its registered criterion: all twelve steps held, lineages reaching 192, 225, 241 and 218 of the 889
items from a base of 12, and the weakest step won 67 against 36. The clustered intervals are more sober —
at generation 3 only one lineage's interval excludes zero on the two-sided reading (0.0267 to 0.1355) and
three include it, so a 3-of-4 rule would not be met at template level. On the fresh 2,340-item exam the
share answered went 8.3% → 8.7% → 13.8% → 16.7%, with round 2 adding 0.050 (0.026 to 0.079) and round 3
adding 0.029 (0.011 to 0.050); **round 1 added nothing** (0.004, −0.016 to 0.023), and the paper diagnoses
why: on the original exam step 1 was inflated by selection, because the base could lose at most the 12 items
it answered. The matched-data control answered 14.6% against generation 3's 16.7%, a 2.1-point lead whose
registered interval (−0.002 to 0.046) includes zero, so **the recursion benefit is inconclusive**; the
data alone scales (900 records 12.4%, 2,100 records 14.6%). Churn is large at item level: generation 3 lost
14–30% of the items its predecessor answered while winning 9.5–14% of those it missed, and identical-data
retrains split 87/57, 70/56, 48/56 and 35/55 against their originals while their means agreed (16.9% vs
16.7%). One identical-data retrain even passed the step rule against its own original (87/57, p = 0.0077).
Descriptively: the same base with thinking on answers at least 47.3% of the fresh exam (372 replies hit the
token budget), and a Qwen3.8-27B answers 39.3% zero-shot.

## Limits and open problems they name

The recursion claim is not demonstrated, only the data-scaling claim; the gain is specific to direct
prediction without reasoning, since reasoning already beats it. The exam that guided the design was reused
adaptively across series, and the fresh exam's items sit at a median token-set Jaccard of 0.54 to their
nearest training program. The learner's dev gate advanced any candidate with a positive net gain and
filtered nothing. The examiner is a separate program run by the same operators, its signature is symmetric
and protects against the learner rather than the authors, and the fresh exam's generator is not released.
Only two of the appendix experiments are replicated; the coding-task experiment, the 27B generations and an
unregistered documentation run showed no gain. The retention check does not test what each release
acquired. They name the open questions as richer oracle supervision (execution traces, verified reasoning),
a proposer-by-selector design that separates who proposes from who selects, compute-matched baselines, a
learnability rather than a failure curriculum, and a retention matrix.

## What we take from it

Four transfers, in order of usefulness to us. (1) **A same-data, different-seed retrain is the control our
battery is missing**, and its result here is brutal: identical data and recipe reproduced item-level
sign-tests against their own originals, so a per-pair item test is a fact about two adapters and not
evidence that a method gains. Any claim of the form "probe A beats probe B on these items" needs a
re-fit-on-identical-data arm beside it. (2) **A holdout selected on the model's failures measures progress
on that hard subset and nothing else** — the paper quantifies the artefact as regression to the mean (base
items fell 0.371 at generation 1 while base-wrong items rose 0.038) — which is exactly the bias in our
"probe only the features the model misses" evaluations. (3) **Three levels of claim with a different test
for each** (item / family / procedure) is a template worth copying verbatim into the methods section.
(4) **Abstention and withheld attempts must stay in the denominator**: a release-gate amendment notes a base
at 21/46 with 9 exposed being read as a win against a candidate at 15/46 with 19 exposed, which is the
denominator lesson the folder keeps relearning, caught here by a gate rule rather than by prose.

## Tags

unit: whole-model
form: program
method: other
eval: benchmark
goal: evaluation-of-methods
