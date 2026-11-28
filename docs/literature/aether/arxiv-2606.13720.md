# Refusal Beyond a Single Direction: A Preliminary Comparison of Diff-in-Means and INLP

- **Key:** arxiv-2606.13720
- **Authors / organisation:** Elisabetta Rocchetti; Alfio Ferrara — Università degli Studi di Milano, Department of Computer Science
- **Date:** 2026-06-11 (v1); v2 2026-09-10
- **Link:** https://arxiv.org/abs/2606.13720
- **Kind:** paper
- **Read from:** full text (v2 — abstract, S1–S6, references, Appendices A–E, Tables 1–13)
- **Relevance to us:** 3 (central)

## What they wanted, and why

A refusal direction recovered by differencing the mean harmful activation and the mean harmless activation is the canonical single-direction account of refusal, and recent evaluations put it at or above more elaborate extraction methods. The authors ask whether a different supervised primitive — one that yields a *subspace* instead of a vector and a one-parameter family of operators instead of one operator — can recover something the difference-in-means cannot, and whether that extra parameterisation buys interventions a single direction cannot offer. Two hypotheses are stated up front: that the subspace method can match the single-direction method at suppressing refusal, and that its richer parameterisation yields a tunable effect-versus-capability trade-off. The larger goal is to know whether "one direction" is a property of the model or a property of the estimator that found it.

## What they studied

Models: five safety-fine-tuned open-weight chat models, the smallest of each family — Gemma 2B-IT, Qwen 1.8B-Chat, Yi 6B-Chat, Llama-2 7B-Chat, Llama-3 8B-Instruct. Unit of analysis: the residual stream at a chosen **layer and post-instruction token position** (one site per model per method), read as a direction (the class-mean difference, unit-normalised) or as a rank-k row space recovered by iterating linear classifiers against the class label. Data: the Arditi-style contrastive construction — 128 harmful training prompts sampled from three harmful corpora, 32 harmful validation prompts from a held-out harmful benchmark, 128/32 harmless prompts from an instruction corpus — evaluated on 100 harmful and 100 harmless test prompts. No neuron, head, circuit or SAE feature is claimed anywhere.

## How they did it

1. Collect residual-stream activations on the contrastive sets, restricted to the post-instruction template positions.
2. Build four interventions: add or subtract the difference-in-means vector at its own layer (activation addition); project every activation at every layer onto the hyperplane orthogonal to it (directional ablation); project activations onto the nullspace of the iteratively-found classifier row space, α = 1 (nullspace projection); or reflect them across that nullspace, α = 2 (counterfactual flipping) — with the row space truncatable to k leading classifiers.
3. Select (layer, position) per method by a composite score rewarding refusal suppression on harmful prompts, refusal induction on harmless prompts, and penalising KL divergence from the unintervened model's final logits; INLP gets an added first-classifier accuracy floor.
4. Apply each intervention **only at the site it was selected at** (a departure from prior work, which intervened at every layer), so the four methods are compared on one footing.
5. Score effectiveness as signed change from baseline in substring-matching non-refusal on harmful prompts, a safety classifier's unsafety rate, and refusal rate on harmless prompts; score capability as change in 5-shot MMLU and ARC accuracy and median per-example perplexity on a text corpus plus an instruction corpus.
6. v2 adds the experiment that matters most: **cross-site application** — rerun each method at the site the *other* method selected, holding everything else fixed, so that a method-versus-method gap can be split into an intervention effect and a selection-procedure effect.
7. Two diagnostic layers on top: a 2D PCA visualisation whose first axis is pinned to the class-centroid direction, with a "target-group fit" ratio (distance from a transformed point to the opposite-class centroid, normalised by that class's own spread); and an LLM judge over 21,995 completions returning refusal status, reason tags, a satisfaction grade, and looping / clarification flags, audited by hand against substring matching.

## What counts as an explanation here

An **intervention plus a selection procedure**: a direction or subspace in the residual stream, the operator derived from it, and the (layer, position) at which the operator is applied — with the honest admission that the operators' differences are read through a geometric picture (where transformed activations land relative to the two class clouds) and a behavioural picture (what the model says), not through a claim about mechanism. It is partial by construction: one concept (refusal), one site per cell, five small models. It predicts nothing, but it does supply a *stability* statement — whether a ranking survives changing the site — which is a prediction-shaped claim about the estimator rather than about the network.

## How they judged success

Signed deltas from an unintervened baseline with standard errors on effectiveness columns and standard deviations on capability columns, oriented so that higher is better; substring matching as the headline non-refusal metric with the safety classifier as a second column; a Wilcoxon-free, plain-delta presentation with no significance testing on the method gaps; a hand-coded audit of the judge on a stratified sample of 50 disagreement cases, reweighted to bucket sizes, producing a bracketed interval for the true refusal rate; and the cross-site grid as the control for the selection procedure. Spearman rank correlation and top-3 agreement between the two methods' selection scores.

## What they found

- **Counterfactual flipping matches directional ablation at suppressing refusal**, on the headline and the safety-classifier column, in most cells: on Llama-3 the flip shows +0.96 non-refusal against ablation's +0.95, and on Gemma +0.90 against +0.89.
- **Nullspace projection is the weak arm** and its weakness is model-specific: it reaches +0.70 on Gemma and +0.55 on Yi, but only between +0.01 and +0.03 on Llama-2 and +0.10 on Qwen — the authors read this as different models organising the refusal subspace differently.
- **The cross-site control is the paper's own correction of its own headline.** Flipping at full rank holds steady or improves at the other method's site on all five models, and Qwen's one *negative* cell (−0.13) turns to +0.70 there — so that negative was an artefact of the selection score, and the authors say so. Directional ablation goes the other way and collapses at the INLP site on three of five models (Qwen +0.67 to +0.03, Llama-2 +0.46 to +0.06, Llama-3 +0.95 to +0.20). The exception to flipping's robustness is one cell — k at the 80%-accuracy threshold on Llama-3, +0.96 to +0.12.
- **Activation addition's harmless-side win is partly a measuring artefact**: it reaches +0.98 (Gemma) and +1.00 (Llama-3) refusal injection on harmless prompts, the best in the table, while being the worst on perplexity, and the judge's looping flag shows many of those "refusals" are repeated degenerate output.
- **Erasing fewer directions is cheaper**: truncating the row space to the leading classifiers at the 80% threshold keeps harmful-side suppression within noise of full erasure on Gemma, Yi, Llama-2 and Llama-3 while holding perplexity within 0.05 of baseline on all five — a tunable effect-versus-capability knob.
- **Geometry separates the two operators**: actual ablation pushes both classes onto the harmless side; addition and flipping swap them; nullspace projection drops them *between* the clusters. The authors read the middle landing as tentative evidence that the model represents the absence of the concept differently from its opposite, and immediately restrict that claim to "preliminary" because refusal is a case where absence and opposite are semantically blurred.
- **The measurement audit is a result in its own right**: substring matching and the judge disagree on 3,482 of 21,995 completions (15.83%, interval 15.35 to 16.32), the disagreement is one-sided (2,908 the substring rule misses against 574 it over-flags), it is concentrated on harmful prompts (25.19% against 4.59%), and it grows with intervention aggressiveness from 7.2% at baseline to 41.9% under the most aggressive flip. The hand audit attributes roughly 1,794 of the judge's refusals (61%) to judge over-fire on taboo topics, inflating its refusal rate by about 15 points on harmful prompts.

## Limits and open problems they name

Everything is called preliminary by the authors, repeatedly. One concept, five small models, one site per cell, and selection scores for both INLP arms computed from nullspace projection only — so the flipping arm is being selected by the wrong score, which they flag as future work. No significance testing accompanies the method gaps, and the table's standard errors are non-trivial next to several of them. The absence-versus-opposite reading is explicitly not established by this data. LEACE, larger models in the same families, a stronger judge, and concepts with a genuine neutral state are all named as needed. The capability side is thin: perplexity and two multiple-choice benchmarks, the first of which moves by zero in nearly every cell.

## What we take from it

- **A method comparison must carry its selection procedure as a factor.** The *instrument* to copy: rerun each method at the site chosen by its competitor and report the pair. Where the ranking survives the swap the difference is in the intervention; where it collapses the difference was in the chooser. This is the readout-side form of the ablation-side lesson in `arxiv-2606.00926`: an intervention that never reaches its own premise reports the premise, not the state. Mechanical rule for us: **every claim our white-box model makes that feature A matters more than feature B at a chosen site is repeated with the sites exchanged, and a gap that only exists at each method's own chosen site is labelled a selection artefact.**
- **The chosen site is itself an estimator, and it is tuned on the outcome.** The composite score that picks the layer rewards suppression, induction and low output divergence — the very quantities later reported as the result. Any of our probes whose layer, position, alpha or threshold is fixed by maximising the downstream metric inherits this; reporting the cross-site value is the cheapest correction available, and it costs one extra forward pass grid, not a new model.
- **A binary behavioural label is an under-specified summary and its errors are not symmetric.** Move the judge's disagreement with the lexical rule is 15.83% overall and 41.9% in the most aggressive condition — exactly where the comparison is decided — and the audit shows the asymmetry is one-directional and large. This is the folder's recurring nuisance-baseline rule in its sharpest form (cf. `arxiv-2606.13870`): **report the outcome under two measurement protocols with different failure directions and bracket the true rate between them, rather than picking the flattering one.** The paper's own bracket (substring estimate plus 0.05 below, judge estimate minus 0.10 above) is the format to copy, including the refusal to claim the midpoint.
- **Redundancy across a subspace is not the same claim as necessity of a direction.** The truncation result — most of the effect survives when the row space is cut back to the directions a weak classifier can still separate — is another instance of the "fragile to this coordinate, not dependent on this feature" discipline (`arxiv-2606.00926`), and it is a cheap, tunable knob our model can borrow as a capability-cost dial.
- **Warning to carry:** the geometric panel is drawn with its axes pinned to the class-centroid direction and each cell in its own coordinate system, so the picture is a story about *relative* placement and never a magnitude. The target-fit ratio computed in the projected space and the same ratio computed raw differ visibly (the raw version compresses the gaps), which is a model of how a visualisation choice can manufacture a finding.

## Tags

unit: direction
form: steering
method: ablation | steering | other
eval: ablation | downstream-task | human-rating
goal: control/editing | safety/auditing | evaluation-of-methods
