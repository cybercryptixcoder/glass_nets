# Pattern Selectivity is Not Task-Causal Structure: A Cross-Architecture Mechanistic Study of Composed-Task Circuits in 1B-Class Language Models

- **Key:** arxiv-2606.05378
- **Authors / organisation:** Yongzhong Xu (independent; code at github.com/skydancerosel/spectral-probe-circuits)
- **Date:** 2026-06-03
- **Link:** https://arxiv.org/abs/2606.05378
- **Kind:** paper
- **Read from:** full text (v1 HTML, 72,685 chars / 1,294 lines after a maths-recovery pass; abstract, §1 Introduction, §2 Related Work, §3 Methodology Recap, §4 Set-up, §5-§9 (the four task sections, with **Tables 2-11 read cell by cell**), §10 the full 4x3 grid and the five-category taxonomy, §11 the MoE hypothesis with all four predictions, §12 the BOS-fraction table, §13 Discussion, §14 Limitations in full, §15 Conclusion; figures read from captions and reporting prose only)
- **Relevance to us:** 3 (central)

## What they wanted, and why

The field has a canonical story: in GPT-2 small, indirect-object identification (IOI) is implemented by name-mover heads suppressing name-duplication, with S-Inhibition heads writing the subject out of contention, and an induction/prev-token backbone. The author asks whether that delegation — take the pattern-screen step, run it on a different model, get the same circuit back — survives contact with other training pipelines. The larger goal is a methodology claim rather than a mechanistic one: a circuit-discovery recipe is only useful if its *products* transfer, and the paper wants to know what of it transfers at the scale where the tasks are actually solved (1B-active parameters, not 124M). A secondary goal is to name the *kinds of outcome* a screen-and-ablate study can produce, since the literature mostly reports "circuit found" or "no circuit found."

## What they studied

Three 1B-active-parameter models deliberately crossed over architecture and corpus: Pythia 1B (GPT-NeoX, dense, The Pile), OLMo 1B (Llama-style, dense, DCLM), and OLMoE 1B-7B (Llama-style mixture-of-experts, 64 experts with top-8 routing, DCLM). OLMo and OLMoE share corpus and architecture family and differ only in the MoE layer, which is what makes the dense-versus-MoE contrast interpretable. Unit of analysis: attention heads at (layer, head) granularity, screened by attention pattern and then causally verified. Four composed tasks: IOI, greater-than, successor sequences, and a two-pair variable-binding task. All four prompt batches were built so every token is single-token in all three tokenizers, which removes a confound the author treats as a standing hazard.

## How they did it

The recipe has three steps. First, a spectral signal per (layer, head, checkpoint): the participation ratio of each head's attention-output singular-value spectrum, integrated over training. This is a general indicator that a head is doing content-dependent work, and the paper is explicit that in these models it mostly selects generic early-layer heads rather than task heads. Second, a task-pattern screen per task — four candidate screens for IOI (prev-token, induction, name-mover, S-Inhibition), each with a stated selectivity threshold and formula; a query-to-start-decade attention screen for greater-than; a self-attention-at-position-T-1 screen for successor; a query-to-binding-value screen for variable binding. Each screen returns a small candidate set, typically three to thirteen heads. Third, group mean ablation of the candidates, measuring delta top-1, delta logit-difference, and frac(correct greater than distractor), against a matched-random control: the same number of heads drawn from the same layers, no overlap with the picks, ten independent seeds, reported as mean plus-or-minus standard deviation.

Three refinements sit on top. Individual head ablations are run wherever the group result is ambiguous, which is how supporters, interferers, and nulls are separated. A BOS-class filter (drop heads whose best classification is first-token attention-sink) is tested as a repair for confounded screens. And the whole thing is repeated per task per model, giving twelve (task, model) cells.

## What counts as an explanation here

A circuit: a named set of attention heads at named layers, characterised by the attention pattern that selects it and by the causal effect of removing it. The explanation is partial by construction — head-level, no SAE features, no residual-stream decomposition, and the author says so in the limitations, naming per-model feature dictionaries as the work this would need to be deeper. The scale of the claim is deliberately not "this is how the model does it" but "this head set is a primary / secondary / correlate / interferer / null cause of this behaviour in this model," and the paper's propagating object is the *taxonomy plus recipe*, not any one circuit.

## How they judged success

Delta top-1 in percentage points under mean ablation, delta logit-difference, and frac(correct greater than distractor), each against a matched-random null over ten seeds in the same layers with equal head count. Specificity is the ratio of the screen's absolute delta to the matched-random mean absolute delta, and the paper adds an explicit caveat that this ratio is not meaningful when the control itself is destructive. Baselines are the other screens applied to the same task and model (a cross-screen comparison rather than a cross-method one), plus a different-task circuit (induction, prev-token) ablated on the same prompts as a negative control. Individual head ablations serve as the diagnostic that the group number is or is not a sum of real contributors.

## What they found

**No two of the twelve cells share a primary screen, and the equivalence is behavioural, not mechanistic.** On IOI all three 1B models clear the same bar — the indirect object outranks the subject on essentially every prompt — yet the primary screen is prev-token in Pythia (delta top-1 of 82pp), S-Inhibition in OLMo (32pp, against a matched-random that goes the *other* way), and name-mover in OLMoE (18pp at roughly 34x specificity). GPT-2 small is included as the negative anchor: 13% top-1, so the canonical circuit's home model cannot solve the task in this panel's terms.

**Two pathways at different depths can each carry the same behaviour, and the late one alone inverts the preference.** In Pythia the late-layer name-mover and S-Inhibition sets are individually correlates and partial causes, but ablated as a union they take top-1 to 37.6% and flip the logit-difference from +4.00 to -0.47 — the model actively prefers the subject. So the early prev-token route is what the model normally uses, and the late route is load-bearing enough to drive the answer by itself.

**The MoE model splits winner from margin.** Ablating OLMoE's S-Inhibition heads leaves the indirect object as argmax and crushes the margin from +3.95 to +0.95; the name-mover set owns the argmax. The paper reports the same argmax-versus-margin dissociation a second time on OLMo greater-than, and turns it into a methodological claim: top-1 is the wrong primary metric.

**The MoE model puts the positional substrate underneath three of four tasks.** OLMoE's prev-token circuit ablation costs more than the task-specific screen on greater-than (6pp versus 5pp), successor (57.6pp versus 1.7pp), and variable binding (37.6pp versus 9.2pp), while doing nothing in either dense model. IOI is the exception, and the paper argues consistently that IOI's structure is itself final-position name-copying, so its screen is already structural. This is offered as a falsifiable cross-MoE hypothesis with four stated predictions, and the limitations concede the base is three positives and one exception.

**A screen can find heads that help the model.** Pythia's variable-binding screen returns five heads whose ablation *raises* top-1 by 37.4pp, and individual ablations make every one of them an interferer. Four of the five are first-token attention sinks; removing them removes a competing signal. Filtering the screen to non-BOS heads recovers a real circuit (16.2pp drop), but the same filter applied to OLMo removes the actual multi-role variable-binding circuit (post-filter delta of -0.8pp) because 78% of OLMo's heads are BOS-class and the non-BOS pool simply does not contain it. The paper's conclusion is that the filter is model-specific and the generalizable diagnostic is the individual ablation.

**Specificity ratios collapse when the screen concentrates in layer zero.** OLMo's successor screen drops top-1 by 80.5pp, but five random heads in the same layer drop it by 57.5 plus-or-minus 18.1pp, so the honest ratio is about 1.4x rather than the 2x a single-seed comparison suggested; Pythia's successor control ranges over 74 percentage points. Where heads spread across five layers — the greater-than screens — the control is inert at 0.0 plus-or-minus 0.0pp and the ratio is effectively unbounded.

## Limits and open problems they name

Twelve cells is a demonstration that the taxonomy is non-empty, not a generalisation study; the interferer category is instantiated with confidence exactly once; each natural-text model is a single released checkpoint, so whether the *primary screen* is seed-stable is untestable without re-pretraining (the companion paper's six-seed small model showed the specific induction heads move across seeds); the MoE hypothesis rests on three points; and the paper explicitly does not measure *why* the screens disagree — it names per-model SAE feature decomposition as the missing work. Metrics are one-dimensional (top-1 and logit-difference), and per-example logit attribution and distributional divergence are listed as absent. The BOS-class filter's failure to port is reported as a finding, not hidden.

## What we take from it

A **measure** and a **warning**. The measure is the matched-random control done properly: same layers, equal head count, ten seeds, reported as mean and standard deviation, with a stated floor problem when the screen concentrates in one layer — and the associated rule that a specificity ratio is meaningless where the control is itself destructive. The warning is the sharpest thing in the paper for us: **a component that shows the right internal pattern and passes its own screen can be causally inert, actively harmful, or merely a correlate of something computed earlier, and only a signed per-component ablation says which.** The correlateness diagnosis is the mechanism we care about most — the Pythia case shows the answer already sitting in the residual stream by the layers where the "right" attention pattern lives, which is precisely the failure mode of reading structure off weights or attention and calling it computation. For a predictive white-box model this is the argument for making every predicted component carry a predicted *sign* and *magnitude* under removal, with a stated null, rather than a predicted co-occurrence.

## Tags

unit: head
form: circuit
method: ablation, probing
eval: ablation, benchmark
goal: understanding, evaluation-of-methods
