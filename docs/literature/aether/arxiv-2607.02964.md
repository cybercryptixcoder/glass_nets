# Individual Parameters in Weight-Sparse Transformers Appear Interpretable

- **Key:** arxiv-2607.02964
- **Authors / organisation:** Arnau Marin-Llobet, Stefan Heimersheim; Harvard University (SEAS)
- **Date:** 2026-07
- **Link:** https://arxiv.org/abs/2607.02964
- **Kind:** paper
- **Read from:** full text (arXiv HTML v1, all 61,633 chars of the extracted page read end to end in three slices: abstract + §1 + §2, §3 methods + §3.1–§3.5, and §4 results + §5 case studies + §6 conclusion/limitations/future work + §7 + the whole appendix A.1–A.8 including Algorithm 1, Tables 1–3, and every figure caption; figures not inspected as images, so all numbers are table cells, explicit sentences, or figure captions)
- **Relevance to us:** 3 (central)

## What they wanted, and why

Circuit-finding explains a *set* of weights with respect to one behaviour on one sub-distribution, and the same weight can enter many circuits with a different contribution to each. The authors ask the complementary question at the smallest possible unit: can a single scalar weight be understood **globally**, across the full training distribution, without first choosing a behaviour? Their specific choice of what "understood" means is narrow and testable — an account of *when the weight matters*, i.e. the inputs on which ablating it changes the model's predictions. Weight-sparse transformers are the natural testbed, because most weights are exactly zero and therefore trivially interpretable, leaving the interesting question of whether the surviving nonzeros are too. The larger goal is a per-parameter interpretability pipeline that can be pointed at frontier models once an interpretable basis for dense parameters exists.

## What they studied

Four transformer models, all analysed at the level of **individual weights** (single scalar entries in MLP and attention matrices, sampled uniformly across MLP layers): the weight-sparse Python-code transformer of Gao et al. (~99% weight sparsity), the SimpleStories sparse (~98%) and dense pair of Drori (2026) — same architecture, one sparse one not — and pretrained Pythia-70m as a dense out-of-family control. Evaluation corpora were matched to each model's training distribution (CodeParrot for the code model, SimpleStories for the Drori pair, The Pile for Pythia), with an in- vs out-of-distribution cross-entropy check (2.34 vs 3.70/4.50 for the code model) used as evidence that the proxies are close enough.

## How they did it

For each sampled nonzero weight they ablate it to zero, run the original and ablated model over the corpus, and record the per-position KL divergence between the two predictive distributions together with the corpus-level cross-entropy gap ΔCE. The positions carrying the top 90% of cumulative KL are kept, each with ±8 tokens of context, and an LLM (Gemini 3 Flash by default) is asked for candidate Python predicates `f(tokens, pos) -> bool` plus a one-line description; best-of-N is taken with N = 100 after a sweep showing saturation near that budget. Scoring is a **conditional-zero ablation**: ablate the weight everywhere, restore its contribution only where `f` fires (the recovery arm), and symmetrically restore it only where `f` does not fire (the inverse arm). Recovery and inverse are expressed as fractions of ΔCE accounted for, a coverage gate rejects predicates firing on nothing or on more than half the positions, and the score is the minimum of recovery and one-minus-inverse, thresholded at 0.75. Predicates are generated on one slice of text and re-scored on twelve non-overlapping slices; ten held-out slices per weight give a mean and a standard error, and the "robust rate" credits a weight only when its held-out mean clears the bar and sits outside a two-sigma not-interpretable cone. Controls are model-family controls (same architecture, sparsity on vs off; pretrained dense) plus ablations of the pipeline's own knobs: threshold sweep, candidate budget, four different auto-interp LLMs.

## What counts as an explanation here

A short executable predicate over token context — a program, not a description — attached to one weight and claiming "this weight is active at positions satisfying this condition". It is deliberately partial: it says *when*, not *what computation*. The paper is explicit that a correct when-predicate is informative but not a full account of a weight's role, and that the reported rates are a lower bound set by the expressiveness of the predicate language.

## How they judged success

Three nested metrics on the same per-weight score: in-sample pass rate (Interp_A, predicate scored on the text it was written from), held-out pass rate (Interp_B, the same predicate re-scored on unseen slices), and the robust rate (mean over ten held-out slices, required to clear the bar outside a ±2σ null cone). Reliability — the fraction of cross-weight score variance that survives redrawing the corpus — is reported separately. Robustness checks are threshold sweeps, candidate-budget sweeps from N=1 to 150, four auto-interp LLMs, and a closed-form random-predicate baseline whose score equals its firing coverage and is therefore capped at 0.5 by the coverage gate.

## What they found

Pass rates fall steeply from in-sample to held-out to robust, and the ordering never changes: Gao-sparse 37.3 / 35.3 / 15.0%, Drori-sparse 21.4 / 17.2 / 9.6%, Drori-dense 13.2 / 9.0 / 1.5%, Pythia-70m 12.0 / 5.0 / 0.4% (Table 1, ±SEM). The interpretable population sits where causal impact is real: in the high-|ΔCE| bin (≥1e-4) roughly a quarter of sparse-model weights are robustly interpretable, while dense models barely populate that bin at all. Reliability is ~85–90% on the sparse models against 47% on Pythia, so dense-model interpretability is largely a property of the corpus slice rather than the weight — and correcting for it *widens* the gap instead of narrowing it, because dense-model scores cluster near the random-chance mean of 0.5 and regress toward it off-corpus. Predicates are short (median at most four lines, two on the Drori pair) and locally scoped (single-token lookups, neighbour tests, occasional two-to-three-token context); none maintain state across a prompt. Three case studies: a digit-detecting Gao neuron whose nine input weights split into per-digit-count gates; a string-closing weight that scores −0.98 under a token-local predicate language and +0.83 once an eight-token look-back is allowed; and a Drori-sparse speech-verb neuron whose 86 nonzero input weights decompose into five functional roles (speech-verb activators, mental-verb suppressors, a modal suppressor, punctuation gates, specific-token gates) that add and cancel into the single scalar an activation-level analysis would label "speech-verb detector".

## Limits and open problems they name

When-predicates are not what-computes-predicates. The predicate language (Python over token context) cannot express semantic features, multi-token spans, or prompt-spanning state, so the rates are lower bounds and some weights are filed uninterpretable for want of a language, not for want of function. Weights below the cross-entropy noise floor (|ΔCE| < 1e-6) cannot be scored at all and are excluded. Absolute rates depend heavily on the pipeline — different auto-interp LLMs move them by up to ~30 percentage points, with stronger LLMs higher — even though the cross-model ordering survives every knob. Held-out evaluation stays inside the pretraining distribution, so nothing is said about distribution shift. Models are small, and the paper's own future work asks whether per-weight interpretability is a property of these models or of sparsity training.

## What we take from it

**A measure, and the cheapest certificate of its kind in this folder.** Score a per-unit claim with a matched two-armed conditional ablation: restore the unit only where the claim says it acts (recovery) and only where the claim says it does not (inverse), divide both by the total ablation effect, and take the minimum. That structure is what makes a read-out falsifiable in both directions and it answers the folder's standing complaint that "the description is wrong" versus "the description is missing a qualifier" are never separated — recovery catches the first, inverse catches over-broad claims, and the coverage cap blocks the degenerate predicate that fires everywhere. For a predictive white-box model this maps almost one-to-one onto our own output: our prediction is a predicate over inputs ("this unit carries the computation here"), so recovery/inverse on held-out text is a ready-made eval that needs no ground-truth circuit.

**A control we owe and mostly do not run: the same method on a matched dense model.** The dense same-architecture twin and the pretrained dense control are what convert a rate into a claim about *sparsity* rather than about *the pipeline being able to fit predicates to anything*. Read against the folder's other matched-object baselines (random-init models, random directions, same-size random circuits), this is the cheapest version yet, and it is the one that catches a method that works on any model equally well.

**A warning about the reporting ladder.** In-sample → held-out → resampled is 37/35/15 and 13/9/1.5: the ordering is stable throughout, but the *level* more than halves at each step, and on the dense pretrained model the last step erases the result. Any rate we publish should be quoted at the strictest rung we ran, with the reliability number beside it, because that is the rung at which the corpus stops being able to manufacture interpretability.

**A framing for the activation-versus-weight gap.** The speech-verb neuron shows a single activation-level label hiding five weight-level roles that add and cancel — the concrete case for the folder's habit of not treating a neuron label as a mechanism, and a cheap diagnostic: a neuron whose recovered roles include suppressors of near-neighbour classes is evidence that the activation summary is a residue, not a computation.

## Tags

unit: weights
form: program
method: weight-analysis, probing, ablation, agent
eval: ablation, prediction/simulation, faithfulness, human-rating
goal: understanding, evaluation-of-methods
