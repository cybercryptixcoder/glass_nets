# Slaying the Hydra: Interaction-Aware Circuit Discovery in Language Models

- **Key:** arxiv-2610.04017
- **Authors / organisation:** Sankaran Vaidyanathan, Rafal Urbaniak, Emily Bunnapradist, Michelangelo Naim, Daniel Waxman; Basis Research Institute, with University of Massachusetts Amherst and MIT.
- **Date:** 2026-10-02
- **Link:** https://arxiv.org/abs/2610.04017
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Circuit discovery scores one component at a time and then treats the resulting ranking as if the effects added up. They do not: a component can suppress its own backup so that patching the primary makes the backup take over (the self-repair that gives the paper its title), and joint effects on a set can sit above or below the sum of the singles. The authors want a component score that is computed *inside a context* rather than in isolation, at a cost that does not grow with the size of the candidate pool. The larger goal is the usual one for this literature — a compact set of components that both recovers a behaviour and can be selectively switched off — but the paper's real contribution is to show that those two goals need not be served by the same set.

## What they studied

Three Transformer language models used only through their attention heads: GPT-2 small (144 head writes as candidates), Qwen3-1.7B-Base (448) and Llama-3.1-8B (1024). Five tasks: indirect object identification, subject–verb agreement, two-digit addition, induction and entity tracking. The unit of analysis is the attention head, as both a candidate mediator and a candidate witness; MLPs and the embedding appear only in the MIB comparison. The outcome is the correct-answer logit minus the incorrect-answer logit at the final position.

## How they did it

The paper starts from the two standard patching estimands — necessity (patch the set to the alternative prompt's values) and sufficiency (restore the set to the factual values) — and adds a third element: a *witness*, a set of components downstream of the patched set that is pinned to its own start-prompt values so that it cannot compensate. Pinning one witness set restores the intuitive ranking: on GPT-2's IOI circuit the backup name movers score higher than the primaries when scored alone, and the primaries lead once witnesses are held. Since nobody knows in advance which witnesses to hold, they average the score over a distribution of (mediator set, witness set) pairs — the witness-integrated set effect. That expectation is still a forward pass per sampled set, so the second half of the method learns which sets to sample: one trainable inclusion logit per candidate, sets drawn by taking the top-k of logits plus Gumbel noise, a straight-through gradient on the hard selection, and Adam on the logits. Training descends a ladder of set sizes (large first, then narrower), a rung ends early once its top-k has settled and the cut margin is still widening, and the ranking is read off the recorded gradients summed over training with per-step normalisation and per-candidate clipping. A single run therefore yields the whole nested family of circuits, one per size.

## What counts as an explanation here

A ranking of attention heads, and from it a nested family of circuits — the top ℓ heads for every ℓ. It is partial by construction: the authors are explicit that they do not attempt the mechanisms' semantics and that a circuit here is a set with a measured causal footprint, not a described computation. Prediction is limited to what happens when that set (or its complement) is patched.

## How they judged success

Three outcome curves over circuit size, each summarised by an area-like score under a log-spaced weighting that rewards small circuits (a discounted-gain / early-recognition weighting, which they name the circuit recognition score). **Faithfulness**: patch everything outside the circuit and ask how much performance survives. **Kill**: patch the circuit itself and ask how much performance is lost — the necessity side, the quantity that faithfulness does not measure. **Net**: kill on the target task minus the mean non-negative kill on four other tasks, i.e. task specificity; the target effect is capped at one and control effects floored at zero. Baselines: exact per-head necessity scores (signed and magnitude-ranked), two attribution-patching variants with and without gradient integration, a matryoshka attribution method, and random orderings. Controls and checks: 95% intervals over 5–35 seeds per default cell, per-model defaults selected by median score across the five tasks so that one configuration is applied everywhere, a tuning sweep of 572 runs selected by expected improvement over a Gaussian process and cross-checked with boosted trees, and a separate evaluation on the public test split of a standard circuit-localisation benchmark over 12 model–task cells.

## What they found

On the 15 model–task cells the tuned configuration leads five of six summary columns (mean and worst cell, for each of the three metrics), with mean kill 0.67 against 0.64 for signed exact per-head necessity, mean faithfulness 0.61 against 0.59, and mean net 0.54 against 0.47. The per-model default configuration is roughly level with signed exact scoring and ahead of the gradient attribution methods. Worst-cell task specificity is where methods separate: several baselines have at least one cell with *negative* net, meaning removing a "circuit" damaged other tasks more than the target. The paper's most important single caveat is quantitative: the baseline with the best faithfulness on Qwen3 keeps heads that recur heavily across tasks (mean head overlap 0.28 against 0.12 for exact per-head scoring) and is therefore negatively specific on three of five tasks — high faithfulness and negative net are the signature of a set of general-purpose heads. Training cost on the largest model is 14–16 minutes against 97 for exact scoring, a 4–7× speedup, while on the two smaller models training is 2–8× *slower* than exact patching, which takes seconds to minutes. Ladder descent beats singleton training on all three metrics on all three models; witness pinning helps on Qwen3 and hurts on GPT-2 and Llama. On the benchmark split they beat the comparison method on kill and specificity but not on faithfulness, where their competitor's advantage comes largely from overshoot, i.e. recovered performance above the intact model's.

## Limits and open problems they name

Attention heads only in the main experiments; a single untuned run on the benchmark split; training only pays off once the pool is large; and the witness benefit is model-dependent, with the authors saying that a principled account of when witnesses help is still missing. They also concede that their own metric choice can reverse a verdict — two of the three standard summaries prefer a curve that recovers late and overshoots, while theirs prefers the curve that recovers early — so the honest reading is that they report all three side by side rather than that one summary is right.

## What we take from it

A method (contextualised scoring by pinning the compensating variables, then learning which sets to score), a measure (kill beside faithfulness, plus a specificity term that compares the same intervention against unrelated tasks), and a warning that is directly useful to us: **a component ranking is not a description of a mechanism unless the components are also shown to be specific to the behaviour.** The specificity control — apply your account's ablation to four other behaviours and report the collateral damage — is cheap, requires no new models, and catches the case where an account "explains" a task by naming the machinery that everything uses. Second takeaway: their cost argument is structural rather than empirical, and the same distinction applies to any probe suite we design — per-candidate exact measurement grows with the pool, sampled-set measurement grows with the budget.

## Tags

unit: head
form: direction
method: patching, attribution, ablation
eval: patching, ablation, faithfulness, downstream-task, benchmark
goal: understanding, control/editing
