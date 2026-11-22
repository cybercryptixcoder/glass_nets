# Capacity, Responsiveness and Alignment: What Makes a Latent Structure Actionable

- **Key:** arxiv-2610.06897
- **Authors / organisation:** Or Shafran, Mor Geva; Blavatnik School of Computer Science and AI, Tel Aviv University
- **Date:** 2026-09-28
- **Link:** https://arxiv.org/abs/2610.06897
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why
The paper starts from a premise the whole steering literature rests on and that keeps failing: if a concept
can be localised as a geometric object in activation space, intervening along that object should reliably
change what the model says. In practice localised directions often do almost nothing, or do something other
than what their label says. The authors ask what *makes a localised structure actionable* rather than merely
detectable. Their answer is to take the causal influence of a structure at a single activation, expand it to
first order through the model's own Jacobian and softmax, and factor the resulting scalar into three
multiplicative pieces that are individually measurable. The larger goal is to replace "does this direction
exist and decode the concept" with an account of why the same direction is strong in one context and inert or
inverted in another — i.e. a predictive statement about intervention, computed from the network's local
derivatives rather than fitted from behaviour.

## What they studied
Four instruction-tuned open models: Gemma-2-9B, Llama-3.1-8B, Qwen-2.5-7B and Mistral-7B. Fifty concepts drawn
from a list used in prior steering work, with two datasets per concept built by prompting gpt-4.1-mini:
concept-positive completions and neutral completions, over 304 concept-independent prefixes (15,200 positive
and 304 neutral examples), split by prefix 80/10/10 so no prefix appears in two splits. The unit of analysis is a
**direction** at a residual-stream activation — a latent structure produced either by difference-in-means or by
a bias-free logistic linear probe, trained separately per concept, per model, per layer. Layers at roughly 20%,
40%, 60%, 80% and 95% of depth. The causal object being measured is not the direction's identity but the
scalar *change in concept expression* produced by pushing the activation along it, at a specific context and
position. Concept expression is the probability mass on a concept's verbalising tokens, encoded as a 0/1 output
vector whose entries are tokens that occur in the concept's positive completions and never in the neutral ones.

## How they did it
Step by step: (1) define concept expression as the dot product of the 0/1 concept vector with the output
distribution; (2) for an intervention vector **v** added to activation **h**, linearise the effect on the logits
through the Jacobian of the remaining computation and push it through the softmax Jacobian, giving a first-order
change in expression as a scalar; (3) rewrite that scalar, using the covariance–correlation identity, as the
product of three terms — a standard deviation of the logit change (capacity), a standard deviation of the concept
vector (responsiveness), and a correlation between the two (alignment); (4) derive the capacity term's closed form
as a quadratic form in a "sensitivity matrix" built from the Jacobian and the softmax Fisher matrix, whose
eigen-spectrum is the useful subspace; (5) predict observed KL for interventions as half alpha-squared times that
quadratic form and check against measured KL; (6) measure each factor empirically and test what it predicts;
(7) restrict linear probes to the top-*r* eigenspace of the averaged sensitivity matrix and evaluate steering.
Jacobian-vector and vector-Jacobian products are computed by automatic differentiation on demand (the full
Jacobian is vocabulary-sized and never materialised); leading eigenvectors of the averaged matrix are estimated by
randomised subspace iteration, with residuals and orthogonality checked against full computations. The
locally-optimal direction is obtained by solving a ridge-regularised linear system in the same matrix with
conjugate gradient (at most 32 iterations, recomputed error checked every four). Sequence-level effects are summed
over up to eight future positions with teacher forcing, and the decomposition applies position by position.

## What counts as an explanation here
A **closed-form multiplicative decomposition** of an intervention's first-order effect into three named scalars,
each attached to a specific context and activation, plus a low-rank subspace of directions the downstream
computation is sensitive to. It is not a dictionary and not a circuit; it is a locally linear model of "what will
this push do here", intended to be predictive rather than complete. Its predictive claims are explicit: the
capacity term gives a quantitative forecast of intervention-induced KL, the product of the three terms is meant to
rank positions by steering efficacy, and the sensitivity matrix's leading eigenvectors are meant to define the
subspace a probe should live in. Context dependence is a substantive part of the explanation, not a caveat: the
authors argue causal efficacy is a property of the structure–context–remaining-computation triple, so the same
labelled direction is expected to be strong, weak or counterproductive depending on where it is applied.

## How they judged success
- **Predicted versus observed KL**: second-order prediction (half alpha-squared times the quadratic form) against
  measured KL for 64 isotropic unit directions across all neutral contexts, at 14 intervention strengths from 0.3
  to 128, at the current position and up to 16 future offsets, over layers.
- **Random-direction baselines**: median capacity of difference-in-means and probe directions compared with 16
  isotropic random directions, all unit-normalised, capacity summed over a nine-position horizon; and random
  isotropic directions carried through the responsiveness and alignment analyses as a control.
- **Fluency-gated evaluation**: an LLM judge scores fluency 0/1/2, normalised, at 11 intervention strengths; each
  support-statistic is reported up to the largest strength whose mean fluency stays above 0.8, 0.7 and 0.6, so
  "the intervention only redistributes within the baseline support" is a statement made inside the fluent regime
  and its degradation outside is shown.
- **Eight-bucket factor design**: 16,000 C4 contexts of 256 tokens, 50 concept vectors, interventions at seven
  source positions; samples are labelled high or low per factor by top/bottom quartile, with thresholds
  cross-fitted (five folds), and the observed effect is compared across the eight joint settings.
- **Transfer test for alignment**: a direction computed in a donor context is evaluated in a recipient context and
  its alignment is normalised by the recipient's own locally-preferred direction, with random directions as the
  floor; the self-context case is checked to return one within 0.002.
- **Steering**: 14 strengths swept, 50 generated tokens, 16 fixed neutral prefixes, harmonic mean of judged
  concept expression and fluency, strength chosen by five-fold cross-validation over prefixes so the reported
  score is out-of-fold. Baselines: unconstrained probe, difference-in-means, and a strong supervised rank-one
  ReFT baseline. Concept-detection accuracy is tracked so the steering gain is not bought with detectability.
- **Position-selection ablation**: intervene at the quarter of positions with highest predicted effect, the
  quarter with lowest, or a random quarter; 20 settings (4 models x 5 layers) reported.

## What they found
- **Capacity predicts magnitude.** Predicted KL tracks observed KL closely across intervention strengths, with
  directions outside the high-capacity subspace producing little effect; the tracking holds across future offsets
  and layers, with a slight worsening for Qwen. This is the paper's strongest validation of a factor as a
  *measurement*, because it is a forecast of an unmeasured quantity rather than a correlation with it.
- **Capacity is low-rank and context-dependent.** Median effective dimensionality (participation ratio of the
  sensitivity spectrum) is 24.8 for Gemma, 47.2 for Llama, 30.1 for Mistral and 23.3 for Qwen — 0.7%–1.2% of the
  residual stream — lowest in early layers (8–13 directions) and rising with depth (37–97).
- **Localisation methods differ in capacity.** Difference-in-means directions have 90.4% more capacity than random
  directions and 26.4% more than probes; probes have 37.6% more than random. So a structure can be perfectly
  decodable and sit mostly outside the subspace the rest of the network can use.
- **Responsiveness is a softmax artefact plus a context fact.** At fluency 0.8, the top 2.5% of baseline tokens
  receive about 94.5% of the intervention-induced probability increase; movement outside that support appears only
  at strengths of 10 and above, where fluency falls below 0.6, and the pattern holds for random directions too,
  which is what makes the restriction a property of the context rather than of the structure.
- **Alignment is context-specific and does not transfer.** Locally-optimal concept-promoting directions for the
  same concept are near-orthogonal across contexts in early layers and remain below 0.2 cosine in most models;
  transferring a donor direction into a recipient context retains only 10%–40% of the local alignment (below 0.1
  normalised in early layers).
- **Any single low factor bottlenecks the effect.** Low capacity reduces concept expression by 84%, low
  responsiveness by 95%, and low alignment can reverse it so the structure suppresses the concept it is labelled
  for; with all three low, promotion is effectively zero. Exactly-one-low cases are common, about 26%–31% of
  retained samples.
- **A worked position example.** In a two-attribute binding prompt, the final position has more than three times
  the capacity of an earlier token but far worse alignment (0.14 vs 0.88), so intervening seven positions earlier
  has a 3.2x larger first-order effect.
- **Causal probes.** Restricting a linear probe to the top-128 capacity subspace improves median steering by 118%
  (Gemma), 57% (Qwen), 23% (Mistral) and 17% (Llama), with concept-detection accuracy falling only from 93.8% to
  90.4%. The conclusion states the aggregate as 29% median improvement with a 3-point detection drop, and the
  appendix shows the gain comes overwhelmingly from concept scores, not fluency.
- **Position selection works over long generations.** Steering only the quarter of positions with highest predicted
  effect beats random selection in 19 of 20 settings, by 44.2% on average, while steering the lowest-predicted
  quarter loses 15.5% and underperforms random in 80% of settings.

## Limits and open problems they name
The paper's own scope statements are thin — there is no separate limitations section — but several constraints are
visible in the design. Everything is first-order: the decomposition is an approximation whose quality is checked
for KL but not for the signed concept-expression quantity under strong interventions, and the responsiveness
analysis is explicitly restricted to the fluency-preserving regime, so nothing is said about what happens to the
factorisation when the intervention is large. The concept definition is verbalised-token mass, so concepts without
a clean token set are out of scope. Fluency and concept scores come from a single LLM judge, and the steering
strength is cross-validated over 16 prefixes — a small sample — while several headline numbers are medians over
concepts whose per-concept spread is not printed in the main text. The capacity subspace is aggregated across
contexts (an average of sensitivity matrices per concept and layer) even though the paper's own alignment result
says the useful direction varies per context; the causal probe therefore uses a context-marginal subspace, which
the authors present as a practical compromise rather than as a solution. The paper is also silent on a control the
folder would want: what a probe restricted to a *random* rank-128 subspace scores, which would separate "capacity
geometry helps" from "any rank restriction regularises".

## What we take from it
- **A measure we should adopt nearly verbatim: forecast the intervention, then check the forecast.** The
  second-order KL prediction from a Jacobian-and-Fisher quadratic form is the cheapest predictive test of a
  white-box claim this folder has seen — it needs no training, no dictionary and no ground truth, and it fails
  loudly when the local linearisation does not describe the network. For our account of a black-box network, the
  equivalent is: predict the effect of a perturbation from the network's own derivatives, then measure it.
- **A framing that dissolves the "the feature is there but steering fails" puzzle:** causal efficacy is not
  intrinsic to the structure. Any claim in our account of the form "this direction controls X" must be indexed by
  context and position, and the account should say which of the three constraints is binding — is the direction
  outside the usable subspace, is the target concept already saturated or absent in this context, or is the
  direction mis-oriented here?
- **A concrete test for transferability of a direction**, which is directly reusable: evaluate a direction
  learned in one context inside another and normalise by that context's own locally-optimal direction. A number at
  or below the random-direction floor settles the question, and it is far cheaper than a steering sweep.
- **A prediction task for our own model**, which is the closest thing in the folder to "predict under
  intervention": the high/low predicted-effect partition beating random in 19 of 20 settings is a benchmark shape
  we can copy exactly — select a subset of positions by a predicted score, compare against random and against the
  anti-selected subset, and report all three.
- **A warning about aggregated subspaces.** The paper averages the sensitivity matrix across contexts to get one
  subspace per concept and layer, which contradicts its own finding that locally-optimal directions are
  near-orthogonal across contexts. Our account should report the context-local subspace and the aggregate
  side by side rather than assuming the aggregate is what a direction should be fit to.

## Tags
unit: direction
form: causal-abstraction
method: probing, patching, weight-analysis
eval: prediction/simulation, ablation, downstream-task
goal: control/editing, understanding
