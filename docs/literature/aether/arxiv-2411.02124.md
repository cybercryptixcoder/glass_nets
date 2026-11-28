# Adaptive Sparse Allocation with Mutual Choice & Feature Choice Sparse Autoencoders

- **Key:** arxiv-2411.02124
- **Authors / organisation:** Kola Ayonrinde (MATS; compute from EleutherAI)
- **Date:** 2024-11-04 (v1), 2024-11-07 (v2)
- **Link:** https://arxiv.org/abs/2411.02124
- **Kind:** paper
- **Read from:** full text (abstract, SS1-SS7, references, Appendices A-G; license CC BY 4.0; 10 pages, 18 with appendices, 7 figures)
- **Relevance to us:** 3 (central)

## What they wanted, and why

A TopK SAE gives every token the same active budget of k features. The authors argue this
is a design mistake rather than a tuning detail: some tokens are simply harder to
reconstruct, and a fixed k spends as much dictionary on a repetitive, easily described
token as on a rare or complex one. Their target is a sparsifying activation that keeps
TopK's advantages (no shrinkage, no conflict between a sparsity penalty and the
reconstruction loss, easy width comparison, a usable progressive code) while letting the
number of active features vary per token. A second, separable complaint is that wide
dictionaries fill up with features that never fire, and with features that fire rarely
enough to fall out of the gradient and never recover. The paper's aim is not a better
feature description but a better allocation rule: the encoder decides how the token-feature
matching budget is spent, and they want the SAE to be able to choose fewer features for
easy tokens and more for hard ones without paying compute for it. Behind both complaints
sits the stated belief that dead and dormant capacity is what blocks scaling dictionaries to
frontier-model sizes.

## What they studied

Models: GPT-2 small, layer-6 residual stream activations, context length 64, mean-subtracted
and unit-L2-normalised, shuffled for training. Dictionary widths 4x to 32x the 768
activations, at 0.8% feature sparsity. They also report dead-feature rates they attribute to
widths of 6k, 16m and 34m latents. Unit of analysis is the **token-feature match**: the pair
(token position, feature) is the thing being allocated, and the objects compared are
sparsifying activation functions, plus one auxiliary loss and one inference-time
distributional hyperparameter (the Zipf exponent of the per-feature token allowance).

## How they did it

The framing is the contribution. Take the batch-by-feature pre-activation affinity matrix
Z', and require at most M token-feature matches. Choosing which M edges of the bipartite
graph to keep is a resource allocation problem that has no exact solution without knowing
how each edge contributes to reconstruction, so they assume the contribution is monotone in
affinity magnitude and keep the M largest entries. TopK is then the special case with a
per-row cap of k; Feature Choice (FC) is the case with a per-column cap, each feature
matching at most m tokens, solved by a TopK over the batch dimension; Mutual Choice (MC) is
the unconstrained case, a TopK over the whole matrix. They draw the analogy to Mixture-of-
Experts routing (token-choice versus expert-choice) and rename TopK SAEs "Token Choice" to
unify notation. FC needs an allowance per feature, and they take it from data: existing
open-source SAEs have feature densities close to a rank-1 Zipf curve (reported R2 of 0.982
over the full distribution and 0.996 over features ranked 100 to 20,000), so each feature's
token allowance is drawn from a truncated Zipf with exponent alpha and bias beta (about 1.0
and 6.8 for GPT-2-width residual activations, exponent fixed at 1 in the experiments).
They measure that curve by running 100 million tokens of inference through a trained SAE of
the same width and counting activations per feature. Their third piece is a definition of a
**dying** feature, distinct from Bricken's dead: among the 25% least prevalent features, a
feature is dying if its prevalence is under 60% of what its rank predicts from the fitted
Zipf curve. The loss they add, aux_zipf, is Gao's aux_k with the dead-feature set replaced by
the dying set, so the residual is reconstructed from the top auxiliary features among those
that are dying. Training procedure: first MC training with both aux_zipf and aux_k, then
optional fine-tuning under the FC activation with no auxiliary terms. Baselines are ReLU
SAEs and TopK SAEs; Gated SAEs are not run, on the argument that Gao found TopK matching or
beating them at lower compute. Evaluation is L0, MSE, the change in the model's own
cross-entropy loss reported as a normalised percentage of loss recovered, dead-feature
share, and automated interpretability in the style of Juang et al. No resampling is used
except where stated.

## What counts as an explanation here

Nothing descriptive. The output is an architecture and a loss: a matching rule, a
distributional prior on the allowance, and an auxiliary objective. "Explanation" in this
paper means a dictionary that is complete enough to be worth reading - every feature firing,
no capacity abandoned - and it is meant to be predictive in a narrow, mechanical sense: the
matching rule plus the Zipf allowance should predict which tokens get more features (hard
ones) and which fewer (the BOS token), and the auxiliary loss should predict that the
dictionary keeps its low-prevalence tail alive. There is no claim that FC or MC features are
more interpretable, and the one interpretability number reported is AutoInterp.

## How they judged success

Four measures: reconstruction (MSE, and normalised loss recovered, which is the measure they
rank methods on because it is the only one tied to what the model does downstream), sparsity
(L0, with the expectation of k fixed rather than k itself), dead-feature percentage against
the published rates of three prior SAEs, and AutoInterp scores. Comparisons are against ReLU
and TopK baselines at matched sparsity. The one controlled ablation of a design choice is
Zipf-versus-uniform allowance, and the one internal control is that the largest runs apply
additional decoder-correlation losses (nfm, nfm_inf) with small weight to reduce feature
splitting and a failure mode they call dictionary collapse.

## What they found

FC improves loss recovered over TopK at equivalent sparsity across the widths swept, which
they describe as a Pareto improvement, and MC with aux_zipf plus FC fine-tuning is where the
best loss-recovered numbers appear. The dead-feature table is the headline: FC is reported at
0% dead at 6k, 16m and 34m latent widths, against 9.0% for a standard ReLU SAE at 6k, over
90.0% for standard at 16m, 64.7% for a resampled 34m SAE, and 7.0% for TopK at 16m; the text
also gives Gao's unmitigated 16m figure as 90%, which does not match the 7.0% TopK row and is
not reconciled. The Zipf allowance is worth more than the loss they stress: uniform
allocation loses more than 10% of loss recovered against the Zipf one, which they read as
evidence that features are genuinely unequally prevalent rather than that any smoothing
would do. Adaptive computation shows up as a bimodal feature-count distribution, with the
BOS token in the lower mode at roughly 10 features where they say a single feature should
suffice. Feature densities drift toward Zipf over training, with the divergence point moving
from about 3k to about 24k in rank. They note Bussmann's concurrent BatchTopK is closely
analogous to MC but does not address underutilisation.

## Limits and open problems they name

The Monotonic Importance Heuristic is the load-bearing assumption and they say so: importance
is treated as monotone in activation magnitude, which means a small activation that is
nevertheless decisive in context cannot be captured. Filtering is applied per feature in
Jump-ReLU and Gated SAEs only as a *weak* form of the same prior, which they concede does not
test it. One model, one layer, one modality, one token budget of 10,000 steps, no extensive
hyperparameter sweep, no agreement on SAE metrics, and no test of whether FC's dead-feature
guarantee survives on continuous modalities. Scaling laws for FC are named as future work.
Separately, they argue the Monotonic Importance Heuristic may be an inductive bias whose
benefit decays with scale, citing Sutton's bitter-lesson argument - which is a claim against
their own framing that they leave unexamined.

## What we take from it

**The accounting rule this paper supplies.** A sparsity constraint is a budget, and there are
three places the budget can be enforced: per token (TopK), per feature (FC), or nowhere (MC =
global). Which one is imposed silently decides which failure mode you get - per-token caps
starve hard tokens and abandon features, per-feature caps starve frequent features and
guarantee full coverage. For us, the rule is that **every dictionary or probe inventory we
build states where its budget is enforced and what observable that choice should starve**, and
we measure the starvation on both axes (per-token reconstruction dispersion, per-feature
usage share) rather than reporting a single aggregate L0. This paper is a clean instance of
why: it holds L0 to within a small margin across all three schemes at 0.8% sparsity and the
dead-feature rate still moves from over 90% to 0.

**Coverage is not the same as allocation.** 0% dead with a Zipf allowance is structurally
guaranteed, not discovered - each feature matches at most m tokens, so unless a batch has
fewer than m tokens no feature can be starved. That makes the headline table a statement
about the constraint, not about the data, and it is the same gap run 951 found in a
different instrument (a repair that sets a dead count to 4 while the abandoned subspace's
reconstruction barely moves). Our rule: **when a method's advertised coverage follows
analytically from its constraint, say so in the same sentence as the number, and report what
the constraint does to the tail on the other axis** - here, that the lowest-prevalence
features get the smallest allowance by construction, which is why aux_zipf had to exist at
all. The paper does not invert this and so does not report the cost.

**Reciprocal definitions of neglect.** "Dead" (never fires over a horizon) and "dying" (fires
less than a rank-expected rate) are two different relations to the same fitted distribution,
and the paper shows the second is where the gradient goes quiet: a feature below its Zipf
expectation receives fewer updates, which makes it less useful, which pushes the expectation
further out, and the loop is what aux_zipf breaks. For us: **a unit counts as neglected only
against an expected-usage curve, and the curve has to be fitted and reported** - a raw
activity threshold cannot distinguish a rarely-used feature from an unusable one, and the
25%-tail and 60%-of-expected cut-offs here are operational, not derived from anything.

**The empirical prior is measured, not assumed.** The whole FC scheme depends on alpha and
beta, and they get them from 100 million tokens of inference through an existing SAE of the
same width, then show the curve is stable enough to reuse across widths. That is the right
shape for us: **a structural hyperparameter of our instrument is estimated from a long
instrumented pass over the model and then frozen and reported, rather than tuned per run** -
which is also the honest objection, since a Zipf prior fitted on one model at one layer has
no reason to hold elsewhere, and the paper does not test another model.

## Tags
unit: feature
form: replacement-model
method: SAE
eval: downstream-task
goal: understanding
