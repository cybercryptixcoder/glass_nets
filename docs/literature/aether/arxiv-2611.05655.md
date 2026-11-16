# When Are Two SAE Latents the Same Feature?

- **Key:** arxiv-2611.05655
- **Authors / organisation:** Alexandra Suvorikova, Ilgam Latypov, Anton Trukhan, Yuriy Dorn; MSU Institute for Artificial Intelligence and two independent researchers
- **Date:** 2026-11-03
- **Link:** https://arxiv.org/abs/2611.05655
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

A dictionary is only useful as a description of a model if its units can be said to be *the same thing* across
two runs, two widths, or two checkpoints — and the field has no shared statistical account of when that sentence
is true. The authors separate a question usually treated as one: comparing two entire dictionaries
(how different are they overall?), locating the correspondence that explains the difference (which latents account
for it?), and separating both from the noise of having measured on a finite sample of text (is any of it outside
sampling error?). Their motivation is that an aggregate agreement number cannot point at a disagreement, that a
matrix of pairwise similarities ignores how all the latents compete for the same partners, and that the evaluated
input distribution is itself a free parameter — the same two trained dictionaries can allocate their mass
differently on different topics. The larger goal is to make "two latents are the same feature" a measurement with
a unit and a confidence interval, rather than a threshold applied to a cosine.

## What they studied

Independently initialised TopK sparse autoencoders trained at one fixed model location, same training data and
minibatch stream, differing only in seed. Four models: TinyStories-1M (block 3 output, d = 64, width 64, k = 16),
TinyStories-33M and GPT-2 small (both d = 768, blocks 1 and 6 respectively), and Pythia-160M (block 6), each of the
larger three at widths 3072 and 6144 with five seeds (seeds 106–120 added for a 20-seed analysis on the toy model).
Evaluation is by topic: disjoint held-out pools of 768 documents each for animals, family, food, toys, nature, plus
1,536 neutral stories, with topic word lists fixed in advance. The unit of analysis is the latent as a pair of
things — an activation function and a decoder direction — and the objects compared are distributions over latents.

## How they did it

Three similarity notions are defined and then shown to be cosines in three different Hilbert spaces: decoder
cosine in activation space; activation similarity as the uncentred correlation of the two latents' activations over
the input distribution; and causal similarity as the cosine between the output changes caused by ablating each
latent contribution, with a linearised variant using the Jacobian of the output map. Each latent is weighted by its
activation energy (second moment of its activation), the weights are normalised into a probability vector per
dictionary, and the two dictionaries are embedded as discrete measures over latents. An entropically regularised
optimal transport plan (Sinkhorn) between the two measures gives, in one object, a dictionary-level discrepancy cost,
an exact additive decomposition of that cost into per-latent contributions, and for each latent the fraction of its
mass assigned to any given counterpart or set of counterparts. Document-level energies and cross-moments are i.i.d.
across documents, so the authors derive influence functions through the transport plan (covering variation in costs,
in both marginals, and the dependence between several SAE pairs estimated on the same documents), give a joint
central limit theorem, validate with a joint document bootstrap, and add a finite-sample plan bound. Two idealised
models — a parent latent split into children with disjoint supports, and a general latent partly absorbed into a
specific one — predict what the transport fractions should be, and those predictions become the interpretive key for
the measured plans. A context contrast compares the same trained dictionaries across two topics, decomposed into a
weights term and a cost-matrix term by substituting one quantity at a time in both orders and averaging.

## What counts as an explanation here

A correspondence table with uncertainty: for any pair of latents, a fraction of one latent's energy mass that goes
to the other, a symmetric discrepancy cost between dictionaries, and a per-latent scalar contribution to that cost,
each with an interval. It is deliberately partial and explicitly not a semantic identity claim — the authors state
that a transport fraction quantifies correspondence under a specified similarity geometry, input distribution and
regularisation, and does not by itself establish that two latents mean the same thing or have equal effects on
outputs. Prediction enters at the model level: the splitting and absorption constructions predict the numbers the
plan should return, so a measurement that departs from them is informative about which regime the dictionaries are
in.

## How they judged success

Calibration of the intervals in three settings: a synthetic known-population experiment with 200 independent
samples per regime and sample sizes 64, 256, 1024; resampling of the fixed TinyStories document pools against
full-pool targets at m = 192/384/768; and a rare-document stress regime. Plus regularisation sweeps, a
seed-variation jackknife, and a head-to-head table against existing agreement measures (Hungarian mean cosine,
nearest-neighbour cosine, decoder shared fraction above 0.7, activation-correlation Hungarian and nearest-neighbour,
uniform-mass optimal transport with and without entropy, and the shared-fraction rule) on the same seeds and topics.
Self-comparison — a dictionary transported to itself — is reported as the floor for the cost.

## What they found

**The same two dictionaries look differently different depending on where you measure.** On the toy model the mean
transport cost is 1.02786 on animals and 1.05422 on family, a difference of 0.02639 (2.57% of the animals mean) with
95% interval [0.02277, 0.03002]; all ten seed pairs show the higher cost on family at every regularisation tested.
Decomposed, the weight term contributes 0.01680 (64%) and the cost-matrix term 0.00956 (36%), both intervals
excluding zero. The decomposition matters because the two terms can cancel: in GPT-2 at width 3072 the cost term is
positive (+0.0027 [+0.0016, +0.0037]) while the weight term is negative (−0.0029 [−0.0044, −0.0015]), so the total
contrast is −0.0003, and in Pythia the high-energy group accounts for most of its negative contrast. Across the six
larger settings the context contrast ranges from −2.3% to +2.1% of the animals mean, near zero in TinyStories-33M,
and with ten seeds at width 3072 the interval includes zero for all three models (TinyStories-33M +0.0004
[−0.0023, +0.0032], GPT-2 +0.0011 [−0.0014, +0.0035], Pythia −0.0016 [−0.0049, +0.0016]).

**Transport is the only measure that sees the topic effect at all.** On identical seeds and topics, the shared
fraction of decoders above 0.7 cosine is 0.0109 on both topics, the Hungarian mean decoder cosine 0.4080 on both, and
the nearest-neighbour decoder cosine 0.4350 on both — differences of exactly zero, because those scores do not depend
on evaluation documents. Uniform-mass transport gives z = 1.1 (unregularised) and 1.5 (ε = 0.30); energy-weighted
transport gives 15.4 unregularised and 15.0 regularised, and the nearest-neighbour shared-fraction variant 6.9. The
authors note these z-statistics estimate different quantities and so do not compare power against one hypothesis.

**Low-energy latents carry a disproportionate share of the discrepancy.** Ranking the 35 dictionaries' latents by
energy on neutral tuning documents and cutting into roughly 50/40/10% energy groups, the group holding 50% of tuning
energy accounts for 16–29% of evaluation cost while the group holding 10% accounts for 28–40%, across the six larger
model-and-width settings. In all 35 dictionaries the high-energy group's cost share is below its energy share (ratio
0.11–0.89) and the low-energy group's above it (1.3–4.5). The largest individual contributions are a mixed bag: at
ε = 0.10 the ten selected latents sum to 17% of the estimated difference, with family-word latents positive and
punctuation, narrative and animal latents negative.

**Correspondence is graded, not one-to-one.** Latent 21 of seed 104 and latent 13 of seed 101 share top tokens; on
family documents at ε = 0.10 nearly all of the first's mass (98.75%) goes to the second, while that mass is only
54.76% of the second's own mass. The same pair of latents is simultaneously a near-total match in one direction and
a partial one in the other, which is what the marginal constraints force and what a pairwise cosine cannot say.

**Theory supplies the failure cases.** Decoder and activation similarity are mutually unconstrained even under exact
reconstruction: a construction gives decoder cosine 1 with zero co-activation, and another gives decoder cosine 0
with co-activation 1. Under a constant output Jacobian the linearised causal similarity factorises into the
co-activation times a Jacobian-transformed decoder cosine and is therefore bounded by the co-activation; with nearly
parallel decoders whose common component the Jacobian annihilates, causal similarity is exactly the negative of the
co-activation, so near-identical directions can have opposite effects. In the splitting model every admissible plan
gives the parent's mass to child i in proportion to its energy share; in the absorption model the plan recovers the
absorbed fraction exactly as regularisation vanishes — which is what makes a transport fraction interpretable as a
split or absorb quantity rather than an arbitrary allocation.

**Inference works, with two named holes.** Simultaneous 95% bands cover 94.5–95.5% in the synthetic regular regime,
and the full-plan radius scales as N^(-1/2); the bootstrap beats pointwise Wald intervals, which are stated to be
anti-conservative under the same correlation. Rare high-impact documents break it badly: with a 0.5% probability
type, full-plan coverage is 0.275, 0.720, 0.850 and 0.935 at N = 64, 256, 1024 and 4096, and at N = 64 in 145 of 200
samples the rare type never appears and every band fails. Seed variation is comparable to document variation on the
toy model (seed standard error 0.00388 against 0.0018 from documents at ε = 0.10), and self-comparison cost is not
negligible — the diagonal is 1.000 at ε = 0.03 but 0.688 at ε = 0.30 and 0.208 at ε = 0.60, where correcting for it
shrinks the contrast by about three quarters.

## Limits and open problems they name

Inference conditions on the trained dictionaries, on fine-tuning-free selection, on positive retained energies, and
on fixed dictionary size and regularisation; causal-cost inference is explicitly out of scope and requires a
separate analysis, so the experiments use activation geometry (and the fixed-decoder variant) only. The guarantees
are about document sampling, not about training variability, which is estimated separately by a jackknife. The
authors state plainly that asymptotic validity does not buy coverage under rare-document contamination, that
sensitivity to regularisation and to self-comparison limits the reading of the numbers as regularisation-free
quantities, and that a small difference between topic means does not imply that either of its components is small.
Fractions depend on the chosen cost and corpus and are not semantic identity claims. The largest model studied is
Pythia-160M, so nothing here is tested at scale, and the causal geometry is characterised theoretically rather than
thoroughly measured.

## What we take from it

**A measure, and probably the single most reusable one this folder has met for its own claims.** Every "feature X
recurs across seeds" or "these two runs learned the same dictionary" statement we make can be replaced by: a
transport fraction with an interval, a per-latent contribution to the dictionary discrepancy, and a self-comparison
floor. The three-part shape — aggregate cost, exact additive decomposition, per-cell fractions — is the template for
any comparison we run between our own fitted dictionaries or between a dictionary and a model's true unit set.

**A warning that costs one line to obey.** The decoder-cosine measures gave a difference of exactly 0.0000 between
topics because they do not read the evaluation data at all, and the activation-correlation variants, which do, gave
z-statistics of 1.1–6.9 where energy-weighted transport gave 15. It is entirely possible for a stability number to be
reported as a property of the model when it is a property of the measure's indifference to the corpus.

**A denominator correction for the folder's energy-ordered thinking.** The entries that rank features by importance
and then check the top of the list are checking the *high-energy* head; here that head is cheaper than its energy
share and the low-energy tail carries 28–40% of the discrepancy from 10% of the energy. Any claim about which units
matter should be accompanied by the same cost-share-to-share ratio per group, because the tail is where split and
absorbed latents live and where seed disagreement concentrates.

**A ready-made negative control for splitting and absorption.** Proposition 3 and Proposition 4 give closed forms
for what a split and an absorption should look like in the plan, so a dictionary pair can be classified by measuring
against a predicted number instead of by inspecting top tokens — and the asymmetric fraction pair (98.75% one way,
54.76% the other) is the signature that a parent/child relation exists without either being "the same latent".

**A calibration protocol we should copy in miniature.** Known-population synthetic, resampling of the real document
pool against full-pool targets, and a rare-document stress regime, in that order, with the failure honestly printed:
a band that covers 95% in the regular regime and 27.5% under contamination has told you what the contamination does,
which is more than an unvalidated interval tells you. The rare-type construction is worth keeping as a checklist item
— a small high-impact stratum is the common failure mode for bootstrap intervals over sampled text.

## Tags

unit: feature
form: dictionary
method: SAE | decomposition | other
eval: prediction/simulation | benchmark
goal: evaluation-of-methods | understanding
