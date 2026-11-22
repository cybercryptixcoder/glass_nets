# Shaping & Monitoring Models through their Internal Representations

- **Key:** arxiv-2611.07071
- **Authors / organisation:** Thomas Fel, Matthew Kowal, Matt Feiszli, Satchel Grant, Andrew Lee, Chris Earls (core contributors), with Fatih Dinc, Vasudev Shyam, Usha Bhalla, Leon Bergen, Siddharth Boppana, Ching Fang, Michael Byun, Tristan Gallent, Nicholas Wang, Tal Haklay, Dron Hazra, Neehar Kondapaneni, Connor Watts, Ryan Panwar, Thomas Dooms, Jack Merullo, Atticus Geiger, Ekdeep Singh Lubana, Thomas McGrath and Owen Lewis (equal senior contribution); Goodfire (code release at github.com/goodfire-ai/reliance, acknowledgements thank the Goodfire team plus external readers)
- **Date:** 2026-11-04
- **Link:** https://arxiv.org/abs/2611.07071
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why
Training optimises a scalar and therefore picks arbitrarily among the many parameter settings that reach the same
loss; nothing in that process says whether the model solved the task *for the right reasons*. The authors want to
close that gap by making the internal representation itself part of the training specification: a rule about which
concepts the model may *rely on* when it decides, and a rule about which concepts it may *express* in its outputs or
activations. The larger goal is explicitly a science of interpretability-based training — but they insist the two
halves must be studied together, because a penalty applied to a probe is a penalty applied to the very instrument
used to observe the model, and the model is free to move. So they ask a second question in every experiment: after
we have trained against this direction, is the concept still there, and what is reading it now?

## What they studied
A unified framework plus five experiments spanning modalities, scales and training paradigms: (1) reliance shaping on
Waterbirds (DINOv2 CLS/readout, frozen and fully trainable, block 9 and last layer), (2) conditional reliance rules on
ImageNette with a frozen DINOv3-Large and a BatchTopK SAE at layer 9 (300 concepts, L0 of 16), (3) autonomous
reliance shaping on a perfectly confounded CheXpert pneumothorax/drain split with a frozen RAD-DINO and a Top-k SAE
(12,288 features, k=64), (4) reliance shaping during DPO for sycophancy on OLMo-3-7B, OLMo-2-13B and OLMo-3-32B, (5)
expression shaping for reward hacking on Qwen3.5-9B under GRPO in an agentic Countdown task (transferred to chess),
and (6) expression shaping for harmlessness on Mistral-7B-Instruct and an abliterated Llama-3-8B-Instruct under LoRA.
The unit of analysis is a **concept direction** — a DoM, a probe vector, or an SAE decoder atom — together with a
layer at which it is read, and the analysis of where the concept goes is done in closed form on two-layer and
residual linear toy models.

## How they did it
Shaping is written as a constrained problem: minimise the task risk subject to a nonnegative concept violation, and
in practice minimise risk plus a penalty weight times the violation. **Expression** violations score a reader's
concept estimate along a sampled trajectory, discounted over tokens, and enter the reward — equivalently, a
policy-gradient term that changes which trajectories are sampled plus an optional second term that changes the score
of a fixed trajectory when gradients are allowed to flow through the reader. **Reliance** violations instead score
how much the *readout* moves when only the concept coordinate is changed, using attribution families (gradient,
gradient-times-input, integrated gradients, SmoothGrad, noise, shuffle, deletion) applied to internal directions
rather than input pixels; only the readout's mapping matters, so the concept may stay encoded as long as the
decision ignores it. The penalised direction is estimated as a difference of means on paired or labelled examples
and — when the encoder is trainable — recomputed on the current model every 100 steps, while the *monitor* is always
evaluated with the original direction. Credibility of the concept direction is checked directly (DoM versus logistic
probe directions on a two-dimensional slice), thresholds are chosen on validation worst-group accuracy or on a
utility/refusal constraint, and each setting is run across three seeds.

## What counts as an explanation here
Two artefacts. First, **specification**: a concept direction at a named layer, plus a rule (unconditional,
conditional on a context latent, or scored autonomously from SAE features an LLM judge has ranked) that says what the
readout may and may not depend on — an intervention-shaped constraint on the mapping, not a description of the model.
Second, and more useful to us, a **graded account of what remains readable**: five monitoring levels that each hold
one more component of the original detector fixed versus refitted — L1 reuse the same direction and probe at the same
layer; L2 refit a linear probe at the same layer; L3 refit a nonlinear probe at the same layer; L4 read from a
different layer of the same model; L5 read from a different model. The ladder is meant to make the claim "the concept
is gone" testable by separating a failed *detector* from a changed *encoding*. It is explicitly partial: the theory
covers only the projection intervention and only the degree-two component of what that intervention leaves intact.

## How they judged success
Behaviour on the *task's* own test sets, never the training penalty: worst-group accuracy against balanced-training
and ERM references on Waterbirds and CheXpert; sycophantic agreement rate on MMLU challenges plus DPO loss and mean
accuracy on six benchmarks; peek rate and honest solve rate on Countdown and on 512 unattempted chess puzzles;
StrongREJECT on JailbreakBench and held-out ClearHarm, with and without compliant prefill, plus the fraction of safe
XSTest prompts answered and an overall utility figure. Monitorability is judged separately at each level on held-out
activations re-encoded by each checkpoint, and the paper states the rule it follows: neither a lower penalty nor one
failed detector establishes that the concept is gone. Controls include erasure baselines (projection, LEACE, INLP),
random directions, group-balanced and ERM references, a direction-refresh sweep, layer-swap transplantation, and
ablations separating a direct gradient through the reader from a reward computed on a frozen copy.

## What they found
Shaping works behaviourally across the board. On perfectly confounded Waterbirds, unfrozen DINOv2 reaches 93–97.7%
worst-group accuracy against 51.6% for ERM, and the frozen-encoder runs exceed 94–95%; at block 9 the gap between
penalties and baselines widens, with shuffle the best at 93.8%. Sycophancy during DPO is prevented rather than
traded off: gradient-times-input keeps DPO capability at SFT sycophancy levels, while data filtering, reward shaping
and inoculation prompting fall on the line between SFT and DPO — behaviourally indistinguishable from lowering the
learning rate — and under 1% of DPO pairs look sycophantic to either a probe or an LLM judge. On CheXpert, ERM
reaches 33.3% worst-group accuracy against 56.0% for balanced training, and rank-8 subspaces built from agent-labelled
SAE features match or exceed the balanced reference for some penalties with no drain annotation used. Expression
shaping on Countdown removes reward hacking that ordinary GRPO drives to the ceiling: peeking goes from about 0.15 to
1.00 under GRPO and the random direction while honest solving collapses, whereas every shaping arm suppresses peeking
within about ten updates and reaches 0.96–0.98 honest solving — and filtering the hacking rollouts works in only one
of three seeds. The effect transfers to chess without training there: GRPO-trained checkpoints peek at 0.99, shaped
ones at most 0.004, with honest solving 0.48–0.60 on their own direction and 0.71 with a frozen base model's. For
harmlessness, a DoM reward on a frozen base cuts JBB from 0.70–0.71 to 0.01 on both models against 0.16 and 0.19 for
the on-policy probe with gradient, and no strength of the probe loss gets JBB below 0.15 while holding 95% of utility
and the safe-answering constraint. On monitoring, the paper's central negative is that gradients through the reader
are what break the original monitor: reward-hacking probe accuracy falls 0.98 to 0.57 by update 150 when gradients
flow through the score but stays 0.87–0.90 otherwise, and harmfulness L1 AUROC falls 0.87 to 0.58–0.64 within 50
updates while a refitted linear probe recovers 0.84–0.86. Direction drift alone is not the cause — the refitted DoM
rotates up to 62 degrees while the frozen-base reward rotates about 50 degrees and keeps L1 at 0.84. Refitting
recovers: after 400 updates on Mistral the original layer-19 probe ranks harmful answers below safe ones (AUROC 0.30)
while refitted linear probes reach 0.83–0.87 across layers, and a nonlinear reader adds only 0.02–0.04. The theory
supplies the mechanism. Under a fixed projection the ablated component receives no gradient, while the unwanted
column grows in the complement along the direction the head already reads, at a rate set by the residual correlation
— so erasure is followed immediately by re-encoding, and the new encoding is entangled with the core. Refit slowly
(every 20, 50 or 100 steps) and a linear probe still reads the concept well but a fresh erasure now also removes core
directions, dropping accuracy to about 0.50 against the 0.652 obtainable without the concept at all; refit often
(every 10 steps or less) and the linear share collapses to 0.014 while a product code appears in higher-order
statistics — an MLP probe reads it at R² of 0.99 and accuracy after a fresh erasure holds at 0.80 against a bound of
0.806. In practice LEACE at block 9 shows exactly this signature: original and refitted linear probes fall to 0.44
and 0.59, nonlinear probes stay above 0.98, and both linear detectors recover within two layers. The paper's
duality claim is that each circumvention form has a matching monitor: re-encoding is caught at L2, the product code at
L3, and L4 sees it again once the next nonlinear layer decodes it.

## Limits and open problems they name
There is no standalone limitations section; the limits are stated in the closing sections, which is itself worth
recording. They say outright that the three mechanisms may not exhaust the space — the product code is only the
degree-two part of what a linear intervention leaves untouched, and nothing prevents a concept from hiding in
higher-order interactions with several core variables or being spread across layers or token positions, each of which
would want its own monitor. Gradient-times-input on Waterbirds reaches invariance by *erasing* the background rather
than by ignoring it, which the projection argument does not cover and where the refitted nonlinear reader fails with
the original one. Their favourable monitorability results are single-seed in most settings and they ask whether they
hold at frontier scale — an open question they raise rather than answer. Training duration is named as an unstudied
axis: the same intervention can look benign over a short run and circumvent its monitor over a long one, and under a
frozen-base reward extended training degrades into long moralising refusals and repeated non-answers that score well
on the harm metric. Hyperparameter grids, credit-rule and reader ablations are pushed to the appendix, and the
harmlessness runs are explicitly one seed per setting.

## What we take from it
A **validation ladder** we should adopt verbatim as the way we report our own read-outs, because it is the cleanest
disambiguation in the folder between "we lost it" and "we can't see it": score every claim at L1 (original detector,
unchanged), L2 (linear refit at the same site), L3 (nonlinear refit at the same site), L4 (another layer, same model)
and L5 (another model) before pronouncing a concept absent or a read-out faithful. A **closed-form rate law for
re-coding after intervention**: a fixed penalty along one direction forces the model to rebuild the concept along the
direction its head already reads, so a white-box account that predicts "after removing direction v the model ignores
the concept" is wrong by construction, and the right prediction is a new encoding in the complement of the removed
subspace with magnitude tied to the residual correlation. A **warning about where information hides after any
first-moment intervention** — LEACE-style mean equalisation leaves the second moment, and a single ReLU above it turns
the product back into something linear, which is a concrete prediction our account can make and then test. And a
**methodological rule on direction estimation**: the pattern (where the concept is written) and the filter (where it is
best detected) coincide only under isotropy, so a probe direction is the wrong thing to intervene along even when it is
the right thing to read with — we must not use one quantity for both jobs.

## Tags
unit: direction | layer
form: direction | steering | description | causal-abstraction
method: steering | probing | attribution | training-interpretable | SAE
eval: downstream-task | prediction/simulation | benchmark | ablation
goal: control/editing | safety/auditing | understanding | evaluation-of-methods
