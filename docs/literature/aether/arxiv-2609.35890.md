# Representational Simplicity and Circuit Size Dissociate in a Threshold-Dependent Way: A Controlled Test via Adversarial Training

- **Key:** arxiv-2609.35890
- **Authors / organisation:** Adam Elimadi; independent (Cambridge, MA); under review at ICLR 2027
- **Date:** 2026-09-27 (v2 2026-11-11)
- **Link:** https://arxiv.org/abs/2609.35890
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

The field keeps using cheap proxies for "this model is easier to understand": an SAE reconstructs the
activations well, or a feature-attribution map is concentrated on few features, and the inference is that the
model's actual computation must be smaller or better organised. The author asks whether that inference holds.
The target is reverse-engineering complexity in a strict sense — how much causal structure you must keep in
order to reproduce the behaviour at a fixed level of faithfulness — and the paper is set up as the controlled
test the proxy literature never ran. The larger goal is to know whether one can buy a more interpretable model
by choosing how it was trained, rather than by choosing a better dictionary afterwards.

## What they studied

GPT-2 Small (124M parameters, 12 blocks, 768-dimensional residual stream), seven continually trained variants
from one shared checkpoint: one standard control and six adversarially trained candidates (three perturbation
radii crossed with two clean/adversarial loss weights). Two tasks: Indirect Object Identification (competence
matched, 99.4–99.6% for all seven runs) and Greater-Than (not competence matched). Units of analysis: the SAE
dictionary at the residual stream entering block 8, the individual SAE features active at task positions, and
the edge-level computational graph (156 nodes — 144 attention heads and 12 MLPs — 32,491 edges).

## How they did it

Continual training rather than from scratch: every arm sees the same one-billion-token stream in the same order
for the same number of updates, with AdamW at 5e-5 and a cosine schedule. Adversarial arms mix clean and
adversarial loss with a weight, where the adversarial term is taken after a ten-step projected-gradient attack on
token embeddings held inside an ℓ2 ball sized relative to the mean token-embedding norm. Candidates are qualified
before any interpretability comparison: an accuracy floor on IOI, plus a sanity check that the loss under a
ten-step attack is close to the loss under a single-step one (so a model cannot look robust merely by having
masked gradients), and the qualified candidate with the smallest attack-induced loss increase is chosen as the
primary robust model. Three matched axes are then measured on the primary pair. First, BatchTopK SAEs of identical
width (12,288 features, 32 active per token) are trained on 300M held-out tokens and scored by held-out
reconstruction error, three seeds per condition. Second, node-level attribution patching assigns each SAE feature
a task score; the authors count the *candidate pool* (features active at task positions, counted before any
ranking), then the number of features needed to cover 50/80/90% of absolute attribution, and finally zero each of
the top-100 features and measure the KL shift in the full next-token distribution. Third, circuits are recovered
with edge attribution patching using integrated gradients over the whole raw graph on 200 name-swap prompts, and
the reported quantity is the first edge count on a fixed grid at which the circuit reaches a target faithfulness
— faithfulness defined as the fraction of the clean-versus-corrupted task-metric gap recovered when top-ranked
edges are clean-patched and all others corrupted-patched. Everything is repeated across the full strength sweep
and on a second corpus (FineWeb), and an SAE-free representational measure (the power-law decay exponent of the
activation covariance spectrum, plus a participation ratio) is added to show the representational trend is not an
artefact of the SAE.

## What counts as an explanation here

Not a description of the mechanism and not a dictionary. The explanatory object is a *size* — the number of graph
edges required to reach a stated faithfulness — treated as a scalar summary of how tangled a computation is. The
SAE dictionary and the attribution map appear only as predictors that are being tested, never as the explanation.
It is explicitly partial and explicitly conditional: the authors say the claim is about the smallest graph a fixed
algorithm recovers at a specified threshold, not a global minimum over all circuits, and they refuse to treat the
area under the faithfulness curve as a circuit-size measure at all, assigning it a different name and a different
status.

## How they judged success

Faithfulness at fixed edge budget on a geometric grid (1, 2, 4, … 16,000, 32,491), reported as the first grid
point crossing each of five targets (70/80/85/90/95%) — with an explicit rule that everything below 85% is a
comparison between two non-faithful partial subgraphs and not a comparison of circuits. Cross-condition replication
across four qualified robust models in a seven-cell sweep, plus a corpus swap. SAE reconstruction error with paired
position-matched blocks and three SAE seeds (non-overlapping seed ranges treated as the separation criterion).
Candidate-pool counts taken before ranking as the primary attribution statistic, with the 50/80/90%-mass counts as
secondary because share-of-pool is not a measure of how much a reader must inspect. An ablation-KL population
comparison as a real intervention (chosen because a linearised estimate of KL is identically zero at the unablated
point — a small but sharp methodological point). Conditional co-ablation for self-repair. A from-scratch control
that failed to acquire the task (55.4% IOI) is used to justify the continual-training design, and an EAP-GP pilot
is selected against by area under the curve *before* the standard-versus-robust comparison is inspected.

## What they found

The proxies move, cleanly and in the expected direction: robust reconstruction error is 1.1293 against 1.1601 for
standard (a 2.65% reduction, seed ranges disjoint), and the robust SAE has fewer candidate features on both tasks
(1,196 vs 1,612 on IOI, a 26% reduction; 360 vs 563 on GT), with fewer features needed to reach every attribution
mass threshold at all three seeds. The SAE-free spectral decay exponent tracks the same ordering across the sweep.
Circuit size does not follow that ordering uniformly. At 70% faithfulness the standard model needs the *smaller*
subgraph (512 vs 1,000 edges), at 80% and 85% the two tie at 1,000, and the robust model pulls ahead only at high
faithfulness: 1,000 edges to 90% against 4,000, and 2,000 to 95% against 16,000 — a fourfold and eightfold gap.
The 85% crossing is flat at 1,000 edges across the whole sweep (one non-qualifying outlier at 2,000, in the
*larger* direction), so the one threshold most papers would have picked shows no effect at all. The 90% and 95%
advantage replicates for all four qualified robust models and on FineWeb (1,000 vs 4,000 at 90%; 2,000 vs 8,000 at
95%). The standard model's faithfulness curve plateaus just above 90% and needs an enormous graph for the last few
points; the robust curve keeps climbing steeply over the same range.

## Limits and open problems they name

One architecture, one attack family, one SAE hook site, one competence-matched task. Circuit recovery is a single
deterministic run on a 200-prompt subsample with no prompt resample, no bootstrap interval, and no measurement of
seed-to-seed variation of the standard training run. The edge grid cannot resolve differences inside a bin. The
Greater-Than comparison is not competence matched, since the robust model scores lower on the same prompts (0.442
vs 0.671), so the GT circuit numbers are reported but not used as a test. The EAP-GP implementation follows a
description without a reference implementation and one approximation, and a bug cannot be ruled out. Whether the
pattern extends to other attacks, tasks, architectures and scales is left open. The AUC ordering, the
ablation-KL tail effect and the 85% flatness all point *against* a simple "robust is simpler" reading, and the
authors keep that tension in the paper rather than resolving it.

## What we take from it

**A measure, and a warning about how to report it.** The paper's decisive move is to make faithfulness a *free
variable* and read circuit size as a function of it, then report the crossings at several targets. On a single
threshold the answer here would have been "no difference" (85%, both 1,000) or the *opposite* answer (70%,
standard smaller) — and both would have been published honestly. Any white-box model we build that claims a
mechanism is compact has to state the faithfulness level it is compact *at*, and show the curve, because a circuit
comparison is a curve and a single number is a choice.

**A matched-instrument design we can copy.** Adversarial training here is used purely as a knob that moves
representations while behaviour is held fixed — the same checkpoint, the same tokens, the same step count, with an
explicit qualification step that refuses to compare models unless both retain the behaviour and both pass an
independent robustness check. That is a cleaner causal instrument than comparing two different models, and the
qualification-then-compare ordering is reusable.

**A dissociation we should expect in our own outputs.** Decomposability, attribution concentration and causal
compactness came apart here in the specific shape that the first two are monotone in the knob and the third is
not monotone at all. Our model should never let an SAE reconstruction number or an attribution-mass number stand
in as evidence about mechanism size; they get their own columns, and mechanism size gets a faithfulness curve.

**A small but sharp measurement warning.** Linearised (attribution-patching) estimates of KL are identically zero
at the unablated point, so a KL-based faithfulness check must use a real ablation. The paper's footnote about this
is the kind of thing that silently produces a zero column in someone else's pipeline.

**A design constraint on comparison.** Features carry no index correspondence across independently trained SAEs,
so the authors compare whole populations unpaired instead of per-feature ratios — a reminder that a per-feature
comparison across two dictionaries is not a comparison of anything.

## Tags

unit: circuit | feature | layer
form: description | graph | dictionary
method: SAE | attribution | patching | ablation | weight-analysis
eval: patching | faithfulness | ablation | benchmark
goal: understanding | evaluation-of-methods | safety/auditing
