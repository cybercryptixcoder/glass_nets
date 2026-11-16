# Mechanistic Interpretability for End-to-End Self-Driving

- **Key:** arxiv-2610.41185
- **Authors / organisation:** Manav Gagvani, Benjamin Namikas, Sivamurugan Velmurugan, Nysa Kumar, Shrey Sharma, Thomas Peterson, Pratyush Mathur, Hyunseo Chang, Rohan Paleja; Purdue University (ECE / CS); CoRL 2026
- **Date:** 2026-10-30
- **Link:** https://arxiv.org/abs/2610.41185
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

End-to-end driving planners score well on open-loop benchmarks but say nothing about *why* a trajectory
was chosen, and the interpretability workarounds in the field (occupancy maps, motion forecasts, VLM
captions) all expose additional outputs rather than the planner's internal state. The authors want the
mining of the planner's own representation: which internal features carry route intent, which respond to
what is visible in the camera, which can be pushed to change the chosen trajectory, and whether that
feature space is good for anything besides looking at. The larger goal is observability — a pre-deployment
instrument that says what the planner is reacting to before it can be tested closed-loop. They also build
a deliberately small planner so that the internal state is one 384-dimensional query token per block, i.e.
a target a dictionary can plausibly cover.

## What they studied

A planner they train themselves, the Simple Monocular Trajectory Scorer (SMTS): a frozen Perception
Encoder ViT-S backbone over one front camera frame (32×32 tokens, d = 384), a learned convolutional
adapter, and four transformer blocks that update a *single* query token which cross-attends to the visual
tokens. The query is initialised from a 96-dim past-ego-state vector plus a one-hot route intent
(straight/left/right), which is the only route by which metadata enters. The query decodes 50 candidate
control sequences, each rolled out by a differentiable kinematic model into a 5-second trajectory, and a
learned scoring head assigns each proposal a scalar cost; the lowest-cost proposal is the output. Data is
WOD-E2E (250k training samples of Waymo's long-tail end-to-end benchmark, 106,360 validation scenes).
Unit of analysis: **SAE latents** on the post-block query activation at each of the four blocks, with the
final block (z₃, 384 latents, TopK k = 24, AuxK over dead latents) used for all the analysis.

## How they did it

Five instruments, in increasing order of causal strength. (1) **Cluster the dictionary**: filter latents to
an activation-frequency band, correlate the binarised firing patterns pairwise over the whole validation
set, take connected components of the correlation graph above a threshold, and label each component by the
majority route intent plus the sign of the mean speed z-score over the fifty scenes where the component's
summed activation is highest. (2) **Variance decomposition over inputs**: replay 9,000 balanced validation
images under all six combinations of three intents and two standardised ego histories (a stopped vehicle,
and cruising at the median moving speed), then run a three-way ANOVA on the 54,000 resulting latent vectors
to split latent variance into image, intent, ego-history and interaction terms. (3) **Paired counterfactual
image edits**: a detector supplies a scene description, a VLM proposes at most one edit from a closed
vocabulary (light colour, light presence, pedestrian, stop sign), a diffusion editor renders it, and the
planner is replayed on the edited frame with ego state and intent held fixed, so the delta in the query
token and in every latent is attributable to the pixels. (4) **Latent steering**: raise one latent by
α × a per-feature scale derived from its own active distribution, decode the modified latent back into the
query, and re-run the remaining planner heads, sweeping α over four levels on the 64 scenes where that
feature is most active; six output statistics are recorded per level and combined into a scalar
*controllability* score that multiplies effect size, monotonicity, cross-scene consistency and selectivity.
(5) **A downstream use test**: freeze everything and train a small adapter that predicts the base scorer's
per-proposal error from the latent plus the proposal and the base score, then reranks. The crucial part of
(5) is the ablation set — the identical adapter with the raw dense block-3 activation instead of the
latents (matched in every other respect), and the identical adapter with a randomly initialised encoder of
matched architecture and parameter count trained end-to-end on the same objective instead of the
pretrained SAE.

## What counts as an explanation here

A **feature dictionary** over one token position, plus three weaker companions: a *cluster label* (a
co-activating group of latents named by intent and speed sign), a *grounding claim* (this latent moves when
this specific thing appears in the image), and a *controllability score* (this latent, pushed, moves this
output statistic in this direction, monotonically, in most scenes). It is explicitly partial: the authors
state at the end of the steering appendix that intervening on the shared query measures differential output
responses, not head-isolated causal effects. Prediction is present but weak and stated as such — the
visual-response-versus-steering comparison is a correlation across features (r = 0.172), not a predictor of
what the planner will do on an unseen scene.

## How they judged success

Planning: ADE@5s on the WOD-E2E validation split against three retrained baselines, plus the private test
evaluator's rater feedback score per scene subset. Clustering: whether the components line up with route
intent (accuracy after majority-vote label assignment, per-class breakdown), stability across three
correlation thresholds, and agreement with a threshold-free community-detection algorithm. Grounding: the
ANOVA variance shares and their interaction terms, with the argument that a non-zero image×intent term
cannot arise if the latents merely copy the metadata. Steering: the four-way controllability score, with a
consistency floor before a feature is allowed into the ranking, and a natural-association-versus-steering
table (sign match, Pearson, Spearman across features) that asks whether the latent's ordinary correlation
with an output statistic predicts the direction pushing it moves that statistic. Practical value: relative
ADE reduction with two matched adapter controls, and the oracle rank of the selected proposal.

## What they found

The planner is competitive (ADE 1.40 m against 1.50 / 1.48 / 1.49 for the three baselines; best average
rater score 7.765, leading 6 of 11 scene subsets), which matters only because it means the dictionary is not
being read off a degenerate model. Clustering returns four components of 64, 36, 34 and 27 latents at the
primary threshold, recovering route intent at **98.3%** overall (99.8% straight, 92.9% left, 86.4% right),
and the two straight clusters separate low-speed from high-speed scenes (speed z of −1.04 versus +0.98);
94.3% of features keep the same label under the threshold-free alternative. The variance decomposition says
image contributes **14.3%** directly and a further **18.5%** through interactions with intent and ego
history, against 26.1% for intent and 24.0% for ego history. On 615 successful counterfactual edits drawn
from 1,000 scenes, stop-inducing edits shift specific latents and shorten or brake the selected trajectory.
Steering finds a strongly controllable braking latent (control score 0.481, effect +0.715 SD, monotonicity
0.941, consistency 1.000), and 254 of the 340 active latents clear a control score of 0.01. Natural
association predicts steering direction best for lateral displacement (59.7% sign match, Pearson 0.686) and
not at all for curvature (40.3%, Pearson −0.289). The adapter result is the one the abstract leads with:
**3.44%** relative ADE reduction and mean selected rank 16.7 → 15.6 out of 50 proposals, against **2.74%**
for the dense-activation control and **2.86%** for the randomly initialised encoder.

## Limits and open problems they name

Entirely open-loop, so nothing is said about how features or interventions behave under feedback. One
camera, no LiDAR or multi-camera planner. The edit analysis inherits the detector's and the editor's errors;
the appendix shows three failure classes (implausible edits, edits to objects off the driving path, and
visible edits with no trajectory change) and says the analysis was narrowed to stop-inducing edits *because*
the visual-to-behaviour link is strongest and most consistent there. Visual response and steering are
aligned but moderate, which the authors read as some visually responsive features not translating into
braking. The sweep is one-sided by construction — TopK latents are non-negative, so only increases were
intervened on, and they explicitly decline to infer symmetric behaviour for negative edits.

## What we take from it

Several things, of which the first two are warnings. **The matched-encoder control is the strongest arm in
the paper and it is under-sold**: 3.44% against 2.74% for the dense activation and 2.86% for a from-scratch
encoder trained on the *same* objective is a spread of 0.7 points on a single validation split with no seed
spread reported, while the prose says the SAE representation "appears to provide a more useful interface"
and the appendix says the controls "do not produce as large of an improvement". Two of the three controls
land within a quarter of the effect being claimed. This folder's recurring demand — print the controls in
the same table with a spread, and quote the margin, not the win — applies exactly here, and the entry should
be remembered as the cleanest example of an ablation set that is *well-constructed* (right units, right
matching) and *under-powered in the reporting*. **Second, an output can be perfectly predictable from a
structure that is not what computes it**: route intent is recoverable from the latent clusters at 98.3% by
an unsupervised procedure that never sees the label, and yet the cluster structure is organised around
intent × speed *modes* rather than around the intent variable, which is why two of the four clusters are the
same intent at different speeds. Recovery accuracy is a statement about the basis; it is not a statement
that intent is a feature the planner uses. Third, the **variance-decomposition instrument** is the cheapest
thing here to steal and the folder has no equivalent: hold two of three inputs fixed, replay every
combination on a balanced sample, and report main effects plus interactions, so that "the representation is
grounded in the image" becomes 14.3% + 18.5% rather than a panel of examples. A non-zero interaction term
is the part that cannot be produced by a model that copies its metadata forward, and that is the claim worth
making. Fourth, the **controllability score** is a better feature-quality scalar than the usual
interpretability score for our purposes, because it monetises exactly the properties we want in a
white-box account: effect size in units of the output's own spread, monotonicity across intervention
strength measured per scene and then averaged, consistency as the fraction of scenes agreeing in sign, and
selectivity as the share of the feature's total effect on the target statistic rather than on the others.
Multiplying them means a feature has to clear all four, and the ranking with a consistency floor of 0.5
produced 68 features above 0.1 out of 384. Fifth, a caveat the paper states and we should keep: the
intervention is applied to a shared downstream read-out, so what it measures is a differential response,
not the causal role of the component that produced the activation — the dictionary is a handle on the
query, and the query is the planner. Sixth, the natural-association table is worth copying as a diagnostic
of whether a dictionary's semantics *transfer to control*: for curvature it does not (40.3% sign match,
Pearson −0.289), which is a concrete instance of a latent whose label describes what it correlates with and
not what it does.

## Tags

unit: feature
form: dictionary
method: SAE | steering | ablation | other
eval: ablation | patching | downstream-task
goal: understanding | safety/auditing
