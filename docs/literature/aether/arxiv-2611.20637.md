# Counterfactual Feature Steering for Adjacency-Preserving Identity Unlearning

- **Key:** arxiv-2611.20637
- **Authors / organisation:** FNU Shivam, Ramana Reddy, Donald Adjeroh, Prashnna Gyawali (West Virginia University); Layth Sliman (EFREI Paris); Binod Bhattarai (University of Aberdeen)
- **Date:** 2026-11-16
- **Link:** https://arxiv.org/abs/2611.20637
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Erasing a concept from a text-to-image diffusion model is easy to do badly: push hard
enough and the surrounding neighbourhood goes with it. The authors name the failure
*adjacency preservation* — removing one public figure must not also remove the visually
and semantically neighbouring figures who share a profession, a prompt distribution or a
look. They keep the whole intervention at inference time, on a frozen model, so the thing
being evaluated is a direction read out of an internal representation and applied, not a
retrained checkpoint. The larger goal is the usual one: use an interpretability tool (a
sparse dictionary over the text encoder) to locate a concept well enough that an editing
operation can be aimed at it and judged separately from everything else.

## What they studied

Stable Diffusion 1.4 with its CLIP text encoder frozen; intervention at hidden layer 9,
selected by an ablation over layers 4, 6, 8, 9, 10, 11 (9 was best). The unit of analysis
is a **per-token direction in the sparse code of a k-sparse autoencoder** trained once on
text-encoder activations from a prompt corpus, plus a per-token retain centroid. Evidence
comes from 4,000 prompts derived from Celeb-100, eight target identities (Obama, Biden,
Musk, Gates, Messi, Ronaldo, Bradley Cooper, Scarlett Johansson), four manually chosen
"adjacent" identities per target and nine generic prompts; then a transfer study on nudity
(I2P, Ring-a-Bell at three strengths, MMA-Diffusion) and a qualitative Van Gogh-style study
on Unlearn Canvas prompts.

## How they did it

Prompts are split into a forget set (those naming the target) and a retain set (all others).
For every dictionary feature and every token position the mean and spread of activation are
computed on both sets, and features are ranked by a contrast ratio: forget-minus-retain mean
in the numerator, the summed forget and retain spreads in the denominator, with a retain
penalty weighting and an epsilon floor — the denominator exists so a feature whose activation
on non-target prompts is erratic is not subtracted, because it has no stable baseline to
subtract from. The top k features (k = 48, retain weight 0.1 for every experiment) define a
support set, and the direction at a token position is the forget centroid minus the retain
centroid on that support, unit-normalised. At generation a forward hook on the conditional
branch of the text encoder does three things per token: measures the signed alignment of the
current code with the direction (clamped at zero), subtracts a strength-scaled multiple of
that measured component along the same axis, and maps the sparse-space change back to hidden
space through the decoder's transpose — which is exact because the dictionary decoder is
linear, so no second forward decode is needed and the decoded direction is cached once per
target (about 925 KB of cached statistics per target; one SAE trains in roughly two hours).
Eight baselines run on identical prompts, seeds, steps and guidance: two dictionary-based
steerers, an SAE feature-suppression method on the U-Net, a fixed mean concept vector, a
classifier-free-guidance suppression, and two weight-editing methods. Strength-controlled
baselines are reported at their best operating point while the proposed method runs at full
strength.

## What counts as an explanation here

Not prose and not a circuit: a **direction plus a bounded projection rule**, and the paper is
explicit that the geometry guarantees are local. What is proved is a bound — for any strength
in the unit interval the alignment with the concept axis never crosses to the retain side, and
at full strength the operation is the minimum-Euclidean-norm projection onto the retain
half-space in dictionary coordinates, with tokens already on the retain side untouched. The
authors say in the same paragraph that these geometric properties guarantee nothing about
neighbouring concepts or image attributes, so the theoretical object is a **certificate about
one axis**, and collateral preservation remains an empirical claim. It predicts the sign and
the ceiling of the intervention's effect along its own axis only.

## How they judged success

Three net-utility scores, each a difference of means between treatment and control rather than
an absolute level. AWUS is the normalised CLIP image-text score drop on target prompts minus
the same drop on adjacent and generic prompts; FaceAWUS repeats that structure with ArcFace
face-identity embeddings against the unmodified generation; LPIPS-gap is the perceptual change
on target images minus that on non-target images. Controls and companions: paired
prompt/seed/generation settings, an independent COCO-30K study (10,000 captions) for general
preservation with FID and CLIPScore, a strength sweep for stability, ablations of k, the
retain weight and the intervention layer, a **raw hidden-space version of the identical
projection rule** as the test of whether the dictionary earns its place, a runtime table, and
a fixed-full-strength CLIP-space table published beside the tuned-operating-point comparison.

## What they found

Across the eight targets the method has the best net trade-off of the seven methods on all
three spaces: AWUS 0.296, FaceAWUS 0.636, LPIPS-gap 0.402, against 0.279/0.233/0.262 for the
best competitors on CLIP and a negative FaceAWUS for one steerer. The paper's framing number
is that methods achieving higher raw erasure pay for it with roughly twelve to twenty-two times
the collateral damage. At full strength the target-face similarity falls to 0.160 while
adjacent identities keep 0.760. On 10,000 COCO captions quality moves only slightly (FID 15.71
to 16.11, CLIPScore 27.39 to 26.95). Nudity detection on I2P falls from 17.80% to 0.47%, and on
the hardest Ring-a-Bell split from 93.68% to 12.63%, though the weight-editing baseline is
better on the adversarial MMA-Diffusion set (12.70% against 17.48%). The necessity ablation is
the most interesting cell: the **same** bounded projection applied directly in raw hidden space
gets slightly *more* target erasure (0.399 vs 0.387) and a slightly higher AWUS (0.391 vs
0.381) — the dictionary's contribution is cheaper collateral drift (0.0065 vs 0.0078) and
noticeably better peer-face retention (0.848 vs 0.722), not a better headline scalar. The
hyperparameter grid mostly *fails*: with k below 48 every cell is zero or near-zero utility,
and the run's chosen (48, 0.1) sits on a cliff edge next to 0.375 at the same k but collapse a
step earlier — and the ablation is a single identity.

## Limits and open problems they name

No standard benchmark exists for identity unlearning with adjacency preservation; the
neighbour sets are hand-curated with two post-hoc similarity views (layer-9 name-token cosines
and CLIP image cosines) offered as justification rather than as a definition, and a model-
internal ranking achieves only MRR 0.487 on thirteen peers. Face detection fails on 20.8% of
target outputs against 3.4% of peer outputs, and the unmodified model's own self-consistency
across three prompt templates ranges from roughly 0.15 to 0.66, so for low-consistency targets
the test of "meaningful removal" is weak. Direction geometry is not guaranteed orthogonal —
one pair is explicitly entangled, and the analysis is aggregated across the eight targets, so
single-target collisions can hide in the average. Everything depends on dictionary quality.
The deployment scope is stated plainly: it is a hosted-inference mechanism, not tamper-proof
against a user who controls the serving code. All experiments are one backbone (SD 1.4), and
the nudity and style studies reuse the same rule with separately trained dictionaries rather
than testing transfer of a direction.

## What we take from it

Four things, and the second is the one worth arguing about.

1. **A one-sided score with a control population in the denominator.** Every metric here is
   `effect on the treated − effect on the control crowd`, and the control crowd is drawn from
   the same data collection rather than from scratch. For our read-outs the analogous move is
   cheap: for every component we claim is specific to a condition, report its effect on the
   condition *minus* its effect on the near-neighbour conditions, because a direction that
   moves both equally has produced a shift, not an attribution.
2. **The nearest-defence ablation: apply the identical rule in the representational space
   underneath.** They replace the dictionary code with raw hidden states and keep everything
   else, and their own headline drops — the method's advantage survives only on the selectivity
   side (collateral drift, peer-face similarity). That is the ablation our battery should run by
   default for every dictionary-mediated claim: the same edit at the same layer with the
   dictionary removed, reporting the scalar the claim rests on *and* the selectivity numbers.
3. **A certificate scoped to one axis, published as scoped.** The boundedness proof is exact
   and says nothing about the neighbourhood; the paper states this in the same sentence rather
   than letting the reader infer generality. Copy the sentence, not just the theorem.
4. **A hyperparameter cliff is a finding.** The k-sweep shows the reported configuration sits
   one step from a zero-utility collapse; publishing the sweep beside the choice is what turns
   a lucky setting into a calibrated one.

## Tags

unit: direction
form: direction
method: SAE, steering
eval: ablation, faithfulness, downstream-task
goal: control/editing, understanding
