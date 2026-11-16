# Bridging Singular Learning Theory and Mechanistic Interpretability at the Billion-Parameter Scale

- **Key:** arxiv-2611.00640
- **Authors / organisation:** Ethan Ji, Wei Wang, Yizhou Sun; Department of Computer Science, University of California, Los Angeles.
- **Date:** 2026-10-31
- **Link:** https://arxiv.org/abs/2611.00640
- **Kind:** paper
- **Read from:** full text (arXiv HTML v1, ~231k clean chars: abstract, §1–§7, references, and Appendices A–Q including the proofs, the developmental protocol, the SGLD displacement measurements, the matched Metropolis calibration, the sensitivity panel and the subspace-verification tables). Figures NOT inspected, so every number quoted here is a table cell or an explicit sentence.
- **Relevance to us:** 3 (central)

## What they wanted, and why

Mechanistic interpretability (MI) arrives at an explanation by hand, one task and one checkpoint at a time, and
what it reports depends on the intervention it chose. Singular learning theory (SLT) supplies an automatic,
theory-backed number — the local learning coefficient — that can be computed for every component at every
checkpoint, but by itself it cannot say *which computation* a change in that number reflects. If the two agreed,
geometry could be used as a cheap signal for *when and where* a mechanism forms inside a model during training,
which is the monitoring question that comes before predicting a model's behaviour under intervention. The
authors frame the obstacle precisely: at billion-parameter scale there has never been an independently identified
mechanism to check geometry against, the standard estimator's parameter draws move further than anyone had
verified, and no work had separated the estimate from plain curvature. This paper supplies all three.

## What they studied

Pythia-1B across **all 154 released checkpoints**, at float64. The unit of analysis is the **component** — one
attention head or one MLP, 144 in total (16 layers × 8 heads, plus one MLP per layer). The task is bounded Dyck
languages of nested brackets (Dyck(2,10) for the main protocol, Dyck(4,16) for the main batch), scored only at
positions whose next token is a closer, on 128 strings of 128 tokens (5,475 closer positions). The model is a base
model, so it never fully learns the skill — final closer accuracy is 0.595 for k = 4, and 77% of its errors are
openers that are legal continuations — which the authors treat as a feature: the task is acquired gradually, so
there is a developmental trajectory to compare against. Inputs, draws, and coordinate conventions are all fixed
before any comparison (Table 1).

## How they did it

1. **Establish the circuit with interventions only, no geometry.** Zero-ablate each of the 144 components, then
   greedily restore components until 80% of the clean contrast returns, and assign roles using attention
   patterns, probes, direct logit attribution and path patching. This yields 26 circuit components in five stages:
   token encoding at layer 0, a stack-update group spanning L1–L5 MLPs plus three heads, a set of readout heads
   (L7H2/H4/H5, L9H4, L14H1), L11.mlp mapping bracket type to closer, and L13–L15 MLPs shaping the final
   prediction.
2. **Verify the role assignments on held-out strings** by fitting four named variable subspaces — TOKEN (rank 11),
   STACK (an additive slot-and-type code, rank 24), POSITION (32 directions) and ANSWER (centred closer
   unembeddings, rank 3) — and then projecting components' outputs onto or off those subspaces. Five rank-matched
   random subspaces per joint intervention are the control, evaluated on 384 strings never used for the fits.
3. **Define the shared object.** For a component c with parameters θ_c, write θ(u) = θ★ + E_c u where E_c rescales
   each tensor by its RMS, and define K_c(u) as the expected KL divergence from the checkpoint's own next-token
   distribution to the perturbed model's, averaged over scored positions. K_c vanishes exactly where predictions
   are preserved. Its Hessian is the component's Fisher information F_c. A parameter displacement is then *also* an
   activation intervention: the patched output change reproduces the perturbed model's prediction exactly (Eq. 6).
4. **Measure geometry inside an explicit region.** U is defined by KL ≤ 0.02 nats, cross-entropy change ≤ 0.1 nats,
   accuracy change ≤ 0.02, and no tensor displaced by more than 0.5 RMS — imposed at u, u/2 and u/4. Draws are 128
   shared Gaussian directions per component, both signs, at eight radii from 0.00625 to 0.4; the actual loss K_c and
   the quadratic Q_c are compared **on the same draws** by importance weighting, which splits each estimate into a
   Fisher part and a quartic part.
5. **Compute the quadratic part at every temperature anyway**, by stochastic Lanczos quadrature — four unit-norm
   probes × 32 Lanczos steps on exact Hessian–vector products — for all 144 components at all 154 checkpoints, giving
   the Fisher spectrum and hence q_Q at any inverse temperature τ.
6. **Define the four predictions up front:** P1 Fisher-trace changes track causal-effect changes; P2 the quartic
   coefficient carries what a non-local sampler registers; P3 for a redundant group, summed single-component geometry
   tracks summed single ablations and exceeds the joint estimate; P4 pairwise interaction curvature tracks the
   non-additive part of joint ablations.
7. **Compare**, per component, using Spearman and *partial* rank correlations with clean loss, weight scale and
   training step as covariates, on levels and on adjacent-checkpoint changes, with moving-block bootstrap intervals
   (blocks of 16 checkpoints) and a replication on an independent input batch.
8. **Audit the standard estimator**, by running the reference library's SGLD chains under the documented settings,
   with the loss-gradient term removed, and under strong localization — recording the applied displacement, the
   localization and noise terms separately.

## What counts as an explanation here

A **correspondence**, stated as a proposition and then tested term by term: the geometry of a component's loss and the
causal effect of intervening on that component are two readings of one map — the predictions as a function of the
component's parameters. Under a constant-rank Jacobian the learning coefficient equals half that rank, and the Fisher
matrix is the pullback of the downstream metric through the component's output map, so ker F_c = ker DG_c(0). The
understanding claimed is deliberately partial and coordinate-specific: which *quantity* answers which question. The
Fisher trace follows the model-wide course of causal effect; the component-specific quantities (gain Fisher, the
large-temperature count) follow a component's own role and give MI a measure that needs **no replacement value** —
no choice of zero, mean or subspace to ablate with; pair terms track interactions. Nothing here is a circuit or a
program; it is a translation between two existing formalisms, verified empirically.

## How they judged success

Rank correlation between a geometric quantity and a causal target — Spearman for levels, partial rank correlation
given clean loss, weight scale and step, block bootstrap for uncertainty — reported separately for levels and for
adjacent-checkpoint changes. Causal targets are KL from the clean model under zero-ablation (primary) and under output
scaling by 1 + g = 0.95 (a local ablation), plus contrast drop and the non-additive part of joint ablations. Controls:
a random other component's trajectory and the population-median trajectory (to separate component-specific co-movement
from a model-wide trend), Shapley attribution of trace change across the upstream/downstream/own weight split, an
independent input batch, matched rank subspaces, a coordinate control (isotropic prior in fixed converted tensors),
radius and temperature sensitivity on a pre-registered 18-component panel, and the same-inputs causal comparison at
the final checkpoint via prior-preserving Metropolis sampling with chain diagnostics.

## What they found

- **The circuit is real by intervention.** Removing the STACK subspace from the five stack-writer MLPs and three heads
  drops held-out closer accuracy from **61.4% to 14.8%** (contrast recovery 0.372 [0.345, 0.400]), against **58.4–62.6%
  for five rank-matched random subspaces**. Removing ANSWER from 11 heads and 4 MLPs gives 39.3% (0.481 recovery)
  against 60.4–61.8% random. The named bottleneck alone preserves 76.3% of the contrast; adding 64 residual-covariance
  directions raises that to 91.9%.
- **Signed contributions are not all positive.** L13–L15 MLP's ANSWER projections are *harmful*: removing them raises
  accuracy by **+2.5 / +2.6 / +2.8** points, while L11.mlp removal costs 8.4.
- **The region matters, and the standard sampler leaves it.** Under documented SGLD settings the median endpoint
  displacement is **4.76× the RMS of a weight matrix** (2.96 and 6.88 under the reference implementation's own
  settings) and **0/48 chains stay inside U**; holding a chain at 0.05 RMS needs γ ≈ 7×10⁵, seven orders of magnitude
  above the sweep's value, at which point only the prior is sampled. Under the reference settings the estimator's own
  statistic nβ(L̄ − L_clean) is **negative in 21–24 of 24 conditions** — the chain has fine-tuned the model on the
  batch. Strong localization reaches 48/48 inside U with 0.078 displacement.
- **Inside U the loss is quadratic plus quartic.** K agrees with its Fisher approximation to a median **0.04%** at
  r = 0.025; B is stable across radii (median ratios 1.000 and 0.998 at r = 0.05 and 0.2 against r = 0.1). Outside U
  the quadratic part is still computable from the spectrum, and q_Q/(τr²A) falls from 1 to medians of **0.58, 0.28 and
  0.04** at τr² = 10³, 10⁴ and 10⁶, growing like (τr²)^0.6 — it counts constrained directions, and the spectrum has no gap.
- **P1 holds, but mostly as trend.** The Metropolis LLC estimate at the final checkpoint ranks components as zero-ablation
  does at **Spearman 0.64** overall (0.51 within heads, 0.80 within MLPs); on natural bracket contexts the same
  quantity falls to **0.16** (heads-only −0.18), a limitation the authors state outright. Across training, Fisher-trace
  changes and ablation-KL changes correlate at median partial **0.51**, positive for 97% of components and with an
  interval excluding zero for 88%; but pairing A_c with the *population median* trajectory already gives 0.50 against
  0.26 for a random other component, so the excess is positive for only 82% of components. Within-class: 0.56 for the 26
  circuit components, 0.49 for the other 118.
- **The component-specific quantity is the gain Fisher.** F_gg (squared output norm under the downstream Fisher metric)
  tracks zero-ablation KL at partial **0.98 in levels and 0.97 in changes**, and is *specific* — 0.12 against the
  population-median trajectory — and once it is a covariate the Fisher trace's association with the ablation collapses
  to **0.02** (levels) and 0.07 (changes). Shapley decomposition assigns 0.44 of A_c's change to downstream weights, 0.21
  to the component's own, 0.35 upstream; the late rise of the trace is carried by the *input* group (median
  share 1.08 of the rise, with the component's own tensors contributing −0.23 — on their own the component's
  parameters would have lowered its own Fisher trace), and no single group carries the co-movement.
- **The large-temperature count does not mark a circuit component's formation.** q_Q at τr² = 10⁶ is component-specific
  (0.40 against its own ablation) and tracks ablation changes across the non-circuit population (0.47) but barely for the
  circuit components (**0.09** changes, −0.01 levels). A circuit component grows *along directions already constrained*,
  which the trace registers and the count does not.
- **P2:** with A_c among the covariates, B tracks the non-local SGLD estimate at partial **0.50** while A_c and F_gg do
  not (−0.07, −0.02), and B is *negatively* associated with the component's effect (−0.24). The quartic part is the
  shared, model-wide excess beyond curvature, not a role marker.
- **P3:** summed Fisher traces over the eight layer-0 heads track the sum of their single ablations (adjacent changes
  **0.71**) but not the joint ablation (0.03), and the joint LLC estimate is **5.8% below** the sum (3.90 vs 4.14), a gap
  5.7× a conservative MCSE bound — in the direction the coupling equation predicts. On natural contexts the joint and
  summed responses differ by under 0.7% and the level association is negative (Table 20: the summed trace bottoms at step
  6,000, the joint ablation peaks at 15,000). From step 8,000 on, the joint drop of the layer-0 bank exceeds even the
  **absolute** sum of the individual drops, so the non-additivity is not sign cancellation.
- **P4:** interaction curvature N_ab tracks the non-additive part of joint ablations at median partial **0.16 / 0.14 /
  0.16** over 43 circuit pairs on three independent batches, against **0.05** for 43 matched control pairs.
- **Everything is sensitive to the coordinate choice.** Repeating all 288 posteriors with an isotropic prior in fixed
  converted tensors gives zero-ablation associations of **0.861** on synthetic inputs and 0.561 on natural contexts,
  against 0.64 and 0.16 for the primary RMS prior — the authors flag this explicitly as a dependence on parameter metric
  and localization rather than a robustness detail. Radius and temperature changes preserve ranks (ρ 0.975–1.000) but a
  *readout* change does not: normalizing to the four closer types instead of the full vocabulary changes the median
  response by a factor of 0.683 on synthetic inputs and **5.9×10⁻⁴** on natural ones.

## Limits and open problems they name

One model, one synthetic task, one input batch, fixed coordinates. On natural bracket contexts — where the circuit was
not identified — the final-checkpoint correspondence collapses from 0.64 to 0.16 and the head-only values go negative,
so the calibration is demonstrated on the distribution the authors chose. The SGLD audit shows the standard estimator
does not stay near the trained parameters, which invalidates previous billion-scale LLC readings rather than repairing
them, and the fix (impose the region by rejection/importance weighting) only works while τ × spread stays below about
one, which bounds the temperatures at which the geometry is meaningful. The task's competence ceiling (0.595 closer
accuracy, 77% of errors being legal openers) is stated but not controlled for.

## What we take from it

- **A measurement theory for projection weights.** The gain Fisher F_gg = ‖o_c‖²_Σ is the component's output norm under
  the *downstream* Fisher metric — i.e. how far that write moves the next-token distribution — and it predicts ablation
  effect at 0.97–0.98 partial correlation while being computable without any replacement value. For a predictive model
  this is the cheapest scalar we have seen for "how much does this unit matter", and it beats the quantity the field
  usually quotes (parameter count, activation norm, weight norm) by construction.
- **The triple-null discipline.** Every geometry number here is reported against (i) its own Fisher approximation on the
  same draws, (ii) the population-median trajectory, (iii) a random other component's trajectory, and (iv) where relevant
  a rank-matched random subspace. The paper's own headline (0.51) survives at only ~0.25 of excess after (ii)+(iii), and
  the authors print that. Copy the triple, not the number.
- **A warning about which coordinate a "component" lives in.** Rescaling the parameter prior (RMS vs isotropic in
  converted tensors) moves the geometry-mechanism association from 0.16 to 0.56 on the same data. Any weight-space
  statistic we compute must state its parameter metric, and the sensible default is per-tensor RMS at a fixed reference
  checkpoint so a displacement means the same thing at every checkpoint.
- **Signed negative components are load-bearing.** Three late MLPs whose removal *improves* the score is the folder's
  inhibitory-cluster pattern in a new place, and here it is identified by subspace removal on held-out strings rather
  than by ablation alone.
- **Trend versus component-specific.** The most useful distinction the paper draws is that most of a "component-level"
  trajectory is a model-wide trend — median pairwise component-component rank correlation is 0.91 for the Fisher trace
  and 0.93 for the quartic coefficient, against 0.47 for ablation effects. A predictive model that ignored this would
  attribute a global learning-rate-shaped trend to individual units.

## Tags

unit: head | layer | weights
form: direction | probe | causal-abstraction
method: ablation | patching | probing | weight-analysis | other
eval: ablation | prediction/simulation | faithfulness
goal: understanding | evaluation-of-methods
