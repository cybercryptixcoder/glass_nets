# Intrinsic Structure: Spectral Identifiability for Mechanistic Interpretability

- **Key:** arxiv-2608.10172
- **Authors / organisation:** Ashim Dhor (IISER Bhopal); Pin-Yu Chen (IBM Research)
- **Date:** 2026-08-10
- **Link:** https://arxiv.org/abs/2608.10172
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Mechanistic interpretability finds circuits, but has no way of saying whether a found circuit is a property of the model
or a residue of the procedure that found it. Sparse autoencoders make the worry concrete: different seeds and widths
recover materially different features from the same activations, and nothing in the field says whether that spread is
noise to be tuned away or structure that the method cannot avoid. The authors want to put a mechanistic primitive on an
identifiability footing — a recovered object that is determined by the model, recoverable from finite data at a stated
rate, and carrying an error bar. The larger goal is the one a safety argument needs: a certificate that a named mechanism
is intrinsic to the model rather than to the search that produced it.

## What they studied

Pretrained decoder-only transformers at two orders of magnitude of scale: GPT-2 small (12 layers, width 768), Gemma-2-2B
(26, 2304), Qwen3-8B-Base (36, 4096), plus three Pythia-160M checkpoints that differ only in initialisation seed and serve
as a negative control. Calibration data is WikiText-103, tokenised at sequence length 128, analysed at relative depths of
one quarter, one half and three quarters. The unit of analysis is unusual for the field: not neurons, heads or features,
but the **eigenvalue multiset of a linear realisation of the depth recurrence**, with its eigenvectors (Koopman modes) and
spectral projectors as the accompanying interpretable objects. Comparisons are made between three dictionary families —
a spectral dictionary of whitened principal directions plus random Fourier features, a public SAE's post-activation
code, and a random orthonormal control — at matched dictionary size and, crucially, at matched conditioning.

## How they did it

They reread the forward pass as a controlled dynamical system: holding a token position fixed, each layer maps the
residual state forward, the attention write of that layer enters as an exogenous control input, and depth plays the role
of time. A nonlinear state map can then be lifted by the Koopman operator, which acts linearly on observables; if the
span of a dictionary is closed under the layer map, the operator restricts to a finite matrix A, and its eigenvalues are
invariant to any change of dictionary basis. The estimated object is the standard EDMDc fit — a least-squares regression
from lifted states and controls to lifted next-states — so the spectrum is fit from cached layer-to-layer transition
triples rather than from weights. They prove the eigenvalues are recoverable up to permutation at the parametric rate in
the number of calibration samples, with a matching minimax lower bound, then test the predictions they can fail:
split-half spectral distance against sample count on disjoint source sequences with a sequence-level block bootstrap;
a median-of-means estimator for heavy tails; mode-based ablations against PCA and against random directions on the
canonical IOI circuit; the invariance residual of public SAEs; an invariance-penalty training sweep; and a cross-model
"universality" test that they run against their own seed replicas. Every modelling convention that defines the estimand
is measured rather than asserted.

## What counts as an explanation here

Neither a description nor a circuit: a **coordinate-free fingerprint with a stated error bar**. The explanation is a
multiset of eigenvalues, each with an interpretive type fixed by its modulus (amplifying, transport, decaying, with
rotation when the argument is nonzero), plus the modes and projectors that ride along with them; the eigenvalue multiset
is claimed to be *the* identifiable object, and the only residual ambiguity is a permutation, because eigenvalues are
rigid under similarity where SAE features are free up to sign, scale and rotation. It is explicitly partial: the paper
states that this spectrum does not decompose computation into human-legible mechanisms, and proves that under the
conditioning these models exhibit, it *cannot* coincide with the basis that carries activation variance. Can it predict?
In one striking case yes — the depth-wise growth of the residual norm comes back as a fitted eigenvalue, close to the
independently measured growth, on all three models.

## How they judged success

Self-consistency under resampling, since no ground-truth A exists on a pretrained model: split the calibration set into
disjoint halves, fit twice, and take the Hungarian-matched spectral distance between the two eigenvalue multisets, whose
log-log slope estimates the exponent; halves are drawn from disjoint source sequences so one document cannot appear on
both sides, and all uncertainty is a sequence-level block bootstrap. Predictions are tested as predictions — the
exponent should be minus one half, the exponent should worsen with dictionary size through a stated factor, a
norm-growth observable should produce a mode matching measured growth. Comparisons always include a random orthonormal
dictionary as a structure-free control and a whiten step to equalise conditioning, and one criterion is pre-registered
before the sweep. For behaviour, the IOI circuit in GPT-2 small supplies the semantic testbed: alignment of modes with
known head subspaces, logit-difference removed by ablating the top-j modes against top-j principal directions and random
directions, and reproducibility of the selected mode set across independent calibration draws.

## What they found

The spectrum converges on every model and attains the predicted exponent on the largest: minus 0.506 plus or minus 0.031
on Qwen3-8B-Base, against minus 0.286 on GPT-2 small and minus 0.329 on Gemma-2-2B. The authors are careful about what
that means — on Qwen the median and mean summaries give minus 0.338 and minus 0.375, so the maximum matched error (the
quantity the theorem bounds) converges faster than the bulk. Three candidate explanations for the shortfall on the
smaller models are excluded; the survivor is that the exponent tracks each cell's sample size relative to its own
threshold, which collapses thirty cells onto one monotone curve. The semantic claim fails in a way the theory predicts:
no head class reaches 0.8 alignment with the modal basis, ablating the top modes removes 25 per cent of the IOI logit
difference where ablating principal directions removes 60 per cent, and the top-16 mode set has a median agreement of
only 0.290 across independent calibration draws — the spectrum is identifiable, the selected sub-collection is not.
Public SAE dictionaries sit furthest from invariance, further than a random orthonormal basis (2.1, 3.1 and 3.4 times
the spectral residual across the three models), but that ordering reverses to 0.70 times under variance-based feature
selection, so the pre-registered criterion was met in only 49 of 75 cells. Adding the invariance penalty cuts the
split-half spectral distance by 41 per cent at fixed sparsity, while reconstruction worsens (FVU 0.222 to 0.255) and
cross-seed feature agreement falls from 0.526 to 0.428. The universality criterion declares all nine seed-replica pairs
distinct, so in the stated form it is not a universality criterion — though it still *ranks* seed pairs below
cross-family pairs, at an AUC rising to 0.97 at mid-depth.

## Limits and open problems they name

The estimand depends on a modelling convention they cannot justify: the attention write is computed from the same
residual stream, and residualising it against the lifted state shifts the spectrum by 5.2 to 9.1 times the resolution
floor, so the naive and residualised conventions select genuinely different objects. Invariance holds only
approximately for real dictionaries, and that bias does not vanish with more data, which makes the method not
model-agnostic. The intervention calculus is proved for first-order perturbations and is untested against measured
patching effects. The rate is asymptotic in a way that bites, since the threshold's unknown constant makes it only an
order-of-magnitude guide. The random-Fourier bandwidth was fixed for smaller widths and degrades in the deepest layers
of the widest model; they declined to retune it per model, on the grounds that tuning until the prediction appears is
the analyst-dependence they are criticising. Two results fail their own controls and are reported anyway.

## What we take from it

Five things. **(1) A vocabulary for the variability question** that our account needs and has so far lacked: the
distinction between *identifiable* (determined by the model, at a stated rate) and *legible* (readable by a human), with
a theorem that the two cannot be the same object whenever the realisation is non-normal — measured conditioning of 38 to
495 puts every model in the suite in that regime. **(2) The construction to copy**: split-half agreement between two
fits of the same object on disjoint data, reported as a distance *between objects* with a bootstrap interval, and used as
an error bar rather than as a pass mark; our probe and component claims have no such instrument, and this is the cheapest
one available. **(3) The negative result worth internalising**: eigenvalue multiset convergence at the stated rate, while
the *selected sub-collection* reproduces at 0.290 across draws — identifiability of the population does not confer
reproducibility of the nomination, and our account nominates sets constantly. **(4) A convention-and-controls discipline**
that is unusually explicit: every modelling choice that defines the estimand is measured and the measurements printed
(residualising the control, per-layer versus pooled fits, the read-out's reconstruction error), each comparison carries a
structure-free control at equal conditioning, and one comparison that inverted when conditioning was not matched is
reported uncorrected beside the corrected version so a reproducer can find it. **(5) A control we should steal**: run the
cross-model test against *seed replicas of one architecture* as models the test should decline to separate — the
authors' criterion separated all nine, which is what told them it was not a criterion but an ordering.

## Tags

unit: weights | direction | whole-model
form: replacement-model | direction | probe
method: weight-analysis | decomposition | probing | ablation
eval: prediction/simulation | faithfulness | benchmark | ablation
goal: understanding | evaluation-of-methods | safety/auditing
