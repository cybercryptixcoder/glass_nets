# Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation

- **Key:** arxiv-2603.28744
- **Authors / organisation:** Vitória Barin-Pacela, Shruti Joshi, Isabela Camacho, Simon Lacoste-Julien, David Klindt; Mila / Université de Montréal, Santa Clara University, Cold Spring Harbor Laboratory
- **Date:** 2026-03-30 (v2 2026-10-07)
- **Link:** https://arxiv.org/abs/2603.28744
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why
The field keeps finding that sparse autoencoders lose to plain linear probes when the test distribution moves, and the usual reading is that sparse decomposition does not generalise. The authors think that reading confuses two different claims: concepts being *encoded* linearly in activations, and concepts being *recoverable* by a linear map. Once the concept space is wider than the activation space, a hyperplane that separated classes in concept space can come out curved after projection, so a linear read-out is the wrong tool for geometrical reasons rather than empirical ones. Their goal is to work out, inside the classical compressed-sensing framework, where the failure actually sits: in the decoding algorithm, in the learned dictionary, or in the premise. The larger target is a principled route to recovering latents under superposition rather than abandoning sparse coding on the strength of a benchmark.

## What they studied
Synthetic data with known ground truth: latents in the unit cube with at most k active, observed through a linear mixing into a *smaller* activation dimension. Latent counts swept over 50 to 10,000, sparsity over a few values, and the undersampling ratio δ = d_y/d_h over a grid; sample counts from 10^2 to 10^5. Methods compared span the whole amortisation spectrum: oracle FISTA on the true dictionary, classical dictionary learning with FISTA inference (DL-FISTA), four SAE encoder variants (ReLU, JumpReLU, Top-K, matching-pursuit), per-sample FISTA on a frozen SAE decoder, and FISTA warm-started from SAE codes, plus a supervised ridge regression onto the true latents as a "skyline" linear probe. Real activations come from Pythia-70M-deduped layer 4 (512 → 2048) and Gemma-2-2B layer 12 (→ 16,384) on the SAEBench sparse-probing suite of eight text-classification datasets.

## How they did it
They build the OOD split by withholding *combinations*, not examples: with k = 2 and three factors, training sees the pairs (z1,z2) and (z2,z3) and test sees (z1,z3), so the label t = 1{z1 > 0.5} can be predicted in-distribution by riding on the always-present z2. The same trick is applied in concept space and in activation space, following the withheld-co-occurrence logic of spurious-correlation benchmarks. Recovery is then measured at three levels that are deliberately separable: mean correlation coefficient (best one-to-one Hungarian match between code columns and latents, invariant to scaling), a logistic probe trained on each method's codes and evaluated OOD with the same classifier for everyone, and per-feature AUC using the single best in-distribution feature. Model-level diagnostics sit underneath: cosine similarity between each learned decoder column and its ground-truth column, norm ratio, and support precision/recall of the binary activation pattern. The decisive experiment holds the dictionary fixed and swaps only the inference — same frozen SAE decoder, run as cold-start FISTA, as FISTA warm-started from the encoder output, and as the encoder itself — against DL-FISTA which learns its own dictionary. A λ sweep over three orders of magnitude checks that the conclusion is not a regularisation mismatch between the SAE's training penalty and FISTA's, and re-normalising columns or substituting oracle column magnitudes checks that the error is directional rather than a scale artefact. A two-dimensional toy with a closed-form OOD accuracy derivation in the appendix backs the geometry claim analytically.

## What counts as an explanation here
Not a description of features but a *decomposition of an error*: a partition of the pipeline into dictionary geometry, feature selection, and overall recovery, each with its own metric, so that a low MCC can be attributed to a named stage. The theoretical object is the LRH split into "linearly represented" versus "linearly accessible", with a dimension-count argument for when each holds. It is a partial explanation by construction — it says where the mechanism breaks, not what any individual feature means — and it predicts one thing sharply: that swapping an amortised encoder for per-sample inference on the *same* dictionary will not recover the gap, while a better dictionary will.

## How they judged success
ID and OOD MCC, accuracy under a shared logistic head, per-feature AUC, decoder-cosine to ground truth, norm ratio, support precision/recall. Baselines: oracle FISTA (upper bound), the supervised linear probe (lower bound), random-dictionary initialisation for the warm-start test, and the compressed-sensing phase-transition curve δ ≥ C·ρ·ln(1/ρ) as a theoretical reference. Checks: the δ sweep, the sample-size sweep, the sparsity sweep, the λ sweep, the column re-normalisation and oracle-norm substitutions, cold- versus warm-start convergence under a convex objective, and three seeds per real-activation cell.

## What they found
- Oracle FISTA stays near-perfect at every scale tested (MCC ≥ 0.95 in the main sweep, OOD accuracy about 0.97), so the OOD failure is not intrinsic to superposition.
- Per-sample methods show the sharp phase transition compressed sensing predicts, at a constant of about 2 in δ ≥ C·ρ·ln(1/ρ); SAEs improve with δ but plateau at 0.2–0.5 MCC and the gap does not close at high δ.
- Swapping the encoder does not help: SAE encoder, frozen-decoder FISTA, and warm-started FISTA all land at roughly 0.15–0.4 MCC on the same dictionary, while DL-FISTA with its own dictionary reaches about 0.75–0.95.
- The dictionary is wrong in *direction*, not magnitude: norm ratios are about 1.0 throughout, cosine spans 0.33–0.93 across SAE types, and re-normalising or substituting oracle norms changes nothing. At d_z = 5000 neither SAE nor DL-FISTA converges (cosine about 0.2–0.35).
- Support selection is worse than the recovery number suggests: at d_z = 5000 with k = 10, ReLU activates about 120 of 5000 atoms (precision 0.009), JumpReLU about 300 (0.005), and even Top-K, which gets the count right, has precision 0.03.
- More data closes the dictionary gap for DL-FISTA (about 0.5 to 0.98 MCC by 10^3 samples) but not the amortisation gap; JumpReLU gets *worse* with more data. Encoder capacity added to a toy SAE makes in-distribution reconstruction near-perfect while OOD stays completely wrong.
- The SAE decoder is still a useful warm start: Top-K's starts at 0.87 OOD MCC and reaches 0.94 in five dictionary rounds where random init starts at 0.30 and takes about fifty.
- On real activations the ordering holds: on Gemma-2-2B layer 12, DL-FISTA means 95.79% against Top-K SAE 94.63% and the raw-activation probe 95.62% (eight datasets, three seeds), and it exceeds Top-K on all eight datasets and the probe on six of eight. On Pythia-70M it is the only unsupervised method above raw probing, 92.38 against 92.21.
- The linear probe's identifiability collapses while its accuracy does not: MCC below 0.1 at 10,000 latents at a per-feature AUC around 0.93, which the authors read as single-label metrics masking an identifiability failure.
- A dimension-count argument gives the scale of the gap: with 10^6 concepts and 100 active, nonlinear recovery needs on the order of 920 activation dimensions while linear recovery needs on the order of 20,000, so a 4096-wide stream clears the first and misses the second.

## Limits and open problems they name
Real-activation results are in-distribution only — no public benchmark holds out known combinations of natural-language concepts, so the compositional failure mode is demonstrated on synthetic data, and building text splits is named as future work. Dictionary learning itself fails at scale for *both* classical and SAE methods, so the paper's own preferred route is bounded by the same unsolved problem it identifies; DL-FISTA also costs up to thousands of seconds at 10^4 latents. The linear-probe baseline is supervised on the true latents, which the authors acknowledge buys it per-feature AUC advantage and is why they lean on accuracy for the headline comparison. Everything is a binary label cut from a single latent and a single-task probe, with no multi-label or generation regime.

## What we take from it
- **Framing, load-bearing:** separate "linearly represented" from "linearly accessible" before claiming a linear read-out is evidence of anything. Our white-box model should state which of the two each probe result speaks to, because a nonlinear-looking decision boundary in activation space may only be a projection artefact.
- **Method, directly reusable:** the staged diagnostic ladder — dictionary geometry (cosine, norm ratio) → support precision/recall → end-to-end MCC — with the explicit statement that MCC alone cannot distinguish "wrong atoms" from "right atoms, wrong encoders". Our extraction pipeline should publish all three or it cannot localise its own error.
- **Control, the one that decides:** hold the dictionary fixed and swap the inference. This is the cheap, decisive experiment for separating "our read-out algorithm is weak" from "our learned basis is wrong", and it should be run before any claim that a better decoder will fix a poor result.
- **Warning:** a supervised probe's accuracy can stay high while its identifiability collapses (MCC 0.07, AUC 0.93). Any downstream-task number our model reports as evidence of recovery is paired with a latent-recovery number.
- **Shortcut:** an existing dictionary is a warm start even when it is a bad one — warm-starting converged to the same optimum and saved dozens of rounds, which suggests reusing a mediocre extraction as initialisation rather than discarding it.

## Tags
unit: feature | layer | weights
form: dictionary | replacement-model | probe
method: SAE | probing | decomposition | other
eval: ablation | prediction/simulation | downstream-task | benchmark
goal: understanding | evaluation-of-methods
