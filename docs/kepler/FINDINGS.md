# Kepler: findings (Nov 24, 2026, as of 10:46 EST)

Every finding below passed tests that were written down before the result existed (ledger entries K1–K4 in
`docs/glass/CHANGES.md`). Evidence files are under `results/kepler/`. The lab page `/kepler` shows them as cards.

## 1. A fluid video model's governing law, read out of the model blind (K1: PASS)

**Setup.** A small convolutional video model, trained on frames of a 2D flow, predicts the next frame. The flow
obeys a hidden law: advection, viscosity, drag, Kolmogorov forcing and a beta effect. Its parameters were sealed in a
file the engine never opened.

**Finding.** From the model alone, Kepler selected exactly the 5 true terms out of 38 candidates, with no distractor
(R² 0.954).

- **In the forced-equilibrium regime** (most of the model's experience), the law read from the model matches the
  truth within about 2%:

  | Term | Read from the model | Truth |
  |---|---|---|
  | advection | −1.0009 | −1 |
  | viscosity | 0.0102 | 0.010 |
  | drag | 0.0511 | 0.050 |
  | forcing | 0.511 | 0.5 |
  | beta | 0.602 | 0.6 |

- **The law holds in amplitude:** advection stays at −1.00 to −1.03 up to 3× the usual field strength.
- **It is regime-bound in two ways:**
  - In early transients, the model applies no drag and only 60% of the forcing.
  - On flows finer than it trained on, its advection fades: −0.91, −0.79, −0.69 at wavenumbers 9, 11, 13 (training
    range 2–8). The model's physics has a **resolution limit**.
  - On the out-of-distribution set (strong and fine flows) it stops applying the law: advection −0.47, and no
    forcing, drag or beta.

  This is the law-versus-lookup question of Vafa et al. (2025), answered regime by regime, with numbers.

Evidence: `fluid/k1_known_test.json` (the extractor proven on known cases first), `fluid/k1_verdict.json`,
`fluid/law_by_regime.json`.

## 2. How that model computes velocity: a learned Biot–Savart law (mechanism)

- **The kernel.** The model's internal velocity (read out at block 5) responds to a vortex blob placed 2–24 pixels
  away as exact physics says it should:
  - correlation 0.974 with the exact Biot–Savart response, overall strength 0.98;
  - in the far field (9+ pixels), 0.972, against 0.698 for an untrained network of the same shape.

  The long-range, nonlocal part of the law, which makes velocity hard, is learned.
- **Where advection is built.** The nonlinear advection product u·∇ω is assembled in blocks 3–6 (readout R² rises
  0.24 → 0.60 → 0.80 → 0.86 → 0.97; the untrained copy peaks at 0.26). No other quadratic quantity is computed.
- **Caveat (recorded).** Linear quantities, velocity included, are readable even from random filters, so linear
  probes say nothing about learning there.
- **The model advects with its own velocity variable (velocity swap).** Swapping in another flow's internal velocity
  at block 5 changes the model's output as the law predicts with the swapped velocity (correlation 0.86), much more
  than with an unrelated flow's velocity (0.28). At block 1, before velocity has formed, the same swap barely moves
  the output (1.4% vs 66%). Evidence: `fluid/swap_controls.json`. (A first random-subspace control, 0.69, was
  confounded by sharing the same spatial pattern and is superseded.)

Evidence: `fluid/mechanism.json`, `fluid/biot_savart_blobs.json`.

## 3. A smoke-only video model rebuilds the invisible velocity field, then applies the exact transport law (K2: PASS)

**Setup.** The same flow carries a dye. The model sees only the dye (3 frames) and never the velocity, like a real
video model.

- **Velocity emerges layer by layer.** Per-pixel readout R² is 0.00, 0.25, 0.49, 0.77, 0.82, 0.87, 0.88, 0.89 from
  the input to block 7; the untrained copy stays at 0. The model reconstructs the hidden flow field from smoke alone.
- **The law, read with the model's own velocity.** Using only the velocity read out of the model (never the
  simulator's), Kepler extracted exactly {advection, diffusion} (R² 0.92).
- **The model implements the exact law.** With the true velocity, the model's dynamics are advection −1.010 and
  diffusivity 0.00307, against a truth of −1 and 0.003 (R² 0.9996).

Evidence: `dye/k2_known_test.json`, `dye/k2_verdict.json`.

## 4. What a superhuman-lineage chess network knows beyond classical chess (K3, K4)

**Setup.** The Leela Chess Zero network LD2 (128×10 SE ResNet), ported to PyTorch and checked against known
positions; it solves 65% of 4,000 Lichess puzzles with no search. A second net, 744706, serves for replication.

- **Where knowledge forms.** Material balance is computed by block 1 (R² 0.96, against 0.18 for an untrained copy),
  then converted into the evaluation, which builds steadily to R² 0.97 at the last block. The pre-registered "material
  at the last layer ≥ 0.9" **failed** (0.79), and is recorded as a failure.
- **The measured gap.** 61 classical concepts (material, mobility, king safety, pawn structure, piece placement,
  threats) explain **78%** of LD2's evaluation on held-out positions (744706: 78%). More data does not close it: the
  learning curve flattens near 0.79–0.80.
- **Where the gap lives:**

  | Positions | Share of evaluation explained by classical concepts |
  |---|---|
  | quiet game positions | 81% |
  | tactical puzzle positions | 36% |

  The network's extra knowledge sits mostly in sharp positions: tactical foresight, in line with the learned
  look-ahead of Jenner et al. (2024).
- **What passed.** A capture-resolution (quiescence search) feature passes all four frozen criteria:
  - gain +0.010 [0.004, 0.017] on LD2, and +0.018 on the replication net;
  - decodable at 0.29 against 0.03 for the untrained copy;
  - not a restatement of any library concept.

  It is classical engine search, not a new idea, and it closes only about 1 point.
- **What did not.**
  - 24 human-style concepts written as programs by the automated scientist (DeepSeek V4.1 Flash): 0 accepted.
    Several are strongly encoded in the network (the opponent's forcing moves at 0.79 against 0.04), but none adds
    predictive power.
  - 8 rules synthesized from the network's own residual-predicting channels: 0 accepted, and the noise control was
    correctly rejected too.
  - Those channels (22 and 15) track the unexplained part at correlation −0.45, replicated across halves, and respond
    to the opponent's activity (its mobility, space and central control).
- **The units are causal.** Resetting channel 22 removes 32% of the residual in quiet positions (channel 15: 14%).
  Channel 22 responds to every opponent piece about equally, regardless of value, but piece counts themselves are
  restatements of the library: its contribution is nonlinear. Evidence: `chess/channel_evidence.json`.
- **Behaviour against humans** (same 16,000 game positions):
  - Leela pushes rook pawns 1.5× as often as humans. This holds in every rating band, the gap grows with rating, and
    the second network shows it too: the famous engine idea, recovered unprompted.
  - Its extra pushes sit in front of its **own castled king** (2.0× humans), against the human shelter rule.

  Evidence: `chess/behaviour.json`, `chess/rook_pawn_context.json`.
- **Honest status.** No new chess strategy has been verified yet. What Kepler established is the size, location and
  character of the knowledge gap, plus a test harness that refuses plausible-sounding explanations.

Evidence: `chess/atlas_LD2.json`, `chess/k3_known_test.json`, `chess/surrogate_learning_curve.json`,
`chess/calculation_features.json`, `chess/dark/`, `chess/units/rules.json`.

## What is new here (against `RESEARCH.md`)

- **Physics.** Video-physics interpretability work in 2026 probes for variables and plausibility. Reading a
  **governing equation** out of a video model (with coefficients, regimes and resolution limits), and reconstructing
  the hidden state it uses (velocity from smoke), with blind pre-registered tests, goes beyond them.
- **Mechanism.** A learned, nonlocal Biot–Savart response measured against an untrained-network control is a
  mechanism-level statement about how a video model does fluid dynamics.
- **Chess.** Schut et al. found concept vectors and described them by hand. Kepler states candidates as executable
  rules and applies frozen statistical, causal-decodability, non-redundancy and replication tests. So far, this has
  shown mostly what the network's extra knowledge is **not**.
