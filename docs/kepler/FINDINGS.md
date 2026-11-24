# Kepler: findings (Nov 24, 2026, as of 14:39 EST)

Every finding below passed tests that were written down before the result existed (ledger entries K1–K11 in
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

## 5. A far stronger network against 32,000 human decisions: which of its strategies are real, and which help humans (K10, K11)

**Setup.**
- **The network.** Leela's transformer t1-256x10, ported to PyTorch and checked against lc0's own code.
  - The port's one bug: lc0 silently gives old-format transformer files a squared-ReLU feed-forward layer.
  - It passes every registered check: opening moves, mate in one for both colours, queen-up values, and agreement with
    LD2 (value correlation 0.97).
- **The question.** In the same positions, what does t1 play that humans do not? Each candidate is a rule of the form
  "in situation C, play move type M".
- **Candidates.** 144 situations (each classical concept high or low, plus king placement) × 42 move types (ordinary
  ones, plus moves that classical principles advise against). That is about 6,000 rules.
- **The bar.** Each rule must clear all of these:
  1. A significance line taken from a control: the largest effect in 20 runs with network and human labels shuffled.
  2. Replication: the two older, independently trained networks must show the same tendency.
  3. Every human rating band must show it.
  4. It must be confirmed on 16,000 fresh positions from games never looked at.
  5. Cause: board edits that switch the situation off.
  6. Worth: the network's own value head.
  7. **Transfer:** the real results of the human games.

**What the miner finds first: the textbook.** The 18 strongest rules in the main pass are opening and endgame
principles that separate strong play from amateur play:
- develop knights while castling is still possible;
- push centre pawns while many pieces are on;
- push flank pawns in endgames;
- push pawns against an uncastled king.

These are known answers, recovered unprompted.

**Confirmed on fresh games** (all three networks and every rating band; rates are moves per position in the situation;
real-game column explained below):

| Rule (situation → move) | t1 | humans | ratio | Real games: humans who played the network's exact move | Same measure, the network's other moves in that situation |
|---|---|---|---|---|---|
| quiet position → **pawn lever** (a pawn push that attacks an enemy pawn) | 5.3% | 3.7% | 1.4× | +0.09 [0.01, 0.17] | +0.07 |
| enemy king still in the centre → pawn lever | 6.4% | 4.0% | 1.6× | **+0.16** [0.04, 0.28] | +0.06 |
| quiet position → pawn sacrifice (pushed pawn left en prise) | 1.0% | 0.6% | 1.6× | +0.21 [0.00, 0.41] (19 games) | +0.07 |
| behind in material → offer a like-for-like trade | 3.7% | 2.3% | 1.6× | −0.00 [−0.13, 0.12] | +0.06 |
| **position lost** (t1's own value ≤ −0.5) → **trade queens** | 3.6% | 1.8% | 2.0× | −0.02 [−0.15, 0.11] | +0.04 |

**How the real-game column is measured.**
- It compares humans who played the network's exact move with humans whose move lacked that type, in the same
  situation. Scores are game points (1 win, ½ draw, 0 loss), adjusted for t1's value of the position.
- **The comparison that matters is with the last column.** Playing the network's move helps humans by 4 to 7 points
  whatever kind of move it is. A rule adds something only if its own number beats that baseline.
- **None does significantly yet.** The pawn-play rules sit above the baseline (+0.16 and +0.21 against +0.06 and
  +0.07). The trading rules sit below it (about 0 against +0.04 to +0.06). The intervals overlap.
- A different lever than the network's own does nothing (−0.01, 33 games). The value lies in the network's specific
  move, not in "levers" as a category.
- **This reverses a reading I first wrote at 14:40** ("levers transfer, +0.06 to +0.10"). The controls for exact-move
  matching and for other move types were added at 14:48 and showed that number is the general value of playing the
  network's move.

**Three results stand out.**

1. **Humans under-play the networks' levers, at almost every rating level.**
   - The networks play about 5% levers in quiet positions at every rating level.
   - Humans play fewer at every level except the lowest. On the fresh games they fall from 4.3% (under 1300) to 3.3%
     (1900+); on the first sample they range from 3.2% to 4.7%, lowest in the middle bands.
   - The gap is largest for d-pawn levers (1.9×) and a-pawn levers (2.5×).
   - **Partly causal.** In matched edits (K11h), moving the enemy king from the centre to a castled square lowers t1's
     appetite for the lever by 13%. Moving it to the other centre square lowers it by only 3% (difference CI 6 to 15
     points). The effect is real, but under the 30% bar set beforehand, so it is not a pass.
2. **A textbook rule inverted, but only where the network is lost.**
   - The textbook says "when behind in material, avoid exchanges, especially of queens" (checked by web search; it is
     everywhere).
   - The networks offer trades 1.6× as often as humans when behind. Split by evaluation, all of the excess is in lost
     positions (4.8% vs 2.6%).
   - When behind but holding, t1 offers *fewer* trades than humans (0.65% vs 1.62%). That much agrees with the textbook.
   - In lost positions it trades queens twice as often as humans.
   - By its own value head, each such trade raises its draw chance from 6.0% to 9.9% (+4.0 points [1.9, 6.2]).
3. **That inverted rule shows no benefit for humans in real games.**
   - Humans who made the network's queen trade scored −0.02, against +0.04 for the network's other moves in lost
     positions.
   - It is right by the network's own judgement. Whether it is wrong for humans is suggested, not shown: the intervals
     overlap. A plausible reason is that self-play opponents do not blunder; splitting by opponent rating did not
     settle it.

**What failed, honestly.**
- **Cause tests.**
  - Being a pawn down is not what makes t1 offer trades. Removing an enemy pawn (now level) or an own pawn (now two
    down) lowers its appetite equally (15% vs 16%).
  - Rescuing a lost position barely changes its appetite for a given queen trade (7% vs 11% for control edits). That
    holds even when the rescue makes it winning.
  - So these rules describe where the network's choices differ from human choices. They are not a switch inside the
    network that the situation flips.
- **Unconfirmed.** Three of the seven exploratory rules did not survive the fresh games.
- **Not novel.** The lever is Kmoch's classical concept (1959), and the networks' extra pawn sacrifices echo AlphaZero's
  known style.

Evidence:
- `chess/t1_check.json`
- `chess/rules/mined_b.json`, `mined_c.json` (main and principle-breaking passes)
- `confirm_fresh.json`
- `queen_trade_fresh.json`, `trade_when_behind_fresh.json`
- `lost_cause_fresh.json`, `king_cause_fresh.json`
- `real_outcomes_fresh.json`, `real_outcomes_k8.json`, `real_outcomes_pooled.json`, `transfer_controls_fresh.json`
- `examples_fresh.json` (teaching positions with Lichess game links)

## What is new here (against `RESEARCH.md`)

- **Physics.** Video-physics interpretability work in 2026 probes for variables and plausibility. Reading a
  **governing equation** out of a video model (with coefficients, regimes and resolution limits), and reconstructing
  the hidden state it uses (velocity from smoke), with blind pre-registered tests, goes beyond them.
- **Mechanism.** A learned, nonlocal Biot–Savart response measured against an untrained-network control is a
  mechanism-level statement about how a video model does fluid dynamics.
- **Chess.** Schut et al. found concept vectors and described them by hand. Kepler states candidates as executable
  rules and applies frozen statistical, causal-decodability, non-redundancy and replication tests. So far, this has
  shown mostly what the network's extra knowledge is **not**.
- **Strong-network strategies, priced twice.**
  - Prior work looked inside AlphaZero (McGrath et al. 2022; Schut et al. 2025, concept vectors taught to grandmasters)
    or modelled human moves by rating (Maia, McIlroy-Young et al. 2020).
  - Kepler mines the gap between a superhuman network's choices and 32,000 human choices. It confirms rules on fresh
    games across three independently trained networks, prices each rule twice (the network's value head and real game
    results), and tests cause with board edits.
  - The real-game check carries its own control: the general value of playing the network's move. With it, the check
    can tell a rule that adds to human play from one that merely rides on it. At 16,000 games no rule clears it yet;
    pawn play leans above, trading below. We found no earlier method that runs this check automatically.

