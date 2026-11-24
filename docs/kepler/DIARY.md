# Kepler diary

Times come from the system clock (`TZ=America/New_York date`).

## 2026-11-24

**09:27 EST.** Shreyas asked (09:20) for the boldest swing yet: one interpretability tool that makes real discoveries,
tested on a fluid video model and a chess network. Branch `kepler` made from `decompiler` (the furthest line: it
contains every other branch except `main`'s literature entries). Intent written (`docs/kepler/INTENT.md`). Ledger
entry K1 pre-registers the first known-answer test before any code: the trained video model's law must come back
with exactly the true terms, while the true parameters sit in a sealed file the engine never reads.

Machine: 2 CPU cores, 7 GB RAM, no GPU. So the models must be small and the work must overlap: data and training
run in the background while the research and engine code are written.

**09:31 EST.** Fluid simulator built and checked: pseudo-spectral 2D incompressible flow, 64 x 64 periodic box, RK4.
Hidden law: d(omega)/dt = -u.grad(omega) + nu lap(omega) - mu omega + F sin(kf y) - beta v (viscosity, drag,
Kolmogorov forcing, beta effect). Stable over long runs (rms vorticity about 0.85 in the forced state); 8 videos of
60 frames take 1.4 s. The frames show wavy zonal jets, so the dynamics is not trivial. Video model: a dilated
residual CNN (width 32, dilations 1, 2, 4, 8, 16, 1; receptive field covers the whole box, which velocity needs),
predicting the next frame. 0.16 s per training step at batch 8 on the 2 cores, so 8,000 steps (about 20 minutes).
Data: 320 training videos of 40 frames (starts from transients to the forced equilibrium), 64 held-out videos, and
32 out-of-distribution videos (stronger and finer fields than any training start) for the law-versus-lookup test.
Generation and training launched in the background at 09:30.

**09:52 EST.** Research block done (summary in `docs/kepler/RESEARCH.md`). Shreyas approved LLM spending (Luminal key)
and Colab; Colab signed in at 09:41 (balance 585 units). L4 and A100 were unavailable ("Service Unavailable"), a T4
was assigned.

- **Chess target ready.** Two Leela networks downloaded from lczero storage: LD2 and 744706, both 128 filters x 10 SE
  blocks, classical 112-plane input, convolutional policy, WDL value. Ported to PyTorch from lc0's weight format
  (batch norm folded as lc0 does). Checks that must hold, and do: opening moves e4/d4/Nf3/c4; mate in one found for
  White and for Black (board flip right); a queen up gives value +1.00 and, with the other side to move, -1.00;
  castling in the Italian is the top move (lc0 encodes it king-takes-rook, e1h1). LD2 solves 64.9% of 4,000 Lichess
  puzzles at top-1 with no search. 16,000 positions from real Lichess games (Jan 2013) and 4,000 puzzles (with
  human theme labels) evaluated; activations stored at all 11 sites; 61 classical concepts computed per position
  (`kepler/targets/chess/concepts.py`).
- **Second fluid target (K2), the smoke video model.** The same hidden flow carries a dye; the model sees only the
  dye frames (3 past frames), never the velocity, like a real video model. Generation and training (20,000 steps)
  launched on the Colab T4 at 09:51.
- **Fluid video model (K1)** training locally: step 3,800, error on the one-step change 22% and falling.

**10:09 EST. K1 known-answer test: ALL PASS (after one translation fix found by the test).**
`scripts/kepler/fluid_known_test.py`, result in `results/kepler/fluid/k1_known_test.json`.
- First run, (a) FAIL: from the simulator's exact derivative the extractor returned the 5 true terms (coefficients
  within 1.4%) plus two tiny extra terms (omega_x 8e-5, omega*ke -0.0018). Cause, shown by the code: the solver
  computes advection dealiased (the product projected onto the resolved 2/3 modes) while the library computed the
  raw product; the extra terms were fitting the aliased difference. A translation error (the library term did not
  match its intent, "u.grad(omega) as the solver represents it"): the library now projects the product the same
  way. Also a numerical floor: unexplained error below 1e-9 of the variance counts as round-off.
- Second run: (a) PASS, exactly {adv, lap, omega, v, sin(4y)} with advection -1.0, nu 0.0100, mu 0.0500,
  beta 0.600, F 0.500 (R^2 1.0). (a') reported: the two-step (Richardson) estimate on the exact solver gives the true
  terms within 1.6% plus a negligible hyperdiffusion term (its O(dt^2) bias). (b) PASS: shuffled targets R^2 0.003.
  (c) PASS: the untrained model is fit to R^2 0.987 but by 9 wrong terms. Lesson kept: R^2 alone would have been
  fooled by the untrained net; the term-set requirement is what makes the test discriminate.
- Extractor made fast (Gram matrices; identical least-squares fits): 4 extractions in 100 s.
- Smoke model on Colab diverged at the learning-rate peak (step 1,000: error on the change 0.15 -> 1.68). Relaunched
  10:09 with peak lr 6e-4, gradient clipping and fp16 mixed precision (10,000 steps).

**10:29 EST. Two blind verdicts, both PASS.**

*K1, the vorticity video model* (`results/kepler/fluid/k1_verdict.json`, `law_by_regime.json`). Trained 8,000 steps;
10.5% error on the one-step change of held-out pairs. From the model alone (no simulator), Kepler picked exactly the
5 true terms out of 38 (R^2 0.954): advection -0.985, viscosity 0.0105, beta 0.593, forcing 0.415, drag 0.021. Then
the sealed truth was opened: -1, 0.010, 0.6, 0.5, 0.05. **PASS** (exact term set, R^2 >= 0.9).
Reading by hand before trusting it: the true law fits the model's dynamics better (R^2 0.988) than the pooled
extraction, and tracks its rollouts better, so the drag/forcing gaps were not the model's physics. Fitting each
regime separately settles it: in the forced equilibrium (spin-up 40) the law read from the model is the true law
within about 2% (advection -1.0009, nu 0.0102, mu 0.0511, beta 0.602, F 0.511, R^2 0.987); in early transients
(random starts) drag disappears and forcing drops to 0.31, so the model applies forcing and drag only in the regime
it mostly saw. Two more findings:
- **Amplitude generalizes:** scaling equilibrium fields up to 3x keeps advection at -1.00 to -1.03.
- **Scale does not:** on fields whose structure is finer than training (dominant wavenumber 9, 11, 13 vs 2-8), the
  model's advection falls to -0.91, -0.79, -0.69 (viscosity also drops). Its physics has a resolution limit. The
  out-of-distribution set (strong and fine) gave advection -0.47 and no forcing, drag or beta: there the model is not
  applying the law (the law-versus-lookup test, Vafa et al. 2025, answered per regime).

*K2, the smoke video model* (`results/kepler/dye/k2_verdict.json`). Trained on the Colab T4 (10,000 steps, 1.6%
error on the change). It sees only dye. Velocity read per pixel from its activations: R^2 0.00, 0.25, 0.49, 0.77,
0.82, 0.87, 0.88, 0.89 from the input to site 7; the untrained copy stays at 0. The model reconstructs the invisible
velocity field, progressively through its layers. Using only that internal velocity, the law came out with exactly
the true terms {advection, diffusion}: -0.88, kappa 0.0025 (R^2 0.92). With the true velocity, the model's dynamics
are the exact law: advection -1.010, kappa 0.00307 (truth -1, 0.003; R^2 0.9996). **PASS** on all three criteria.
The weaker -0.88 with the internal velocity matches the readout's 89% accuracy (errors in a regressor shrink its
coefficient). Known test before it: (a) exact, (b) shuffled velocity R^2 0.016. Colab runtime stopped after
downloading the model (no idle units).

**10:35 EST. Fluid mechanism; chess round 1; where the chess network's extra knowledge lives.**

*Fluid mechanism (vorticity model; `results/kepler/fluid/mechanism.json`, `biot_savart_blobs.json`).*
- Atlas caveat found by the control: every LINEAR quantity (vorticity, its gradients and Laplacian, even velocity,
  which is a linear function of vorticity) is almost as readable from the untrained copy's random filters (control
  R^2 up to 0.96-1.00). Linear probes cannot show learning there. The nonlinear advection product u.grad(omega) can:
  R^2 0.13, 0.20, 0.24, 0.60, 0.80, 0.86, 0.97 from site 0 to site 6, untrained copy at most 0.26. The model
  assembles advection mostly in blocks 3-6, and computes no other quadratic quantity (energy, Okubo-Weiss: about 0).
- **Learned Biot-Savart law.** A vortex blob placed at 56 positions (distances 2-24 pixels, 8 angles) on 6 held-out
  backgrounds: the response of the model's internal velocity (probe at site 5) at the centre matches the exact
  Biot-Savart response with correlation 0.974 and overall strength 0.98. In the far field (9+ pixels) the trained
  model keeps 0.972, the untrained copy 0.698: the long-range, nonlocal part of the law is learned. Strength by
  distance: 1.11, 0.92, 0.87, 0.79, 0.78, 0.90, 1.13 (2 to 24 pixels).
- First kernel attempt (single-pixel Jacobian) was uninformative (correlation 0.07-0.13): Jacobians are dominated by
  local spikes; recorded and replaced by the blob test, which measures the physical response directly.
- Velocity swap (interchange intervention on the 2-D velocity readout at site 5): the output change follows the law
  with the swapped velocity (correlation 0.86 +/- 0.03), but the random-subspace control is also 0.69 +/- 0.03,
  because its change shares the same spatial pattern. Specific but partly confounded; a cleaner control is needed.

*Chess round 1 (`results/kepler/chess/dark/`).* The 22% of LD2's value that classical concepts miss was clustered
on LD2's own last-layer representation (8 clusters, 118-607 positions; e.g. 606 positions where LD2 is more
pessimistic than the classical estimate by 0.41, 89% consistent). The scientist (DeepSeek V4.1 Flash via Luminal,
high effort, 8 calls in parallel) proposed 24 concepts as python-chess programs. **0 of 24 accepted.** Several are
strongly encoded in the network (the opponent's forcing moves: decodable 0.79 vs 0.04 untrained; development deficit
0.92) but add nothing to the classical surrogate (gains inside the noise). The frozen tests did their job: human
chess ideas that sound right were not allowed through.

*Calculation (`results/kepler/chess/calculation_features.json`).* A material quiescence search (resolve all captures,
evade checks, depth <= 6) gives `qs_gain`, which passes all four criteria: gain +0.010 [0.004, 0.017] on LD2,
+0.018 on 744706, decodable 0.29 vs 0.03, max |corr| 0.48. It is classical engine knowledge (search), not a new idea,
and it closes only about 1 point of the gap.

*Is the gap real? (`surrogate_learning_curve.json`).* The surrogate's held-out R^2 with 1k, 2k, 4k, 8k, 15k training
positions: 0.712, 0.730, 0.756, 0.770, 0.778, flattening toward about 0.79-0.80. The gap is real, not a starved
surrogate. **Where it lives:** classical concepts explain 81% of LD2's value in game positions but only 36% in
tactical puzzle positions. The network's knowledge beyond classical concepts sits mostly in sharp positions (deep
tactical foresight, consistent with the learned look-ahead of Jenner et al. 2024). Next: look for strategic
(non-tactical) knowledge only in quiet positions, with a data-driven search over the network's own units that does
not start from human chess ideas.

**10:53 EST. Chess: rules from units (K4), the units' causal role, behaviour against humans; the lab page.**

- **K4 rules from units: 0 of 8 accepted, the noise control correctly rejected** (`results/kepler/chess/units/rules.json`).
  In 7,287 quiet game positions (no check, no capture resolution, no threat), the library plus calculation explains
  R^2 0.821 of LD2's value. Single channels at the last site track the rest: channel 22 correlates -0.45 (and -0.41
  on the other half), channel 15 -0.44/-0.41. Depth-3 trees over 42 per-square descriptors capture only 6-27% of
  each channel's activity, and the compiled rules add nothing: these units are not simple square predicates.
- **The units are causal** (`channel_evidence.json`). Resetting channel 22 to its mean moves LD2's value toward the
  classical estimate (correlation +0.41 with undoing the residual) and **removes 32% of the residual in quiet
  positions**; channel 15 removes 14%; channels 80 and 122 about 1%. Counterfactual drivers: removing any opponent
  piece (queen, rook, bishop or knight) lowers channel 22 by about the same amount (about 65), so it responds to
  the opponent's pieces regardless of their value. Tested as concepts: opponent / own / total piece counts and the
  count difference are restatements of the library (|corr| 0.82-0.97 with phase and material) and add nothing
  (`piece_count_tests.json`). The channel's contribution is a nonlinear function of the pieces, not their number.
- **Behaviour against humans** (`behaviour.json`, `rook_pawn_context.json`). Same 16,000 game positions (the
  sampler is deterministic: all 16,000 came back identical, now with the human's move): LD2 plays the human's move in
  37.3% of positions. It pushes **rook pawns 1.5x as often** (7.2% vs 4.7% of moves; interval +0.020..+0.029),
  in every rating band (the gap grows with rating: 7.7% vs 4.9% at 1800+) and on the second network. This is the
  famous engine idea (AlphaZero's h-pawn), recovered unprompted: a known answer for behavioural mining. Its extra
  rook-pawn pushes concentrate **in front of its own castled king: 2.0x humans (323 vs 162)**, against the human rule
  of keeping that shelter; on the opponent king's wing only 1.4x. Also: fewer captures (25.4% vs 27.2%), more quiet
  moves. The 'sacrifice-like' excess (4.7% vs 2.6%) does not replicate on 744706 (1.9%), so it is not claimed.
- **Scientist round 2** (refinement with round-1 scores) is slow on the endpoint: 3 of 8 answers after 30 minutes;
  still running.
- **Lab page `/kepler` (K5)** done: Fluid, Smoke and Chess as cards, details in sheets, dark and light, 1600 and 1280,
  no JS errors (Playwright). Fixed on the way: NaN in the summary JSON, the missing shell wrapper, chart text
  scaling on wide cards, chess glyphs drawn as emoji.
- **Velocity swap, cleaner controls** (`results/kepler/fluid/swap_controls.json`): at block 5 the output change after
  swapping in flow B's internal velocity matches the law's prediction with B's velocity (0.86) and much less with an
  unrelated flow C's (0.28); at block 1 the same swap changes the output by 1.4% (66% at block 5). The model advects
  with an internal velocity variable formed in its middle layers. Supersedes the confounded random-subspace control.
- **Scientist round 2: 0 of 15 accepted** (39 total over both rounds). The endpoint was slow (6 of 8 answers in
  30+ minutes; 2 clusters' answers could not be parsed, 2 never arrived, evaluated without them via NO_NEW_CALLS).
  Revised ideas are often strongly encoded in the network (unsupported pieces: decodable 0.88) but either restate the
  library (|corr| 0.88-0.94) or add no held-out gain. The frozen tests keep refusing plausible stories.

## Nov 24, 12:04-12:24 (resumed by the safety check-in) and 14:10 (after the session limit)

- **Minimal pairs (K9): 0 of 20 accepted.** Pairs of quiet positions that the 64-feature classical library cannot
  tell apart (nearest neighbours, classical distance 1.4-1.5) but that differ a lot in LD2's value residual
  (|difference| 0.96) or in channel 22's unexplained activity. The first attempt returned four empty answers: the
  model spent the whole token budget (38,000 tokens) reasoning. Second attempt: 6 pairs per prompt, low effort, a
  brevity instruction; 20 programs. None had a held-out gain with an interval above zero (best +0.0001); the FEN-hash
  noise program was rejected, as registered. Five programs were never scored because they named their function
  differently: a reader bug, fixed 14:13 (`compile_concept` accepts one function under any name). Conclusion after
  59 programs over three rounds: asking an LLM to guess what explains a weak network's value residual does not
  find new chess knowledge. The residual is mostly not a simple static feature.
- **Leela transformer t1-256x10 (K10) works.** The port was failing (start position q +0.93 with a flat policy). The
  cause, found by reading lc0's own C++: the file is an old-format attention network, and lc0's loader silently
  switches such networks with smolgen to a squared-ReLU FFN and swish smolgen (`src/neural/loader.cc`); the port used
  Mish everywhere. Everything else (gates, DeepNorm scaling, positional encoding, policy map, promotion offsets,
  value head) matched lc0 line by line. All four registered checks pass: start position d4 0.24 / Nf3 0.20 / e4 0.10
  / c4 0.09, q +0.04; mate in one for White (Ra8#) and the mirrored one for Black; queen up +0.998 / -0.986; value
  correlation with LD2 0.966 and top-move agreement 56.7% on 1,000 game positions (22 s on two CPU threads). The
  shuffled-encoder copy fails the same checks.
- **Session limit at about 12:25.** The work had finished but was not yet committed; it was committed and pushed at
  14:13 (73218fa9). That commit carries an AI co-author line against the repo rule; rewriting it needs a force-push,
  which was blocked, so it is left for Shreyas. Later commits are clean.
- **New direction for the last two hours (K11, registered before code):** stop guessing at value residuals; mine what
  the strong network PLAYS. For the 16,000 human game positions: t1's move against the human's move, conditioned on
  the situation (each library concept high or low, king placement). A rule "in situation C, play move type M" must
  replicate on both older networks, hold in every rating band, be causal (board edits that switch C off must lower
  t1's appetite for M more than edits that keep C), and be worth something (in those positions, human moves without
  M must lose more of t1's expected score than human moves with M). Novelty is checked by web search afterwards.
  Hard stop 16:09; clock checks scheduled for 15:00, 15:40 and 16:05.

## Nov 24, 14:14-14:40: the strong network against 32,000 human decisions (K11)

- **Evaluation (14:14-14:18).** t1 on the 16,000 K8 positions in 2.5 minutes on two CPU threads; 744706 too. t1 plays
  the human's move in 39% of positions (LD2 37%).
- **K11 as registered found nothing (14:18).** Zero of 3,600 situation x move-type pairs met "the network plays M at
  least 1.5x as often as humans". The networks and humans play every coarse move type at similar base rates, so the
  1.5x line was a guess that could only catch rare moves. Recorded as an intent error in the selection; the new line
  (K11b) comes from a control: the largest z in 20 shuffled-label nulls (z > 4.12).
- **K11b (14:19): the textbook, recovered.** 18 rules pass replication and every rating band: knight development while
  castling rights remain, centre pawn pushes with many pieces on, flank pawns in endgames, pawns against an uncastled
  king. These are known answers. Cause tests on three: one FAILED (its situation was a proxy for the opening), two could not
  be tested (single-piece edits never moved kings). Found two translation errors while testing: the worth loss mixed
  before/after positions, and t1 rates the side to move about 0.09 low with FEN-only history (its own top move
  "gained" 0.18 on average); worth is now measured against t1's own move. Edits now move kings too.
- **K11c (14:24): principle-breaking moves.** A second vocabulary (bishop for knight, own doubled or isolated pawns,
  pawn pushes in front of the own castled king, king walks, early queen sorties, pawn and exchange sacrifices, declining
  castling, queen trades, levers...). Everything that passed was "pawn lever in quiet positions": Kmoch's lever, played
  about 1.45x as often by all three networks.
- **K11d/e (14:26-14:34): exploration, then confirmation on fresh games.** The 5% firing rate hid rare moves, so I
  listed the strongest rare-move departures as exploration and froze 7 of them, plus two evaluation-conditioned ones
  (K11e), before sampling 16,000 fresh positions (games 12,001 onward). Confirmed: levers (two situations), pawn
  sacrifices, trade offers when behind, and **queen trades when lost** (t1 3.6% vs humans 1.8%; LD2 2.0%, 744706 2.9%;
  every band). The textbook (web search) says the opposite: when behind, avoid trades, especially of queens. t1's value
  head says each such trade raises its draw chance from 6.0% to 9.9%.
- **Cause tests failed (14:34-14:36).** Symmetric pawn edits (one pawn down: remove an enemy pawn, now level, or an own
  pawn, now two down) change t1's trade appetite equally. Rescuing a lost position (removing an enemy piece, q up to
  +0.62) barely changes its appetite for the queen trade in front of it. The rules describe where the networks' choices
  differ from humans'; they are not switches the situation flips inside the network.
- **Transfer to humans (14:38-14:40), the new measurement.** I recovered the fresh games' results (the sampler is
  deterministic; all 16,000 FENs matched) and compared humans who played the rule's move with humans who did not, in
  the positions where t1 plays it. **Levers transfer:** +0.08 [0.01, 0.15] and +0.13 [0.01, 0.24] game score, adjusted for
  the position's value. **Trading when lost does not:** -0.04 [-0.16, 0.09]; pooled with the first sample +0.04
  [-0.06, 0.13]; splitting by opponent rating did not show a pattern. So the same pipeline separates a lesson people can
  use from behaviour that pays only against a superhuman opponent. Humans play fewer levers than the networks at
  almost every rating level (on the fresh games 4.3% under 1300, 3.3% at 1900+; networks about 5%), most of all with
  the d-pawn (1.9x) and the a-pawn (2.5x).
- **K10 results file** re-run and saved (14:39): PASS; t1 solves 49 of 50 mate-in-one puzzles without search (LD2 44).
- Written: `docs/kepler/RESULTS.md` (the one-page deliverable) and FINDINGS section 5.

## Nov 24, 14:41-14:48: the cause and transfer claims, checked harder

- **Matched cause test (K11h, registered first).** Enemy king moved from the centre to a castled square vs moved to
  the other centre square, nothing else changed: t1's lever appetite falls 13.4% vs 3.2% (difference CI 0.06 to 0.15).
  A real causal effect, but under the 30% bar, so FAIL as registered. The first, unmatched cause test had failed for a
  design reason: its control edits often removed the lever's own pawns.
- **Transfer, with the controls it needed (and a correction).** My 14:40 reading, "levers transfer to humans
  (+0.06 to +0.10)", does not survive. Playing t1's exact move helps humans by 4 to 11 points for every move type
  (knights, bishops, queen moves, captures alike). Playing a different lever than the network's does nothing.
  Measured against the right baseline (the network's other moves in the same situation), no rule adds a significant
  amount: levers against an uncastled king +0.16 vs +0.06 and pawn sacrifices +0.21 vs +0.07 lean above; trading
  when behind (-0.00 vs +0.06) and queen trades when lost (-0.02 vs +0.04) lean below. FINDINGS, RESULTS and the page
  now say this; the reversal is recorded in K11g.
- **Page.** `/kepler` has a strong-network card (5 rules, rates, real-game numbers against the baseline) with sheets for
  the tests and teaching positions; checked in dark and light, no console errors.

