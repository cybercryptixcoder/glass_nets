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
