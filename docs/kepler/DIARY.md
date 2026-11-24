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
