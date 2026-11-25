# Planets diary (times from the system clock, America/New_York)

Target (accepted 20:14, Nov 24 2026): a compact, physical formula for when compact
three-planet systems go unstable that matches the Bayesian neural network in SPOCK
(Cranmer et al. 2021), judged on systems it never saw, and checked against fresh
N-body integrations. Ledger row: PL1 in docs/glass/CHANGES.md.

## 20:15 to 20:45 — data, protocol, first laws

- Data: SPOCK 2.0 dataset (resonant 113k systems, random 25k systems), REBOUND
  initial conditions plus N-body instability times and shadow-integration times.
- Protocol (registered in PL1): fit on the resonant set only, judge on the random
  set only; score = RMSE of log10(t/P1) over systems that survive 1e4 orbits and go
  unstable before 1e9 (7,609 random systems). Predictions clipped to [4, 9].
- Chaotic floor from shadow integrations: 0.437 dex resonant, 0.467 dex random.
- Published law of Lammers, Hadden and Murray 2024 on these systems: 2.01 random
  (worse than predicting a constant, 1.40).
- Our first laws (scripts/planets/fit_law.py): best so far L4, 6 constants, 1.253
  random. Variables: three-body spacing (harmonic combination of the two pair
  spacings, in units of the pair mass to the 1/4), both pairs' relative
  eccentricity in units of the orbit-crossing eccentricity.
- Ceiling check: a gradient-boosted tree on the same initial-condition variables
  reaches 1.140 random; the network on a first 24-system subset 1.149.
- So the initial conditions alone carry about as much as the network gets from its
  1e4-orbit integration; the remaining gap is the functional form (about 0.1 dex).

## 20:45 — next

- Network baseline batched (0.44 s per system instead of 20 s): 3,000 random-set
  systems in the background (scripts/planets/nn_batch.py).
- Close the functional-form gap; then fresh N-body integrations.
