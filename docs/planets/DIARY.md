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

## 20:45 to 21:07 — where the law falls short, and why

- Saturating eccentricity forms (scripts/planets/fit_law2.py, results/planets/laws_round2.json):
  4 constants reach 1.25 on the random dev half, the same as the 6-constant L4.
  Binned table: the slope of log t in the three-body spacing falls as the relative
  eccentricity approaches orbit crossing and is nearly flat once orbits cross
  (36% of evaluated systems have crossing relative eccentricity yet survive 1e4 orbits).
- Censored maximum-likelihood fit (Normal scatter, truncated at 4, censored at 9) failed:
  sigma ran to 5.5, predictions worse (1.49). Mis-specified (outcome bimodal near the
  stability boundary). Abandoned; results kept in laws_round3_tobit.json.
- Network baseline batched: 1,750 random-set systems scored (samples=300, model
  samples=30). On the 638 unstable ones: network 1.168, law 1.229, so the gap is 0.06 dex.
- The decisive diagnostic: correlation with the truth is 0.70 for the network and 0.46
  for the initial-condition law; the network is biased high by +0.33 dex on this metric,
  the law is unbiased. For compact systems (spacing below -0.25) the network gets 0.96
  against the law's 1.24. So the network's 1e4-orbit integration carries real information
  that the initial conditions do not: an initial-condition-only law cannot honestly be
  said to match it, even where the RMSE numbers come close through the network's bias.
- Change of approach (not of target): the law may use what the network uses, a 1e4-orbit
  integration, but only through physically meaningful summaries: the chaos indicator
  MEGNO (Lyapunov time), the eccentricity-mode variation relative to orbit crossing,
  resonance strength, secular time (Tamayo et al. 2020 quantities, computed by SPOCK).
  This is the same information budget as the network and as Alford et al. 2026.
- Compute: short-integration features for 1,600 resonant (training) systems and the
  1,750 network-scored random systems, split between this workspace and an 8-CPU Colab
  runtime. Fresh verification set made: 800 new systems (population members never used,
  with new random angles), scripts/planets/fresh.py; integrations to 1e6 orbits next.
