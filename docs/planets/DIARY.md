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

## 21:07 to 21:20 — the law, and the checks on it

- Short-integration summaries (SPOCK FeatureClassifier, Tamayo et al. 2020 quantities)
  computed for 1,720 resonant training systems and 1,736 of the 1,750 network-scored
  random systems (workspace plus Colab, 8 CPUs). Merged by scripts/planets/short_table.py.
- Final round (scripts/planets/final_law.py, results/planets/final_law.json): six forms,
  each fit twice on resonant only (unstable only; and censored, stable systems penalised
  only when predicted below 9). Chosen on the random DEV half: F4, censored fit:

      log10(t/P1) = 5.36 + 4.94 Xh - 0.53 log10(MEGNO - 2) - 1.27 log10(sigma_EM / EM_cross)
                    - 0.54 min(R, 1) - 0.96 log10(EM_cross,near)
      (systems that already go unstable inside the 1e4-orbit integration: 4.34)

- All 1,736 test systems, network on identical systems:
  unstable (pre-registered, n=731): law 1.104, network 1.159, difference -0.055
  (95% bootstrap CI -0.145 to +0.038); including stable (n=1,736): law 0.880, network
  0.872, difference +0.008 (CI -0.050 to +0.069). DEV and FINAL halves both within
  0.05 or better.
- Controls (scripts/planets/controls.py): shuffling the short-integration summaries
  across systems wrecks the law (2.37 / 2.07), as it must; the published Lammers, Hadden
  and Murray 2024 law on the same systems 2.00 / 2.15; a constant 1.46 / 2.80. Constants
  refit on five disjoint fifths of the training set: the chaos exponent is 0.533 +- 0.035;
  the two spacing coefficients trade off against each other (4.9 +- 1.0, -0.86 +- 0.56).
- Information check on the same systems (information_same_systems.json): correlation with
  the truth 0.46 (initial-condition law), 0.53 (initial-condition tree), 0.66 (final law),
  0.71 (network). The network's raw score suffers from a +0.33 dex bias on this metric.
- Hard check (recalibration_check.json): give both the network and the law an affine
  recalibration fitted on the random DEV half (test-distribution data), score on FINAL:
  network 1.015, law 1.041. With its bias removed the network edges the law by 0.026 dex,
  still inside the pre-registered 0.05. Stated plainly in FINDINGS.
- Fresh N-body set: 800 new systems; integrations to 1e6 orbits running here (items
  0-399) and on Colab (400-799), with short summaries and network predictions on Colab.

## 21:20 to 21:34 — freezing, ablation, fresh integrations

- Law parameters fixed by final_law.py at 21:18; written to results/planets/FROZEN_LAW.json
  at 21:25. Nothing below refits it.
- Network fidelity: reduced sampling against SPOCK defaults on 40 systems, mean absolute
  difference 0.063 dex, correlation 0.997.
- Ablation (ablation.json): every term earns its place; the chaos term matters most
  (dropping it: 1.104 -> 1.219 unstable, 0.880 -> 1.020 with stable).
- Fresh set: angle, inclination and mutual-inclination distributions match the original
  random population (checked), so it is a new sample of the same population.
- Pipeline cross-check: network and summaries for 16 fresh systems recomputed in this
  workspace match the Colab values (network 0.05 dex mean difference, sampling noise;
  summaries identical to 5 digits). Same spock 2.1.1 and rebound 4.6.0 on both machines.
- Interim fresh scoring just after 21:32 (343 integrated, 79 unstable before 1e6 orbits): law
  ahead of the network, with the bootstrap interval excluding zero on both fresh metrics.
  For like-for-like context, the held-out random set scored as if stopped at 1e6 orbits
  (random_as_if_1e6.json): law 1.034 vs network 1.121 on the 398 unstable systems,
  censored error 0.522 vs 0.549. So the gap on the fresh set is larger than on the held-out
  set; part of it is likely chance in a small sample.

## 21:34 to 21:46 — fresh verification scored

- Literature check before claiming (FINDINGS, What is new): Alford et al. 2026 say no general
  equations for instability times existed before theirs; Lammers, Hadden and Murray 2024
  fit idealized systems only; Hussain and Tamayo 2020 set the chaotic limit.
- Final pull of the fresh set at 21:45: 693 integrated (400 here, 293 on Colab), 659 in the
  protocol population, 154 unstable before 1e6 orbits. Frozen law against the network:
  0.973 vs 1.320 on the unstable ones (CI of the difference -0.56 to -0.12), censored error
  0.486 vs 0.644 on all 659 (CI -0.27 to -0.05), false alarms 3.6% vs 4.8%.
- Robust view (fresh_robust.json): the network has the smaller median error (0.58 vs 0.66)
  but misses by more than 2 dex on 12.3% of the unstable systems against the law's 3.2%.
- Decision at the 21:45 check: no extension needed. Everything registered is delivered.
