# Planets: results (PL1)

All numbers are root-mean-square errors in log10(instability time / inner period),
predictions clipped to [4, 9] for every method. Test systems: 1,736 drawn at random from
the SPOCK 2.0 'random' dataset (systems that survive 1e4 orbits), the same systems for
every method. Training (fits) used the 'resonant' dataset only. Files in results/planets/.

## Primary result (final_law.json, law F4, censored fit)

| | systems | law (7 constants) | network (Cranmer et al. 2021, SPOCK) | law minus network, 95% CI |
|---|---|---|---|---|
| unstable before 1e9 orbits (pre-registered metric) | 731 | 1.104 | 1.159 | -0.055 (-0.145 to +0.038) |
| including stable systems (truth 9) | 1,736 | 0.880 | 0.872 | +0.008 (-0.050 to +0.069) |
| DEV half, unstable / with stable | 385 / 904 | 1.119 / 0.905 | 1.121 / 0.863 | -0.002 / +0.042 |
| FINAL half, unstable / with stable | 346 / 832 | 1.087 / 0.852 | 1.200 / 0.882 | -0.112 / -0.030 |

Pre-registered criterion (c), within 0.05 dex of the network or better: met on the
pre-registered metric (point estimate better), on the with-stable metric (+0.008), and on
both halves of the test set except DEV with-stable, which is at +0.042 (inside 0.05).

## Every method on the same 1,736 systems (information_same_systems.json, controls.json)

| method | inputs | unstable | with stable | correlation with truth (unstable) |
|---|---|---|---|---|
| constant | none | 1.459 | 2.804 | - |
| Lammers, Hadden and Murray 2024, as published | initial conditions | 2.004 | 2.152 | - |
| our best initial-condition law (4 constants) | initial conditions | 1.234 | 1.661 | 0.46 |
| gradient-boosted trees | initial conditions | 1.176 | 1.606 | 0.53 |
| **our law F4 (7 constants)** | initial conditions + 4 physical summaries of a 1e4-orbit integration | **1.104** | **0.880** | 0.66 |
| network | the 1e4-orbit integration's full time series | 1.159 | 0.872 | 0.71 |
| must-fail: law with the integration summaries shuffled across systems | | 2.366 | 2.074 | - |

Published numbers (their own evaluation choices, not re-run here): Petit et al. 2020
3.09 random (Cranmer et al. 2021); Alford et al. 2026 distilled equations 1.42 random against
their network's 1.26 on the same evaluation, i.e. 0.16 dex worse than the network. On our
evaluation the law is 0.055 dex better than the network on the same metric family.

## Every term earns its place (ablation.json)

Refit without one term at a time (censored, resonant only), same 1,736 systems:

| | unstable | with stable |
|---|---|---|
| full law | 1.104 | 0.880 |
| without the chaos term, log10(MEGNO - 2) | 1.219 | 1.020 |
| without the three-body spacing Xh | 1.189 | 0.989 |
| without the eccentricity forcing, log10(sigma_EM / EM_cross) | 1.144 | 0.973 |
| without the near-pair spacing, log10(EM_cross) | 1.127 | 0.892 |
| without the crossing term, min(R, 1) | 1.125 | 0.919 |

## Checks

- Recalibration (recalibration_check.json): an affine correction fitted on the DEV half's
  unstable systems (test-distribution data, which favours both), scored on FINAL:
  network 1.200 -> 1.015, law 1.088 -> 1.041. With its bias removed the network is ahead
  by 0.026 dex, still inside the pre-registered 0.05.
- Constants refit on five disjoint fifths of the training set (controls.json):
  intercept 5.41 +- 0.38, Xh 4.93 +- 0.98, MEGNO exponent 0.533 +- 0.035, eccentricity
  forcing exponent 1.30 +- 0.31, crossing term -0.55 +- 0.14, near-pair spacing -0.86 +- 0.56
  (the two spacing coefficients trade off; their combination is stable).
- Network baseline fidelity (nn_settings_check.json): our reduced sampling (300 samples,
  30 model samples) against SPOCK's defaults (1,000, 100) on 40 systems: mean absolute
  difference 0.063 dex, correlation 0.997.
- Chaotic floor (shadow integrations, law.py): 0.467 dex on the random set. Nothing can
  beat it; the law and the network sit about 0.6 dex above it.

## Fresh N-body verification (fresh_verification_F4_censored.json, fresh_robust.json)

800 new systems: members of the random population never used in this study, with new
random angles (pericentres, nodes, mean longitudes; distributions checked against the
original population). Integrated here with SPOCK's own N-body setup to 1e6 inner orbits,
scored with the law frozen at 21:25 (FROZEN_LAW.json). 693 were integrated by the stop;
659 survived the first 1e4 orbits (the protocol's population); 154 went unstable between
1e4 and 1e6 orbits (exact truth), 505 survived to 1e6 (censored).

| | law | network | law minus network, 95% CI |
|---|---|---|---|
| RMSE, the 154 that went unstable | 0.973 | 1.320 | -0.347 (-0.56 to -0.12) |
| censored RMSE, all 659 (survivors penalised only if predicted below 1e6) | 0.486 | 0.644 | -0.158 (-0.27 to -0.05) |
| survivors predicted to go unstable before 1e6 (false alarms) | 3.6% | 4.8% | |
| median absolute error, the 154 | 0.657 | 0.579 | |
| misses larger than 2 dex, the 154 | 3.2% | 12.3% | |

Reading: on new systems the law beats the network on both registered-style metrics, with
intervals that exclude zero. The network is slightly better on the typical system (median
error) but misses by more than 2 dex four times as often (it calls some short-lived systems
long-lived). For context, the held-out random set scored the same way (as if stopped at 1e6
orbits; random_as_if_1e6.json) gives law 1.034 vs network 1.121 on its 398 unstable systems
and censored error 0.522 vs 0.549: same direction, smaller gap, so part of the fresh gap is
likely sampling luck.

## How to reproduce

features.py and physfeat.py (initial conditions) -> nn_batch.py (network) ->
short_feats.py (summaries; Colab launchers in scripts/planets/colab) -> short_table.py -> final_law.py ->
controls.py, ic_only_same.py, recal_check.py, nn_settings_check.py -> fresh.py make/run
-> fresh_eval.py.
