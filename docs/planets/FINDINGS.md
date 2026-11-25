# Planets: findings (PL1)

## The finding

When does a compact system of three planets go unstable? The most accurate predictor so
far has been a Bayesian neural network (Cranmer et al. 2021, shipped in SPOCK). It runs a
short simulation (10,000 orbits) and feeds the whole time series to the network. Earlier
attempts to replace it with equations either lost accuracy or lost physical meaning: the
2026 distillation of the network into equations (Alford et al.) is 0.16 dex worse than the
network on random systems and uses terms with no units (a sine of a non-angle).

We found a seven-constant formula, in physical quantities only, that predicts instability
times as accurately as the network on 1,736 systems neither had seen:

    log10(t / P1) = 5.36 + 4.94 Xh - 0.53 log10(MEGNO - 2) - 1.27 log10(sigma_EM / EM_cross)
                    - 0.54 min(R, 1) - 0.96 log10(EM_cross,near)

and t / P1 = 10^4.34 for the few systems that already go unstable inside the short simulation.

| test metric | formula | network | difference (95% CI) |
|---|---|---|---|
| systems that go unstable (pre-registered), 731 | 1.104 | 1.159 | -0.055 (-0.145 to +0.038) |
| all systems, stable ones counted at 10^9, 1,736 | 0.880 | 0.872 | +0.008 (-0.050 to +0.069) |

Errors are in dex (log10 of the instability time). The two are statistically tied; the
formula is ahead on the pre-registered metric and level on the other. It was fit on a
different dataset (the 'resonant' systems) and never tuned on these.

## What the quantities are

- Xh, the three-body spacing: the harmonic combination of the two neighbouring pairs'
  fractional spacings (a2 - a1)/(a2 + a1), in units of the pairs' mass ratio to the 1/4
  power (the three-body resonance-overlap scaling of Petit et al. 2020). From the starting orbits.
- R, how close the starting orbits are to crossing: the pair's relative eccentricity
  |e2 exp(i w2) - e1 exp(i w1)| over the orbit-crossing value (a2 - a1)/(a2 + a1); the
  larger of the two pairs, capped at 1 (crossing). From the starting orbits.
- EM_cross,near: the fractional spacing (a2 - a1)/a1 of the closer pair. From the starting orbits.
- MEGNO: the standard chaos indicator, measured over the short simulation. It sits at 2 for
  regular motion; for chaotic motion it grows like half the simulation time over the
  Lyapunov time. Floored at 2.01.
- sigma_EM / EM_cross: how much the closer pair's eccentricity mode swings during the
  short simulation, as a fraction of its orbit-crossing value (Tamayo et al. 2020).

## What it says physically

- Spacing: together the two spacing terms give a time that grows roughly as the fourth
  power of the spacing measured in units of planet mass to about the 0.3 power, between
  the three-body (1/4) and Hill (1/3) scalings.
- Chaos: at fixed spacing and forcing, t grows as the Lyapunov time to the power 0.53.
  This is the most stable constant in the formula (0.533 +- 0.035 across five disjoint
  training subsets). It is a Lecar-type relation (Lecar et al. 1992 proposed t ~ T_Lyap^1.7
  for asteroids), but far shallower once spacing and forcing are accounted for.
- Forcing: t falls as the eccentricity swing (in crossing units) to the power 1.27; pure
  diffusion to crossing would give 2.
- Starting near crossing costs up to 0.54 dex.

## How sure we are

- Judged on data the formula never saw, against the network on identical systems, with
  bootstrap intervals (RESULTS.md).
- Must-fail control: scramble the four simulation summaries across systems and the formula
  collapses (2.37 dex), so its skill comes from those physical measurements, not from the fit.
- The network's raw score carries a +0.33 dex bias on the pre-registered metric. With that
  bias removed by a recalibration fitted on test-distribution data (which favours the
  network), the network is ahead by 0.026 dex, still inside the pre-registered 0.05 margin.
  The network still ranks systems slightly better (correlation 0.71 against 0.66).
- Selection honesty: earlier law rounds were scored on the whole random set, and the final
  form and fit were chosen on its DEV half with the FINAL-half numbers printed in the same
  run. So the random set is held out from fitting, not from our eyes. The strictly untouched
  test is the fresh N-body set: new systems integrated here after the formula was frozen
  (RESULTS.md, last section).

## What it is not

- Not a formula from the starting orbits alone. We tried that first (DIARY 20:15 to 21:07):
  the best such formula reaches 1.23 and even a flexible tree on starting orbits 1.18. The
  short simulation carries information the starting orbits do not (correlation with the
  truth 0.46 to 0.53 without it, 0.66 to 0.71 with it). That is itself a result: the
  network's advantage over every initial-condition law in the literature is real
  information, and four physical numbers from the simulation capture almost all of it.
- Not below the chaotic floor (0.47 dex from shadow integrations); nothing can be.
- The test set is a random 1,736 of the random dataset (the network costs about 0.4 s per
  system), not all of it.

## What is new

- Tamayo et al. 2020 introduced these simulation summaries for a tree-based classifier;
  Cranmer et al. 2021 used them in a tree-based regressor. Both are black boxes.
- Alford et al. 2026 distilled the network into equations of complexity 26 that are 0.16
  dex worse than the network and not unit-consistent.
- Here: a formula with seven constants, every term a named physical quantity, that matches
  the network on unseen systems, with a must-fail control and an independent N-body check.
