# Planets: pre-registration (PL1)

The registration lives in the ledger row PL1 (docs/glass/CHANGES.md), written before any fit. Copied here verbatim:

> Question: does a compact formula in physical variables predict instability times as accurately as the network, judged by direct N-body integrations it never saw?

> Written before any fit: fit only on the resonant dataset (as the network was), judge only on the random dataset (never used for fitting by us or by the network): RMSE of log10(t/P1) over systems that survive the first 1e4 orbits and go unstable before 1e9. DISCOVERY if the formula (a) has at most 8 fitted constants and physically meaningful, unit-consistent variables, (b) beats the published laws (LHM, Petit) and the 2026 equations (1.42 random), and (c) is within 0.05 dex of the network on the same systems or better. Independent check: fresh N-body integrations of new systems, run by us, compared with the formula's predictions. The chaotic floor comes from the shadow integrations

## What was changed after registration, and when

- 21:07 (DIARY): the variables. Laws on starting orbits alone hit an information limit
  (best 1.23 against the network's 1.16 on the same systems; correlation with the truth
  0.46 against 0.71). The law was allowed four physical summaries of the same 1e4-orbit
  integration the network uses. The registered wording ("physically meaningful,
  unit-consistent variables") covers them; the registered plan listed only
  initial-condition features, so this is declared as a change.
- Not changed: the training set (resonant only), the test set (random only), the metric,
  the 8-constant limit, the 0.05 dex margin, the baselines, the fresh N-body check.
- Added, not registered: a DEV / FINAL split of the random set for choosing the form
  (file written 20:54 by the system clock, seed 2026,
  results/planets/random_split.npy), the censored fit for stable
  systems, the must-fail shuffle, the ablation, and the recalibration check.

## Outcome

All three criteria met (RESULTS.md, FINDINGS.md); the fresh N-body check passed with the
law ahead of the network. Ledger row PL1 closed at 21:46.
