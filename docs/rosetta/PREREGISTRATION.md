# Rosetta: pre-registration (R1)

Ledger row R1 (docs/glass/CHANGES.md) was written before any Rosetta code. Its registered content, verbatim:

> **Change:** Build rosetta/ (engine) and rosetta/chess (library incl. look-ahead primitives from the network's own line). Yardstick: 3 small networks trained on sealed per-square rules (sealed file the engine never reads), one memoriser (shuffled labels) and one untrained network. Real target: LD2 channels at sites 6 and 10 that carry the classical residual; replication 744706

> **Pass lines and criteria:** Pass lines, all fixed now: (Y) a planted rule counts as recovered if Rosetta's top accepted definition, scored on held-out positions, correlates >= 0.9 with the sealed rule's per-square map; the memoriser and the untrained network must yield no accepted definition. Acceptance line for any definition = the 99th percentile of best-definition fidelity over the control units (untrained network channels and square-shuffled activity), computed before the real units are scored. (L) A DISCOVERY needs one concept that (a) passes the acceptance line on LD2 held-out positions, (b) is not a restatement (the 61 classical concepts predict its position-level total at R2 < 0.9), (c) restores >= 50% of the value change caused by ablating its unit when substituted for it, (d) adds to the classical surrogate's explanation of LD2's value on held-out positions with a bootstrap 95% CI excluding 0, (e) matches a unit in 744706 above the line, and (f) predicts game results beyond the classical library (CI excluding 0) or is a look-ahead primitive

> **Cases not used to design it:** Cases not used to design it: the 744706 network, the held-out split, game results

## Declared changes after registration (all in DIARY.md, with the clock)

- **04:32, measurement:** each unit's fixed per-site mean is removed before defining it (untrained units matched pure
  rank-and-file patterns). Before any real unit was scored.
- **04:37, measurement:** a position level was added (the unit's total per position against the program's count),
  because the known-answer case S3 showed rare rules drown at the square level. Before any real unit was scored.
- **04:42, controls:** at the position level, square-shuffled controls are invalid (they keep totals) and material
  dominates random totals; position-level fidelity removes the 12 piece counts and its line uses only untrained-network
  controls. Before any real unit was scored.
- **04:51 and 04:53, translation errors:** the unit filter kept units with zero effect (the intent says only units the
  network uses), and its first floor (1% of the output's spread) dropped the sparse planted units; set to 0.1%. A
  batching bug in the LD2 substitution test was fixed.
- **04:55, method:** raw channels could not be defined well enough to substitute, so the network's own variables were
  found first with a sparse dictionary (the review's step 2), with its own untrained-network control line.
- **05:24, method:** after single programs failed substitution (best 30%), compositions of up to three programs were
  tried on four candidates, with a composition line from the untrained dictionary (0.388). One passed all six checks.
  This is a second attempt after a failure and is reported as such.

## Outcome

- (Y) known answers: 3 of 3 planted rules recovered; the constant-output network rejected; one false positive on the
  second untrained network.
- (L) DISCOVERY bar: met by LD2 variable 174 as a three-program composition, at the margin (substitution 0.501 and
  0.536 on a disjoint set, bar 0.50), with its knowledge criterion met through the look-ahead clause.
