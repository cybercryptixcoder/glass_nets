# Rosetta diary

Times come from the system clock (`TZ=America/New_York date`), copied from its output.

## 2026-11-26

**04:09.** Sprint start (Shreyas, 04:08: build Rosetta, the Hail Mary; three hours; then point it at targets and make
discoveries). Read the review (docs/review/HANDOFF.md, BRIEF.md, NEXT_BUILD.md sections 3.3, 3.8, 6). Branch
`rosetta` cut from `review-next-build`. Intent in docs/rosetta/INTENT.md; pre-registration in ledger row R1
(docs/glass/CHANGES.md) before any code.

**Design choice.** The review's central gaps are: a supplied vocabulary, interpreting the decision instead of the
computation, and no test against disguised copies. Rosetta answers them in one loop: read the network's own units,
define each with a short program in a restricted language (no weights, so a copy cannot pass), substitute the program
back into the network, and grow the vocabulary with what is accepted. The chess library carries look-ahead primitives
taken from the network's own three-ply line, so search inside the network can be recognised.

**04:18.** Library built for all 20,000 positions (16,000 game positions, 4,000 puzzles): 61 per-square primitives
(Kepler's 42 descriptors, 11 threat/mobility/check primitives, 8 look-ahead primitives from LD2's own line,
computed by running LD2 three plies; its top move agrees with the stored one on 100%). 18 s.

**04:30.** Known-answer networks trained (small 4-block CNNs on the 13 piece planes): S1 held-out R2 0.990, S2 0.656
(only partly learned), S3 0.998. The memoriser did not memorise (train R2 0.000 with shuffled labels at 120 epochs):
it is a flat network, a weaker must-fail control than planned. Recorded as such.

**04:32. Measurement change 1, before any real unit was scored.** Untrained networks' channels reached 0.5 to 0.6
per-square fidelity with rank-and-file programs: a fixed spatial bias, the same for every input. Each unit's fixed
per-square mean is now removed before defining it, for controls and real units alike. Line after the change: 0.348.

**04:34.** The record's classical surrogate (gradient-boosted trees on 61 concepts) reproduced: held-out R2 0.778 of
LD2's value (record: 78%).

**04:37. Measurement change 2, from the known-answer cases (their purpose).** Blind, Rosetta's top program for S1's
readout direction was exactly the sealed rule, but per-square correlation only reached 0.43, and for S3 the right rule
did not win at all: a rare rule (an advanced passed pawn is under 1% of squares) is swamped by the unit's other
activity. Added the level the rest of the network can pool: the unit's total per position against the program's count
per position, with its own search and its own control line. On S3 the position-level search returns
"own_P and rel_rank>=4 and passed_pawn" at held-out 0.962 (before the sealed file was read). Controls and planted
networks rerun with both levels; LD2 not yet scored.

**04:42. Measurement change 3 (a control error, fixed before LD2 was scored).** At the position level the
square-shuffled controls are not controls (shuffling squares inside a position leaves its total unchanged), and the
untrained networks' totals were matched at 0.958 by plain piece counts: random functions of the board are mostly
material. Position-level fidelity now removes the 12 piece counts from both the unit's total and the program's count
(a definition must explain what is not material), and the position-level line uses only untrained-network controls.
Check after the change, before the sealed file was read: S3's readout direction still gets
"own_P and rel_rank>=4 and passed_pawn" at 0.955; an untrained network's readout gets -0.51 and its top channel 0.46.

**04:48. LD2, first pass (site 10, the last block).** The record's surrogate leaves 22% of LD2's value unexplained.
Rosetta ranked LD2's 128 last-block channels by correlation with that residual among those whose ablation moves the
value (above the median): channels 15, 80, 22, 122, 14, 103, 76, 89 (the record's two causal channels, 22 and 15,
came back unprompted). Lines from controls (an untrained LD2 with its weights permuted, plus square-shuffled units):
per-square 0.166, position-level 0.434. Defined above the per-square line: 15, 80, 122, 14, 103, 76 and the
residual direction; 22 and 89 not. The two strongest:
- channel 122: "not opp_P and outnumbered_opp and la_m1_to" (0.377): the square the network's own best move goes to,
  when an enemy piece there has more attackers than defenders;
- channel 80: "opp_Q and att_us<=1 and not la_m1_to" (0.369): the enemy queen, attacked at most once, not our target.

**04:51. Two translation errors found and fixed.**
- The memoriser (whose output is exactly constant: std 0.0) got 3 accepted definitions, because the unit filter
  kept the top 6 units by ablation effect even when every effect was zero. The intent says a variable is kept only
  if the network uses it. Fix: keep a unit only if ablating it moves the output by at least 1% of the output's spread
  across positions (a network with a constant output uses nothing). Rerun.
- The LD2 substitution test crashed (batching mismatch between the network pass and the program map). Fixed; rerun.
