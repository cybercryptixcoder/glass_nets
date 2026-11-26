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
