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

**04:57. Yardstick, final (blind; the sealed file read only by the scorer).** Planted rules recovered: S1 1.000,
S2 0.988 (Rosetta's equivalent: "squares we attack whose only enemy attacker is the king"), S3 1.000: 3 of 3. Must-fail:
the constant-output network, 0 definitions (pass); a second untrained network, 1 of 7 units over the position-level
line (0.667 against 0.586). The line is the maximum over 32 control units, so a chance pass on one of several units
per network is expected; recorded as a false positive of the line, not hidden. (results/rosetta/yard_score.json)

**04:55. LD2 raw channels, confirmation.** Every single-channel definition failed substitution (restores 0.4% to 4%
of the ablation effect: a 0.37 correlation is not a stand-in for the unit); channel 15's definition is a restatement
(R2 0.95 from the library). All seven together add +0.47% of LD2's value variance to the textbook surrogate (CI
0.12% to 0.82%), and nothing measurable for game results. Composed definitions (four programs) reach 0.46 at best.
Conclusion: raw channels of this dense network are mixtures; the review's step 2 (find the network's own variables)
is needed before definition.

**04:58. Variable finder added.** A top-k sparse autoencoder (512 features, 8 active per square) on LD2's last block
rebuilds 85.5% of its activity variance. Control: the same on the untrained LD2 (68%).

**05:01. Residual-carrying variables.** The 12 features most correlated with the textbook residual (and moving the
value when removed) are frequent (active in 49% to 74% of positions). Four are defined above the line (0.33): mostly
"not an enemy minor or major piece" and pawn-support patterns; none at the position level. The residual is not carried
by crisp, rare variables at the last block.

**05:01. Search-procedure recogniser.** All 512 features scored against the network's own line (top move, the reply,
the follow-up), each best match then defined twice: with the full library and with every look-ahead primitive removed
(the static rival). Results (results/rosetta/feats_lookahead_s10.json):
- feature 43: "own_P and control>=-1 and la_m1_from" 0.469, static rival -0.18: the pawn the network is about to push;
- feature 502: "not empty and not open_file and la_m1_to" 0.459, static rival 0.343: the piece it is about to take;
- feature 174: "not opp_P and outnumbered_opp and la_m1_to" 0.684, static rival 0.662: an outnumbered enemy piece on
  the square its move lands on, mostly static;
- the reply and follow-up primitives (la_m2, la_m3) only reach 0.13 to 0.19, and every such feature's best definition
  is static (enemy pieces where a reply starts). So at the last block LD2 binds its own move to its reason, but no
  variable represents the opponent's reply or its follow-up above what static structure explains.

**05:02. Next: the whole layer.** Translating all 512 variables into programs, then rebuilding the layer from the
translations and running the rest of LD2 on it (a glass model of the layer): how much of its evaluation and move choice
survive, and how much of what the textbook misses the translation reproduces.

**05:08. The whole layer translated (results/rosetta/dict_s10_w*.json).** 485 of LD2's 512 last-block variables get a
program; held-out fidelity median 0.15; 114 at or above 0.3, 49 at or above 0.5, 14 at or above 0.7, 3 at or above
0.9. The strongest are piece-identity variables ("own_N and not pinned", 0.94); real chess concepts among them:
"own_K and own_att_by_lower" (in check, 0.84), "rel_rank>=3 and safe_from_opp_pawns and mob_us>=4" (a hole in the enemy
half that four or more of our moves reach, 0.67), "not opp_P and outnumbered_opp and la_m1_to" (0.68).

**05:09. Glass model of the layer (results/rosetta/glass_s10.json).** LD2's last block rebuilt from (a) the true
sparse codes, (b) Rosetta's 512 translations, (c) each variable's mean map only, then run through LD2's own heads
(1,000 held-out positions): top-move agreement with the network 0.695 / 0.413 / 0.047; value R2 against the network
0.990 / 0.062 / -0.022. The translation keeps much of the move logic and almost none of the evaluation.

**05:10. Cheap model at volume (results/rosetta/ld2_propose_site10.json).** 18 live calls returned 45 usable
expressions; none beat the exhaustive search (channel 80: 0.365 against 0.368; channel 76: 0.303 against 0.478).

**05:11. Residual curve (results/rosetta/residual_curve_s10.json).** The 113 translated variables with fidelity at
or above 0.3, as program counts, added to the record's surrogate: LD2 value +1.46% of variance (CI 0.93% to 2.02%),
so 6.6% of the 22.2% the textbook misses; game results +1.31% (CI 0.16% to 2.45%), against +3.64% for LD2's own value
(CI 2.20% to 5.13%): the translations carry about a third of what the network knows about who wins beyond the
textbook. (Forward selection still running.)

**05:10. Look-ahead profile (results/rosetta/lookahead_profile.json).** Linear probe on LD2's square vector for "this
square is where the network's own follow-up move (ply 3) lands": AUC 0.804 / 0.824 / 0.833 at blocks 2 / 4 / 6,
against 0.792 for gradient-boosted trees on all 53 static board primitives and 0.64 for the untrained network. The
network's state carries its own future move beyond what the board shows, and more so with depth.

**05:11. Causal look-ahead, first run (block 4, 1,200 held-out positions).** Mean-ablating the network's state on
the follow-up move's destination square (never a square of the first two moves) lowers the log-probability of its
current move by 0.097, against 0.008 for a control square of the same occupancy (difference 0.089, CI 0.059 to 0.120;
for reference, the current move's own destination: 2.93). Jenner et al. (2024) found look-ahead in Leela's
transformer and built their method on its squares-as-tokens structure; LD2 is a convolutional network. Tightening
the control (also matched on being attacked by the moved piece after the move, and on attack counts) and running
blocks 4, 6, 8 and the second network now.
