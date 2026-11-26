# Rosetta: results

All numbers come from files in `results/rosetta/`. Held-out means positions never used to fit or choose anything in
that step. Fidelity is the Pearson correlation between a program's map and a unit's activity on held-out positions
(per square, after removing the unit's fixed spatial pattern), or between counts per position with the 12 piece counts
removed ("position level"). Pass lines come from control units, computed before the units they judge.

## 1. Known answers: blind recovery of planted rules (yard_score.json, yard_engine.json, yard_controls.json)

Three small networks trained on sealed per-square rules (the engine never reads `rosetta/sealed/`), a network with a
constant output, and a second untrained network. Lines: per square 0.401, position level 0.586.

| network | held-out skill of the network | sealed rule | Rosetta's answer, blind | match with the sealed rule |
|---|---|---|---|---|
| S1 | R2 0.990 | own_N and pawn_att_us and safe_from_opp_pawns | own_N and pawn_att_us and safe_from_opp_pawns | 1.000 |
| S2 | R2 0.656 | opp_king_zone and att_us >= 1 and att_them <= 1 | not opp_R and min_att_us>=1 and min_att_them>=100 | 0.988 |
| S3 | R2 0.998 | own_P and passed_pawn and rel_rank >= 4 | own_P and rel_rank>=4 and passed_pawn | 1.000 |
| constant output (must fail) | - | none | no definition | pass |
| second untrained net (must fail) | - | none | 1 of 7 units over the position-level line (0.667) | fail, a false positive of the line |

S2's answer is an equivalent formulation: "a square we attack whose only enemy attacker is the king" is a square in
the enemy king's zone that no other enemy piece covers.

## 2. Leela LD2, raw channels of the last block (ld2_units_site10.json, ld2_define_site10.json, ld2_confirm_site10.json)

- Textbook baseline (the record's gradient-boosted surrogate on 61 classical concepts): held-out R2 0.778 of LD2's
  value; the residual is 22.2% of its variance.
- Channels chosen by correlation with that residual among those whose ablation moves the value: 15, 80, 22, 122, 14,
  103, 76, 89 (the record's causal channels 22 and 15 come back unprompted). Lines: per square 0.166, position 0.434.
- Six channels and the residual direction defined above the line; the two strongest: channel 122 "not opp_P and
  outnumbered_opp and la_m1_to" (0.377) and channel 80 "opp_Q and att_us<=1 and not la_m1_to" (0.369). Channel 22: 0.125.
- Every single-channel definition fails substitution (restores 0.4% to 4% of the ablation effect). Four-program
  compositions reach 0.46 at best. All seven together add +0.47% of value variance (CI 0.12% to 0.82%) and nothing
  significant for game results.
- The cheap model at volume (18 live DeepSeek V4.1 Flash calls, 45 usable expressions) never beat the exhaustive
  search (ld2_propose_site10.json).

## 3. The network's own variables: a dictionary of LD2's last block (dict_s10_w*.json, feats_*.json)

- Sparse dictionary (top-k autoencoder, 512 variables, 8 active per square) rebuilds 85.5% of the block's activity
  (untrained-network control: 68%).
- 485 of 512 variables get a program; held-out fidelity median 0.15; at or above 0.3: 114; 0.5: 49; 0.7: 14; 0.9: 3.
- Examples: "own_N and not pinned" 0.944; "own_K and own_att_by_lower" (in check) 0.843; "rel_rank>=3 and
  safe_from_opp_pawns and mob_us>=4" (a hole that four or more of our moves reach) 0.672; "not opp_P and
  outnumbered_opp and la_m1_to" 0.684; "dist_own_king<=1 and piece_reach>=5 and hanging_opp" 0.705.
- Residual-carrying variables (the 12 most correlated with the textbook residual, ablation above the median): four
  defined above the line (0.33), none at the position level; they are frequent (49% to 74% of positions).

## 4. Glass model of the layer (glass_s10.json)

LD2's last block rebuilt from the variables, then run through LD2's own heads, 1,000 held-out positions:

| rebuilt from | top move same as the network | value R2 against the network |
|---|---|---|
| the true sparse codes (ceiling) | 0.695 | 0.990 |
| Rosetta's 512 programs | 0.413 | 0.062 |
| each variable's average map only (knows nothing) | 0.047 | -0.022 |

## 5. Residual curve and knowledge (residual_curve_s10.json, fresh_knowledge_s10.json)

- The 113 variables with fidelity at or above 0.3, as program counts added to the textbook surrogate: value +1.46%
  (CI 0.93% to 2.02%), that is 6.6% of the 22.2% the textbook misses.
- Game results (held-out positions of the same games' split): +1.31% (CI 0.16% to 2.45%), against +3.64% for LD2's
  own value (CI 2.20% to 5.13%).
- Forward selection on a training sub-split, scored held-out: see the file (first concepts added: an enemy knight we
  barely control, squares two or more of our moves reach that no enemy pawn covers, the piece our move takes).
- Fresh games (fresh_knowledge_s10.json): 16,000 positions from Lichess games never used anywhere in Rosetta (Kepler's
  fresh set), fit on the 16,000 original game positions. The textbook library reaches R2 0.149 for the result; the 114
  translated variables add +0.65% (CI +0.13% to +1.17%); LD2's own value adds +3.58% (CI +2.89% to +4.22%). The
  translations carry about a fifth of what the network knows about who wins beyond the textbook, on games it never saw.
- Concentration (knowledge_select_s10.json): the five variables with the largest single gains on the original
  held-out games do not add significantly on the fresh games, alone (best +0.18%, CI -0.06% to +0.42%) or together
  (+0.22%, CI -0.14% to +0.56%). The knowledge is spread over many variables.

## 6. Look-ahead (lookahead_profile.json, lookahead_causal*_site*.json, feats_lookahead_s10.json)

- Recogniser at the last block: variables bound to the network's own move (feature 43, "the pawn it is about to push",
  0.469 against -0.18 for the best static rival; feature 502, "the piece it is about to take", 0.459 against 0.343); no
  variable represents the opponent's reply or the follow-up beyond static structure.
- Probe for "this square is where the network's own follow-up move lands" (ply 3): AUC 0.804 / 0.824 / 0.833 at
  blocks 2 / 4 / 6, against 0.792 for gradient-boosted trees on all 53 static board primitives and 0.64 to 0.67 for the
  untrained network.
- The same probe on the transformer t1 (lookahead_profile_t1.json; t1's own line; board-only baseline 0.798):
  AUC 0.851 / 0.894 / 0.911 at blocks 2 / 4 / 6 (untrained t1: 0.61 to 0.65). The transformer carries its follow-up
  move much more strongly than LD2 (0.833 at block 6).
- Causal test (mean-ablate one square's state, measure the drop in the log-probability of the network's own move):

| network, block | follow-up square | matched control (occupancy, attacked by the moved piece, attack counts) | difference, 95% CI |
|---|---|---|---|
| LD2, 4 | 0.024 | 0.008 | +0.016 (-0.002 to +0.036) |
| LD2, 6 | 0.016 | 0.011 | +0.006 (-0.015 to +0.027) |
| LD2, 8 (tight control, first version) | -0.021 | 0.013 | -0.033 (-0.057 to -0.013) |
| 744706, 4 | 0.042 | 0.019 | +0.023 (+0.004 to +0.041) |
| 744706, 6 (tight control, first version) | 0.034 | 0.011 | +0.023 (+0.007 to +0.040) |
| t1 (transformer), 4 | 0.081 | 0.007 | +0.075 (+0.049 to +0.104) |
| t1 (transformer), 6 | 0.085 | -0.001 | +0.086 (+0.061 to +0.114) |
| t1 (transformer), 8 | 0.048 | -0.006 | +0.054 (+0.035 to +0.079) |

  For scale, ablating the destination of the current move itself costs 1.6 to 3.4 (t1: 1.5 to 2.4). In t1 the effect
  holds where the follow-up square is attacked by the moved piece (+0.16 to +0.20) and where it is not (+0.04 to +0.06,
  all CIs above zero); in LD2 only the first (block 6: +0.079, CI +0.013 to +0.152, n 268). The transformer is also the
  stronger network (mate in one: 49 of 50 against LD2's 44, results/kepler/chess/t1_check.json), so architecture and
  strength are confounded. A first run with a looser control
  (occupancy only) showed +0.089 (CI 0.059 to 0.120) at LD2 block 4; tightening the control removed most of it.

- Split by the follow-up move's type (lookahead_causal2_*_bytype.json, lookahead_causal2_t1_bytype.json; exploratory,
  after the main test), cost minus matched control, 95% CI:

| network, block | check | capture | quiet |
|---|---|---|---|
| t1, 4 | +0.143 (+0.041 to +0.267), n 80 | +0.911 (+0.501 to +1.384), n 40 | +0.025 (+0.012 to +0.040), n 791 |
| t1, 6 | +0.182 (+0.117 to +0.255), n 80 | +0.822 (+0.440 to +1.273), n 40 | +0.039 (+0.025 to +0.054), n 791 |
| 744706, 4 | +0.051 (+0.000 to +0.108), n 114 | +0.241 (+0.048 to +0.486), n 59 | +0.011 (-0.006 to +0.027), n 1,327 |
| 744706, 6 | +0.050 (-0.004 to +0.111), n 140 | +0.262 (+0.062 to +0.505), n 51 | +0.010 (-0.005 to +0.024), n 1,309 |
| LD2, 4 | +0.050 (+0.009 to +0.097), n 143 | +0.138 (+0.011 to +0.296), n 65 | +0.006 (-0.013 to +0.026), n 1,292 |
| LD2, 6 | +0.108 (+0.051 to +0.169), n 154 | +0.069 (-0.047 to +0.222), n 73 | -0.011 (-0.032 to +0.012), n 1,273 |

  The convolutional networks look ahead only along forcing follow-ups; the transformer along all three kinds. t1's
  probe at block 8: AUC 0.922.

- Confirmation on 2,000 fresh positions (ledger R2, predictions registered before the run; lookahead_fresh_*.json):

| network, block | all follow-ups | forcing (check or capture) | quiet |
|---|---|---|---|
| t1 (transformer), 4 | +0.052 (+0.039 to +0.067) | +0.262 (+0.162 to +0.381), n 196 | +0.029 (+0.019 to +0.039), n 1801 |
| t1 (transformer), 6 | +0.062 (+0.048 to +0.077) | +0.310 (+0.213 to +0.417), n 196 | +0.035 (+0.025 to +0.045), n 1801 |
| 744706 (convolutional), 4 | +0.030 (+0.016 to +0.045) | +0.102 (+0.043 to +0.171), n 188 | +0.023 (+0.009 to +0.038), n 1812 |
| 744706 (convolutional), 6 | +0.028 (+0.017 to +0.040) | +0.080 (+0.027 to +0.144), n 188 | +0.023 (+0.012 to +0.035), n 1812 |
| LD2 (convolutional), 4 | +0.010 (-0.006 to +0.027) | +0.087 (+0.041 to +0.136), n 193 | +0.001 (-0.016 to +0.020), n 1732 |
| LD2 (convolutional), 6 | +0.013 (-0.005 to +0.032) | +0.085 (+0.042 to +0.131), n 193 | +0.005 (-0.016 to +0.026), n 1732 |

  Prediction 1 (t1 quiet above zero at both blocks): held. Prediction 3 (forcing above zero in all three): held.
  Prediction 2 (each convolutional net's quiet upper bound below t1's estimate at the same block): held for LD2 at both
  blocks and for 744706 at block 6 (0.035 against 0.035, by a hair); failed for 744706 at block 4 (0.038 against 0.029).

## 6b. The DISCOVERY battery (battery_s10.json, compose_sub_K3_s10.json, rep_compose_K3_s10.json, sub_check_f174_s10.json)

Ten translated variables through the six pre-registered checks (a fidelity above the line, b not a restatement, c
substitution restores at least 50%, d value gain CI above zero, e replicates in 744706, f game-results gain CI above
zero or a look-ahead primitive). Single programs: none passes; substitution restores 21% to 47%. Compositions of three
programs (line for compositions from the untrained dictionary: 0.388), four candidates:

| variable | fidelity | restatement R2 | substitution restores | value gain (CI) | 744706 | results gain (CI) | look-ahead | all six |
|---|---|---|---|---|---|---|---|---|
| 174 | 0.759 | 0.72 | 0.501 (0.43 to 0.57); disjoint set 0.536 (0.48 to 0.58) | +0.57% (+0.21% to +0.93%) | 0.699 | +0.61% (-0.23% to +1.41%) | yes | **yes, at the margin** |
| 502 | 0.499 | 0.40 | 0.353 | +0.30% (-0.00% to +0.61%) | 0.599 | +0.15% (-0.75% to +1.00%) | yes | no |
| 43 | 0.491 | 0.22 | 0.328 | -0.06% | 0.354 (below) | -0.45% | yes | no |
| 166 | 0.420 | 0.84 | 0.347 | +0.17% (-0.04% to +0.38%) | 0.349 (below) | -0.72% | no | no |

Variable 174's three programs: "not opp_P and outnumbered_opp and la_m1_to" (weight 1.78), "min_att_us<=3 and
hanging_opp and not check_from_us" (0.63), "not empty and not opp_N and la_m1_to" (0.51).

## 6c. Rosetta inside the transformer t1, block 6 (t1_rosetta_site6.json)

Sparse dictionary (512 variables) of t1's block 6, scanned against t1's own line; line from an untrained t1: 0.041.

| variable | definition (full library) | fidelity | static rival | fidelity |
|---|---|---|---|---|
| 245 | not pawn_att_them and outnumbered_opp and la_m1_to | 0.728 | not opp_P and mob_us>=1 and outnumbered_opp | 0.686 |
| 354 | opp_K and la_m1_attacks (the move gives check) | 0.665 | opp_K and dist_own_king<=6 and not outnumbered_opp | 0.594 |
| 335 | check_from_us and la_m1_to and not la_m2_to | 0.560 | rel_rank>=6 and check_from_us and control>=1 | 0.560 |
| 341, 444, 110, 291, 353 | static definitions (no look-ahead primitive wins) | 0.64 to 0.80 | same | same |

The capture concept of LD2's variable 174 appears in all three networks (LD2 174, 744706 455, t1 245). No variable is
defined by the reply or the follow-up move.

Substitution inside t1 (t1_sub_site6.json): variable 245 as three programs ("not pawn_att_them and outnumbered_opp and
la_m1_to" 2.90, "not opp_P and not pawn_att_us and outnumbered_opp" 1.82, "hanging_opp and mob_us>=1 and not
la_m1_to" 1.13), fidelity 0.800, restores 73.1% of the variable's effect on t1's value (bar 50%).

Substitution inside 744706 (sub_744706_capture.json): its variable 455, the best match to LD2's composed variable 174,
defined from 744706's own line as three programs ("not empty and control>=0 and la_m1_to" 1.80, "opp_R and semiopen_us
and outnumbered_opp" 1.87, "not pawn_att_us and piece_reach>=5 and hanging_opp" 1.22), fidelity 0.732, restores 54.9%
of the variable's effect on 744706's value (mean effect of removing it 0.002).

## 7. Replication in the second network (replicate_744706_s10.json)

22 of 22 LD2 definitions tested (the look-ahead-bound variables, the residual carriers and the 12 strongest) are
matched by a variable of 744706's own sparse dictionary above the line (0.329), using 744706's own line.

## 8. Teachability, live (teach_dict_s10.json)

Five translated variables, plain-English sentences written from the programs alone, 8 held-out positions each, a
fresh DeepSeek V4.1 Flash call (no thinking; a run with low thinking did not return within 11 minutes and was stopped)
names the 3 squares where the variable is most active: hit rates 0.00, 0.08 and 0.13 against chance 0.05, rival
sentences 0.04 to 0.25, and 3 of 10 answers unparseable. The test failed: this reader cannot apply a square-level
definition to positions given as FEN text. Second try with the boards drawn as 8 x 8 grids
(teach_dict_boards_none_s10.json): "we are in check" 0.21 (rival 0.04); the others 0.00 to 0.13 (chance 0.05). A run
with low thinking gave no answer in about ten minutes and was stopped. Teachability stays essentially untested: a
reader strong enough to apply square-level rules (a frontier model, or people) is needed.

## 9. Where the evaluation and the extra knowledge pass (glass2_*.json, fresh_where_s10.json, top_pos_s10.json, line_swing*_s10.json)

Exploratory, run after the main tests (diary 06:10 to 06:22). The 512 variables are ranked by attribution: activation
times the gradient of LD2's value along each variable's decoder direction, on 600 training positions.

The glass model with the K most important variables put back at their true values and every other variable at its
single program (1,000 held-out positions):

| K | value R2 against the network | top move same as the network |
|---|---|---|
| 0 (the glass model of section 4) | 0.062 | 0.413 |
| 8 | 0.476 | 0.392 |
| 16 | 0.697 | 0.400 |
| 32 | 0.846 | 0.385 |
| 64 | 0.918 | 0.402 |
| 128 | 0.969 | 0.513 |
| 256 | 0.987 | 0.611 |
| 512 (true codes) | 0.990 | 0.695 |

The evaluation passes through a few variables; the move choice is spread over many. Richer definitions (sums of three
programs) for the 64 most important variables do not repair it: value R2 0.074, top move 0.399.

Fresh games (16,000 positions never used in Rosetta), textbook concepts plus the board-wide totals of LD2's own
variables (true codes), trained on the original games:

| added to the textbook concepts | gain in R2 for the game result | 95% CI | share of LD2's own gain |
|---|---|---|---|
| LD2's value (ceiling) | +3.58% | 2.89 to 4.26% | 100% |
| totals of the 8 most important variables | +2.54% | 1.93 to 3.25% | 71% |
| totals of the 16 most important | +3.25% | 2.59 to 3.94% | 91% |
| totals of the 64 most important | +2.99% | 2.14 to 3.78% | 83% |
| totals of all 512 | +3.66% | 2.79 to 4.47% | 102% |
| control: 16 random variables (three draws) | +2.22%, +0.60%, +0.48% | | 62%, 17%, 13% |
| control: the 16 least important | +0.16% | -0.07 to 0.40% | 4.5% |

(The 62% random draw contains the variables ranked 5th and 6th.) The network's knowledge beyond the textbook passes
mostly through about 16 of its 512 variables at block 10. Rosetta cannot define most of them yet: as counted programs
at the position level (material removed), only 2 of the 16 pass the untrained-network line of 0.456 (variables 77 and
228, which are also well defined square by square); the evaluation carriers (434, 415, 303, 304, 192 and others) reach
0.11 to 0.36. Their board totals track LD2's value (434: -0.69; 304: +0.72) and the part of it textbook concepts miss
(434: -0.46; 415: +0.43).

The material LD2's own line wins (registered before the run, diary 06:22): side to move's material balance after 1, 2
and 3 plies of LD2's greedy line, minus now. On fresh games, textbook concepts plus these three numbers recover +1.18%
(CI 0.78 to 1.60%), 33% of LD2's gain; the same numbers along random legal moves recover +0.02% (CI -0.25 to 0.32%).
Adding them to LD2's value adds nothing (102%), and adding them to the 16 variables' totals adds little (+3.32% against
+3.25%). Yet no single one of the 16 carries the swing (held-out correlations -0.12 to +0.13), and the totals predict
the swing itself only a little beyond textbook concepts (R2 0.34 from textbook concepts, 0.38 with the 16 totals, 0.42
with all 512). A third of what LD2 knows beyond the textbook is what its own forcing line wins; the 16 variables hold
that knowledge for predicting the result without holding the swing as a number.

Win, draw and loss (carriers_wdl_s10.json, 3,000 held-out positions; correlation of each variable's board total with
LD2's own probabilities). The 16 split into "we are winning" variables (304: win +0.66, loss -0.68; 493: win +0.62;
193, 415, 50, 0, 184: win +0.34 to +0.46) and "we are losing" variables (434: loss +0.78; 192: +0.59; 303: +0.54; 249:
+0.53). Three also track drawishness beyond who is ahead (draw probability with W - L removed linearly): 434 -0.49,
493 -0.40 (decisive positions), 508 +0.34 (drawish). The evaluation is held as separate win, loss and draw signals.

Replication in 744706 (registered before the run, DIARY 06:27; fresh_where_744706_s10.json). Its own dictionary at
block 10 and its own attribution ranking, the same fresh games:

| added to the textbook concepts | gain | 95% CI | share of 744706's own gain |
|---|---|---|---|
| 744706's value (ceiling) | +4.10% | 3.42 to 4.77% | 100% |
| totals of its 8 most important variables | +3.07% | 2.34 to 3.84% | 75% |
| totals of its 16 most important | +3.19% | 2.41 to 3.95% | 78% |
| totals of its 64 most important | +3.77% | 2.93 to 4.54% | 92% |
| totals of all 512 | +3.44% | 2.63 to 4.23% | 84% |
| control: 16 random variables (three draws) | +1.44%, +1.51%, +1.59% | | 35%, 37%, 39% |
| control: the 16 least important | +0.04% | -0.11 to 0.19% | 1% |

Replicates: the top 16 beat every control and recover more than half of the network's gain.

Static-tactics control for the line swing (registered before the run, DIARY 06:31; line_swing_static_s10.json).
Value-weighted totals of outnumbered and attacked-by-lower pieces, both sides (the textbook concepts already hold
hanging pieces, hanging value, captures available and the best free capture): static tactics add +0.23% (CI -0.10 to
+0.56%); the swing adds +0.61% (CI +0.22 to +1.01%) over textbook plus static tactics, so it is not static tactics.
Together the two score below the swing alone (+0.84% against +1.18%), so these fitted gains carry noise: the swing's
share of LD2's extra knowledge is better given as a sixth to a third.

Line swing in 744706 (registered before the run, DIARY 06:33; line_swing_744706_s10.json). Along 744706's own 3-ply
line: +0.87% over textbook concepts (CI +0.46 to +1.33%), 21% of its own gain (+4.10%); along random legal moves +0.02%
(CI -0.25 to +0.32%); static tactics +0.22% (CI -0.10 to +0.56%); the swing over textbook plus static tactics +0.50%
(CI +0.09 to +0.92%). Replicates: in both networks the material won along the network's own line carries part of its
knowledge beyond the textbook (LD2 17% to 33%, 744706 12% to 21%), and random lines carry none.

How much of the 16 the textbook restates (carriers_textbook_s10.json; board totals, held-out positions, the 61
textbook concepts in gradient-boosted trees): R2 0.19 (variable 172) to 0.91 (77); the evaluation carriers 434 0.56
(most correlated: material -0.31, their space +0.20), 304 0.65 (material +0.41, our mobility +0.40, attacks on our king
zone -0.36), 303 0.59 (their space, their rooks on open files, their castling rights), 192 0.52, 415 0.51, 461 0.32.
Each is partly a blend of familiar quantities and 35% to 80% something the textbook does not hold: the target for the
next build.
