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

## 6. Look-ahead (lookahead_profile.json, lookahead_causal*_site*.json, feats_lookahead_s10.json)

- Recogniser at the last block: variables bound to the network's own move (feature 43, "the pawn it is about to push",
  0.469 against -0.18 for the best static rival; feature 502, "the piece it is about to take", 0.459 against 0.343); no
  variable represents the opponent's reply or the follow-up beyond static structure.
- Probe for "this square is where the network's own follow-up move lands" (ply 3): AUC 0.804 / 0.824 / 0.833 at
  blocks 2 / 4 / 6, against 0.792 for gradient-boosted trees on all 53 static board primitives and 0.64 to 0.67 for the
  untrained network.
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

## 7. Replication in the second network (replicate_744706_s10.json)

22 of 22 LD2 definitions tested (the look-ahead-bound variables, the residual carriers and the 12 strongest) are
matched by a variable of 744706's own sparse dictionary above the line (0.329), using 744706's own line.

## 8. Teachability, live (teach_dict_s10.json)

Five translated variables, plain-English sentences written from the programs alone, 8 held-out positions each, a
fresh DeepSeek V4.1 Flash call (no thinking; a run with low thinking did not return within 11 minutes and was stopped)
names the 3 squares where the variable is most active: hit rates 0.00, 0.08 and 0.13 against chance 0.05, rival
sentences 0.04 to 0.25, and 3 of 10 answers unparseable. The test failed: this reader cannot apply a square-level
definition to positions given as FEN text. Teachability stays untested; it needs a reader that is shown the board.
