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

**05:15. Look-ahead, both controls on the same positions (lookahead_causal2_*.json).** With the control square also
matched on being attacked by the moved piece and on attack counts, the effect shrinks: LD2 block 4 +0.016 (CI -0.002 to
+0.036), block 6 +0.006 (CI -0.015 to +0.027); 744706 block 4 +0.023 (CI +0.004 to +0.041), block 6 (tight control,
first version) +0.023 (CI +0.007 to +0.040). Split at LD2 block 6: where the follow-up square is attacked by the moved
piece, +0.079 (CI +0.013 to +0.152, n 268); elsewhere -0.011. The first, looser result (+0.089) was mostly the moved
piece's attack pattern, not look-ahead. Status: information present (probe), causal role small and only consistent in
744706.

**05:15. Replication (replicate_744706_s10.json).** 22 of 22 LD2 definitions tested are matched by a variable of
744706's own dictionary above the line (0.329), with 744706's own line for the look-ahead primitives.

**05:21. The DISCOVERY battery on ten translated variables (battery_s10.json).** None passes all six pre-registered
checks. Closest: variable 502, "not empty and not open_file and la_m1_to" (the piece the chosen move takes): fidelity
0.459, not a restatement (R2 0.32), value gain +0.31% (CI +0.04% to +0.59%), replicates (0.592), a look-ahead
primitive; substitution restores 30% (bar 50%). Substitution restores 21% to 47% across the ten (in check: 47%, but a
restatement, R2 1.0). Forward selection (residual_curve_s10.json) reached +0.96% of value variance with six concepts.

**05:22. Next.** Compositions of up to three programs for the strongest candidates, with a composition line from the
untrained dictionary, to see whether substitution clears 50%; fresh-games knowledge test; live teachability.

**05:24. Fresh games (fresh_knowledge_s10.json).** On 16,000 positions from games never used anywhere, the 114
translated variables add +0.65% (CI +0.13% to +1.17%) to the textbook library's prediction of the result (R2 0.149);
LD2's own value adds +3.58% (CI +2.89% to +4.22%).

**05:25. Compositions (compose_sub_K3_s10.json; declared in PREREGISTRATION.md).** Three-program definitions, line
0.388 from the untrained dictionary. Variable 174 reaches fidelity 0.759 and substitution 0.501; 502: 0.353; 43:
0.328; 166: 0.347.

**05:26. Checks of the marginal pass.** Variable 174's composition on a second, disjoint set of 800 positions:
substitution 0.536 (CI 0.485 to 0.585); first set 0.501 (CI 0.433 to 0.568). Its composition is matched in 744706
(variable 455, 0.699). All six pre-registered checks pass, at the margin.

**05:26 to 05:28. Look-ahead in the transformer (lookahead_causal2_t1.json).** Same test on Leela's transformer t1:
+0.075 (CI +0.049 to +0.104) at block 4, +0.086 (CI +0.061 to +0.114) at block 6, +0.054 (CI +0.035 to +0.079) at
block 8, holding where the follow-up square is attacked by the moved piece and where it is not. Jenner et al.'s known
answer is recovered; the convolutional networks show 0.006 to 0.023. Strength is a confound: t1 is the stronger net.

**05:28. Teachability, live (teach_dict_s10.json).** Failed: a fresh Flash call without thinking could not apply the
sentences to FEN positions (hit rates 0.00 to 0.13, chance 0.05; 3 of 10 unparseable). A low-thinking run did not
return in 11 minutes and was stopped.

**05:29.** Docs written (RESULTS, FINDINGS, DISCOVERY, STATUS, PREREGISTRATION); ledger row R1 closed. Next: Rosetta
on the transformer's middle layer, to name its look-ahead variable directly.

**05:35. Rosetta inside the transformer (t1_rosetta_site6.json).** A sparse dictionary of t1's block 6 (512 variables)
scanned against t1's own line (line from an untrained t1: 0.041). The strongest look-ahead-bound variable is the same
concept as LD2's variable 174 and 744706's variable 455: variable 245, "not pawn_att_them and outnumbered_opp and
la_m1_to", 0.728 (static rival 0.686). Others: variable 354, "opp_K and la_m1_attacks" (the move gives check), 0.665
(static rival 0.594). No single variable is defined by the reply or the follow-up move: the transformer's look-ahead,
causal at the square level, is not one variable of this dictionary. (The causal check of variable 354 in that run had
a seeding bug in its bootstrap, so its interval is not reported; its effect was near zero: 0.001.)

**05:40. The capture concept inside the transformer, substituted (t1_sub_site6.json).** t1's variable 245 as three
programs ("not pawn_att_them and outnumbered_opp and la_m1_to" 2.90, "not opp_P and not pawn_att_us and
outnumbered_opp" 1.82, "hanging_opp and mob_us>=1 and not la_m1_to" 1.13): fidelity 0.800; put in place of the variable
inside t1 at block 6, it restores 73.1% of the variable's effect on the value (bar 50%). The same concept is a working
part of all three networks: LD2 (restores 50% to 54%), t1 (73%), matched in 744706 (0.70).

**05:44. Which variables carry the knowledge? (knowledge_select_s10.json).** Each of the 114 translated variables
scored alone on the held-out split of the original games; the five best then tested on the fresh games (untouched by
the choice): none adds significantly alone (best +0.18%, CI -0.06% to +0.42%), and the five together +0.22% (CI
-0.14% to +0.56%). The knowledge the translations carry on fresh games (+0.65% with all 114) is spread over many
variables, not concentrated in a few nameable concepts.

**05:49. Transformer probe profile (lookahead_profile_t1.json).** Follow-up square (t1's own line): AUC 0.851 /
0.894 / 0.911 at blocks 2 / 4 / 6, board-only 0.798, untrained t1 0.61 to 0.65. LD2: 0.804 / 0.824 / 0.833. The
transformer both carries its follow-up move more strongly and uses it (causal effect about four to five times larger).

**05:52 to 05:57. Look-ahead by the kind of follow-up move (exploratory split, after the main test;
lookahead_causal2_*_bytype.json, lookahead_causal2_t1_bytype.json).** Cost minus the matched control, blocks 4 and 6:
the transformer t1 looks ahead along checks (+0.14, +0.18), captures (+0.91, +0.82) and quiet moves (+0.025, +0.039),
all intervals above zero; 744706 along captures (+0.24, +0.26) and marginally checks (+0.05), not quiet moves (+0.01,
+0.01, intervals include zero); LD2 along checks (+0.05, +0.11) and captures at block 4 (+0.14), not quiet moves
(+0.006, -0.011). Convolutional Leela networks look ahead only along forcing lines; the transformer along every kind
of move. The low-thinking teachability run was stopped (no answer after about ten minutes).

**05:59. R2 registered (ledger row R2) before the run.** The by-type split was exploratory, so it is retested on 2,000
fresh positions (Kepler's fresh Lichess set, seeded sample), each network's own line, blocks 4 and 6, with three
predictions fixed in advance: t1 looks ahead along quiet follow-ups (CI above 0 at both blocks); the convolutional
nets' quiet effect stays below t1's (upper CI bound under t1's estimate); all three look ahead along forcing follow-ups
(CI above 0 at block 4 or 6). Running.

**06:06. R2 result: partly confirmed (lookahead_fresh_*.json).** On 2,000 fresh positions: forcing follow-ups, t1
+0.26 and +0.31, 744706 +0.10 and +0.08, LD2 +0.09 and +0.09 (blocks 4, 6; all intervals above zero); quiet
follow-ups, t1 +0.029 and +0.035 (above zero), 744706 +0.023 and +0.023 (above zero), LD2 +0.001 and +0.005 (not).
Predictions 1 and 3 held; prediction 2 failed for 744706 at block 4. The exploratory "convolutional nets look ahead
only along forcing lines" is withdrawn for 744706; what stands is that forcing lines carry most of the look-ahead in all
three networks and the transformer's is about three times as strong.

**06:07. The capture concept substituted in 744706 (sub_744706_capture.json).** 744706's variable 455, defined from its
own line as three programs (fidelity 0.732), restores 54.9% of its effect on 744706's value at block 10. The capture
concept is now a substitutable part of all three networks: LD2 50% to 54%, 744706 55%, t1 73%.

**06:10 to 06:17. Why the glass model misses the evaluation (glass2_s10.json, glass2_where_s10.json,
glass2_global_check_s10.json; rosetta/glass2.py).** The variables were ranked by attribution (activation times the
gradient of LD2's value along each decoder direction, 600 training positions). (1) Defining the 64 most important ones
as sums of three programs instead of one barely helps: value fit 0.062 to 0.074, top-move agreement 41.3% to 39.9%
(1,000 held-out positions). (2) Putting the K most important variables back at their true values, all others at their
single programs: value fit 0.06 (K=0), 0.48 (8), 0.70 (16), 0.85 (32), 0.92 (64), 0.97 (128), 0.99 (512). Move
agreement stays near 40% until K=128 (51%), 256 (61%), 512 (70%). The evaluation goes through a few variables; the move
choice is spread over many. (3) About half of the top 16 are variables the per-square vocabulary barely defines
(fidelity 0.07 to 0.26), and their board-wide totals track the evaluation: variable 434's total correlates -0.69 with
LD2's value and -0.46 with the part textbook concepts miss; 304 +0.72 and +0.39; 415 +0.51 and +0.43; 192 -0.61 and
-0.43. They carry an evaluation already computed upstream, so a rule about one square cannot define them.

**06:17. Registered before the run (exploratory, no pass line): where does LD2's extra knowledge go?** On the 16,000
fresh positions: textbook concepts plus the board-wide totals of the K most important variables (true codes, K = 8, 16,
64, 512), trained on the original games, scored on fresh games, against LD2's own value (+3.58% earlier) as the
ceiling. Reported as a share of the network's gain with bootstrap intervals (rosetta/fresh_where.py).

**06:19 to 06:21. Where LD2's extra knowledge goes (fresh_where_s10.json) and whether its carriers can be counted
(top_pos_s10.json).** On the 16,000 fresh game positions, textbook concepts plus the board-wide totals of LD2's 8 most
important variables (true codes) recover 71% of LD2's own gain over textbook knowledge (+2.54% against +3.58%); the
top 16 recover 91% (+3.25%, CI 2.59 to 3.94%); all 512, 102%. Controls added after the first result: 16 random
variables 62%, 17% and 13% (the 62% draw contains the variables ranked 5th and 6th); the 16 least important 4.5%. So
the network's knowledge beyond the textbook passes mostly through about 16 of its 512 variables at block 10. Defining
those 16 as counted programs at the position level (material removed) passes the untrained-network line for only 2 of
them (77 and 228, already well defined square by square); the evaluation carriers (434, 415, 303, 304, 192 and
others) stay at 0.11 to 0.36, below the line.

**06:22. Registered before the run: the carriers hold what LD2's own line wins (rosetta/line_swing.py).** Material swing
along LD2's own 3-ply line (after 1, 2 and 3 plies, side to move's balance minus now). (1) Held-out correlation of the
top 16 variables' totals with the 3-ply swing, raw and with textbook concepts removed. (2) Fresh games: textbook plus
the three swings, against LD2's value as the ceiling; control: the same swings along random legal moves. Yes if (2)'s
interval sits above the control's and recovers a clear share of the network's gain; no otherwise.

**06:22. Line-swing result (line_swing_s10.json): yes for (2), no for (1).** (2) On the 16,000 fresh positions,
textbook concepts plus the material LD2's own 3-ply line wins recover 33% of LD2's gain over textbook knowledge (+1.18%,
CI 0.78 to 1.60%); the same swing along random legal moves recovers 0.6% of it (+0.02%, CI -0.25 to 0.32%). Adding the swing to LD2's
value adds nothing (102% of the value's gain, intervals overlapping), so the value already contains it. (1) But the
16 evaluation variables do not carry it one by one: held-out correlations of their totals with the 3-ply swing are
between -0.12 and +0.13 (after removing textbook concepts, -0.09 to +0.10). A third of what LD2 knows beyond the
textbook is what its own forcing line wins; which variables hold it jointly is the next question.

**06:25. Pushed f5888a55 (R3 in the ledger).** Then, exploratory: do the 16 evaluation variables track winning, drawing
or losing (carriers_wdl_s10.json, 3,000 held-out positions)? They split into "we are winning" variables (304: +0.66
with LD2's win probability, -0.68 with loss; 493 +0.62 with win; 193, 415, 50, 0 and 184 +0.34 to +0.46) and "we are
losing" variables (434: +0.78 with loss; 192 and 249 +0.59 and +0.53; 303 +0.54). Some also track how drawish the
position is beyond who is ahead (draw probability with W - L removed): 434 -0.49, 493 -0.40, 508 +0.34. The network
keeps win, loss and draw as separate signals rather than one score.

**06:26. Error: the 06:26 update to Shreyas was headed "06:27", a time typed by hand before reading the clock (E8
again).** The clock read straight after it says 06:26:13. Corrected in the next message.

**06:27. Registered before the run: does the second network 744706 also pass its extra knowledge through a few
variables? (rosetta/fresh_where_744706.py)** Its own dictionary at block 10 (same recipe), variables ranked by
attribution to its own value, the same fresh-game comparison and controls as LD2's. Replicates if the top 16 recover
more than every control set (three random draws of 16 and the 16 least important) and more than half of 744706's own
gain; fails otherwise.

**06:30. 744706 replication: replicates (fresh_where_744706_s10.json).** Its own dictionary (fraction of variance
unexplained 0.16), its own attribution. On the 16,000 fresh positions, 744706's value adds +4.10% (CI 3.42 to 4.77%)
over textbook concepts; the board totals of its top 8 variables recover 75% of that, the top 16 78% (+3.19%, CI 2.41
to 3.95%), the top 64 92%, all 512 84%; controls: 16 random variables 35%, 37%, 39%; the 16 least important 1%. Both
registered conditions hold (top 16 above every control, above half). In both networks the extra knowledge passes
through a few variables.

**06:31. Registered before the run: is "what LD2's own line wins" only static tactics? (rosetta/line_swing_static.py)**
Control: value-weighted totals of outnumbered and attacked-by-lower pieces, both sides (the textbook concepts already
hold hanging pieces, hanging value, captures available and the best free capture). The look-ahead reading stands if
the swing still adds over textbook plus static tactics with an interval above 0; otherwise the one-third claim is
reworded as static tactics.

**06:32. Static-tactics control: the look-ahead reading stands (line_swing_static_s10.json).** On the fresh games,
static tactics add +0.23% over textbook concepts (CI -0.10 to +0.56%), the line swing +1.18%, both together +0.84%
(CI 0.40 to 1.27%); the swing's addition over static tactics is +0.61% (CI +0.22 to +1.01%), above 0 as registered.
Caveat: together they score below the swing alone, so the fitted gains carry some noise, and the swing's own share is
better quoted as between a sixth and a third of LD2's extra knowledge (+0.61% to +1.18% of +3.58%).
