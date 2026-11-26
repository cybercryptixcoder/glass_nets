# Rosetta: findings (sprint of Nov 26, 2026, 04:09 to 07:09 ET)

## What was built

Rosetta is a translator from a trained network's own internal variables into short, executable, named programs, each
checked by intervening on the network. One loop, domain-agnostic in its core (`rosetta/core.py`, `rosetta/engine.py`),
with a chess library plugged in (`rosetta/chess_lib.py`):

1. **Map** what a known vocabulary explains of the network and what it leaves (the residual).
2. **Find the network's own variables**: raw units, or a sparse dictionary of the layer (`rosetta/sae.py`), kept only
   if removing them changes the network's output.
3. **Define each variable** with a program of up to three conditions over 61 per-square primitives (or a weighted sum
   of up to three such programs), by exhaustive search on training positions; scored on held-out positions. No network
   weights or fitted sub-models are allowed in a program, so a disguised copy of the network cannot pass. Pass lines
   come from control units (an untrained copy of the network, scrambled activity), computed before the real units.
4. **Substitute** the program for the variable inside the network and measure how much of its effect is restored.
5. **Re-measure the residual**, test against game results (including 16,000 fresh games), replicate in a second
   network, and test whether a fresh model can apply the plain-English definition.
6. **Recognise search**: the library includes the squares of the network's own best line (its move, the reply, its
   follow-up), so look-ahead inside a network can be detected, separated from static rivals, and tested causally.

A cheap model at volume (DeepSeek V4.1 Flash, live) proposes richer definitions; every proposal is scored by code.

## What it found

1. **It recovers planted knowledge blind.** On three small networks trained on sealed board rules, Rosetta's answers
   matched the sealed rules at 1.000, 0.988 (an equivalent formulation) and 1.000, without reading the sealed file. A
   network with a constant output got no definitions. One false positive on a second random network (1 of 7 units
   just over the line, which is the best of 32 control units, so chance passes are expected).

2. **All three Leela networks look ahead along forcing lines; the transformer does so about three times as strongly,
   and also along quiet moves.** Removing the network's state on the square where its own follow-up move will land
   (never a square of its first two moves) lowers the probability of its current move, against a control square matched
   on occupancy, on being attacked by the moved piece and on attack counts. The split by the follow-up move's type was
   found on the main sample and then retested, with predictions registered in advance (ledger R2), on 2,000 fresh
   positions from games never used in this work:

   | network, block | all follow-ups | forcing (check or capture) | quiet |
   |---|---|---|---|
   | t1 (transformer), 4 | +0.052 (+0.039 to +0.067) | +0.262 (+0.162 to +0.381), n 196 | +0.029 (+0.019 to +0.039), n 1801 |
   | t1 (transformer), 6 | +0.062 (+0.048 to +0.077) | +0.310 (+0.213 to +0.417), n 196 | +0.035 (+0.025 to +0.045), n 1801 |
   | 744706 (convolutional), 4 | +0.030 (+0.016 to +0.045) | +0.102 (+0.043 to +0.171), n 188 | +0.023 (+0.009 to +0.038), n 1812 |
   | 744706 (convolutional), 6 | +0.028 (+0.017 to +0.040) | +0.080 (+0.027 to +0.144), n 188 | +0.023 (+0.012 to +0.035), n 1812 |
   | LD2 (convolutional), 4 | +0.010 (-0.006 to +0.027) | +0.087 (+0.041 to +0.136), n 193 | +0.001 (-0.016 to +0.020), n 1732 |
   | LD2 (convolutional), 6 | +0.013 (-0.005 to +0.032) | +0.085 (+0.042 to +0.131), n 193 | +0.005 (-0.016 to +0.026), n 1732 |

   (Cost in log-probability of the network's current move, minus the matched control; 95% bootstrap intervals.)

   Of the three registered predictions, two held (the transformer looks ahead along quiet moves at both depths; all
   three networks look ahead along forcing moves) and one failed in part: 744706, a convolutional network, also looks
   ahead weakly along quiet moves (+0.023), about as much as the transformer at block 4. LD2 shows none along quiet
   moves. So the robust statement is about strength and kind, not a clean split by architecture: forcing lines carry
   most of the look-ahead in every network, and the transformer's is the largest.

   Jenner et al. (2024) found look-ahead in Leela's transformer and built their method on its squares-as-tokens
   structure; Rosetta reproduces that known answer with its own recogniser and adds the comparison with convolutional
   Leela networks. The transformer also carries the follow-up square more clearly: a probe reads it at AUC 0.91 at
   block 6 (board alone 0.80), against 0.83 for LD2 (board alone 0.79, untrained copy 0.64).

3. **One internal concept passes every pre-registered check.** LD2 variable 174, defined as a weighted sum of three
   programs: "an enemy piece other than a pawn, with more attackers than defenders, on the square the network's chosen
   move goes to" (weight 1.78) + "an undefended enemy piece attacked by one of our pawns or minor pieces, on a square
   none of our moves gives check from" (0.63) + "the chosen move captures something other than a knight" (0.51):
   - fidelity 0.759 on held-out positions (line 0.388);
   - not a restatement of the 61 classical concepts (R2 0.72, bar 0.9);
   - substituted for the variable inside LD2 it restores 50.1% of the variable's effect (CI 43% to 57%), and 53.6%
     (CI 48% to 58%) on a second, disjoint set of positions (bar 50%);
   - adds +0.57% of LD2's value variance to the classical surrogate (CI +0.21% to +0.93%);
   - matched by a variable of the independently trained 744706 at 0.699;
   - uses the network's own move (a look-ahead primitive); it does not add significantly to predicting game results.

   The same concept is a working part of the other two networks as well, each defined from its own line and swapped
   in at the block where it lives: t1's variable 245 (fidelity 0.800) restores 73% of its variable's effect on t1's
   value at block 6; 744706's variable 455 (fidelity 0.732) restores 55% at block 10 (its effect on 744706's value is
   small, 0.002 on average).

   In chess terms it is a winning-capture detector tied to the move being chosen: all three Leela networks, convolutional
   and transformer, dedicate a variable to "material I can win, especially with the move I am about to play". That is
   known chess (capturing an insufficiently defended piece), now as a verified, executable, replicated internal variable
   of Leela, not new chess knowledge.

4. **The translated vocabulary carries real knowledge about who wins.** The 114 variables Rosetta translated at
   fidelity 0.3 or more, as counts per position, add to the classical library's prediction of the game result:
   - on held-out positions: +1.31% of variance (CI +0.16% to +2.45%), against +3.64% for LD2's own value;
   - on 16,000 positions from games never used anywhere: +0.65% (CI +0.13% to +1.17%), against +3.58% for LD2's value.

   About a fifth of what the network knows about game results beyond the textbook is carried by variables Rosetta can
   name, but spread over many of them: the five that looked best on the original games add nothing significant on the
   fresh ones, alone or together.

5. **A full dictionary of a chess network's last layer.** 485 of LD2's 512 last-block variables translated into
   programs: 114 at fidelity 0.3 or more, 49 at 0.5, 14 at 0.7. Rebuilt from those programs alone and run through
   LD2's own heads, the layer gives the network's top move 41% of the time (5% for a layer that knows nothing, 70% for
   the dictionary itself), but almost none of its evaluation (R2 0.06 against 0.99). The translations keep move logic,
   not judgement. All 22 definitions tested in the second network replicate.

6. **Where the network's judgement lives (exploratory, after the main tests; RESULTS.md section 9).** The glass model
   loses the evaluation because of a handful of variables: putting back at their true values the 8 that LD2's value
   depends on most lifts the evaluation fit from R2 0.06 to 0.48, and 16 lift it to 0.70; the move choice needs more
   than 128. On 16,000 fresh games, the board-wide totals of those 16 variables carry 91% of LD2's gain over textbook
   knowledge (16 random variables: 13% to 62%; the 16 least important: 4.5%); in the second network 744706, with its
   own dictionary, its top 16 carry 78% (random 16: 35% to 39%; registered before the run). The 16 split into "we are
   winning", "we are losing" and "drawish" signals, and the concentration builds up toward the output (the top 16
   carry 56%, 66% and 91% at blocks 6, 8 and 10). Rosetta's programs define only 2 of the 16, square by square or
   counted over the board. A sixth to a third of LD2's knowledge beyond the textbook is the material its own 3-ply
   line wins (33% alone, 17% over static tactics; the same count along random legal moves: 0.6%), and its value
   already contains it; in 744706, along its own line, 12% to 21% (registered before the run). So the translated rules
   spread their small share of the knowledge thinly (item 4), while the network concentrates its judgement in about 16
   variables that need a vocabulary of lines and exchanges, not squares.

## What it did not find

- **No new chess concept.** The unexplained 22% of LD2's evaluation is still mostly unexplained: all translated
  variables together close 6.6% of it (+1.46% of variance), and the best six, chosen on training positions, +0.96%.
  We now know where it passes (about 16 variables, item 6) but not what those variables compute.
- **Raw channels cannot be defined well** (best 0.46 with four programs; substitution restores at most 4%); the sparse
  dictionary was needed.
- **The cheap model at volume added nothing here**: 45 usable proposals from 18 live calls, none better than the
  exhaustive search.
- **No clean architecture split in look-ahead**: 744706 looks ahead weakly along quiet moves too; LD2 does not.

## How sure we are, and where to doubt

- Pass lines came from controls. Three measurement changes were made after the known-answer cases or controls showed
  a flaw, before the real network was scored (DIARY 04:32, 04:37, 04:42), and two translation errors were fixed (04:51).
- Finding 3 is at the margin, and came after a declared second step: single programs failed substitution (best 30%),
  so compositions of three programs were tried on four candidates and one passed. The line for compositions came from
  the untrained dictionary. Its knowledge criterion is met only through the look-ahead clause.
- Finding 2 compares networks of different strength and age as well as architecture; the transformer t1 is a newer,
  stronger network, so architecture and strength cannot be separated with three networks. The split by move type was
  confirmed on fresh positions for two of three registered predictions; the third (no quiet look-ahead in either
  convolutional network) failed for 744706.
- Finding 4's effect is small, and the fresh-games test is the strictly untouched one.
- Finding 6 is exploratory work done in the last hour (ledger R3). Its knowledge numbers are all on the untouched fresh
  games. The 744706 replication, the line-swing test, its static-tactics control and its 744706 replication were each
  registered in the diary before they ran; the 16-variable count and the random-variable controls for LD2 were not.
  The variables were ranked by attribution to the network's own value, so they are expected to predict what the value
  predicts. The finding is how few variables suffice (16 of 512, while the move choice needs more than 128), not that
  they predict at all. With static tactics added, the line swing's gain fell below its gain alone (+0.84% against
  +1.18%), so these fitted gains carry noise. That is why its share is quoted as a range.

## Compared with prior work

- Jenner et al. 2024 (learned look-ahead in Leela's transformer): reproduced as a known answer; the comparison with
  convolutional Leela networks is not in their paper, which relies on the transformer's per-square tokens.
- McGrath et al. 2022 (AlphaZero concepts) probed for supplied human concepts; Rosetta defines the network's own
  variables with programs it searches for, tests them by substitution, and grows the vocabulary.
- The project's own record (Kepler, docs/kepler/FINDINGS.md section 4): 59 one-shot concept programs and 8 channel rules
  were all rejected on the same residual; Rosetta's variable-first route is what produced definitions that pass.
