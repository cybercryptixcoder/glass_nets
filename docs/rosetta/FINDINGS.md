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

2. **Leela's transformer looks ahead; its convolutional siblings barely do.** Removing the network's state on the
   square where its own follow-up move will land (never a square of its first two moves) lowers the probability of its
   current move. Against a control square matched on occupancy, on being attacked by the moved piece and on attack
   counts:

   | network | block | cost of removing the follow-up square, minus the matched control (log-probability) | 95% CI |
   |---|---|---|---|
   | t1 (transformer, 256 x 10) | 4 | +0.075 | +0.049 to +0.104 |
   | t1 (transformer, 256 x 10) | 6 | +0.086 | +0.061 to +0.114 |
   | 744706 (convolutional) | 4 | +0.023 | +0.004 to +0.041 |
   | LD2 (convolutional) | 4 | +0.016 | -0.002 to +0.036 |
   | LD2 (convolutional) | 6 | +0.006 | -0.015 to +0.027 |

   Jenner et al. (2024) found look-ahead in Leela's transformer and built their method on its squares-as-tokens
   structure; Rosetta reproduces that known answer with its own recogniser. The convolutional networks carry the
   information (a probe reads the follow-up square at AUC 0.83 at block 6 of LD2, against 0.79 from the board alone and
   0.64 for an untrained copy) but use it little. A looser first control had suggested a large effect in LD2 (+0.089);
   the matched control removed it, which is why the matched control is the one reported.

3. **One internal concept passes every pre-registered check.** LD2 variable 174, defined as a weighted sum of three
   programs: "an enemy piece other than a pawn, outnumbered by attackers, on the square the network's chosen move goes
   to" + "a hanging enemy piece we attack (no check involved)" + "any non-knight piece on the chosen move's target":
   - fidelity 0.759 on held-out positions (line 0.388);
   - not a restatement of the 61 classical concepts (R2 0.72, bar 0.9);
   - substituted for the variable inside LD2 it restores 50.1% of the variable's effect (CI 43% to 57%), and 53.6%
     (CI 48% to 58%) on a second, disjoint set of positions (bar 50%);
   - adds +0.57% of LD2's value variance to the classical surrogate (CI +0.21% to +0.93%);
   - matched by a variable of the independently trained 744706 at 0.699;
   - uses the network's own move (a look-ahead primitive); it does not add significantly to predicting game results.

   In chess terms it is a winning-capture detector tied to the move being chosen. That is known chess (capturing an
   insufficiently defended piece), now as a verified, executable, replicated internal variable of Leela, not new
   chess knowledge.

4. **The translated vocabulary carries real knowledge about who wins.** The 114 variables Rosetta translated at
   fidelity 0.3 or more, as counts per position, add to the classical library's prediction of the game result:
   - on held-out positions: +1.31% of variance (CI +0.16% to +2.45%), against +3.64% for LD2's own value;
   - on 16,000 positions from games never used anywhere: +0.65% (CI +0.13% to +1.17%), against +3.58% for LD2's value.

   About a fifth of what the network knows about game results beyond the textbook is carried by variables Rosetta can
   name.

5. **A full dictionary of a chess network's last layer.** 485 of LD2's 512 last-block variables translated into
   programs: 114 at fidelity 0.3 or more, 49 at 0.5, 14 at 0.7. Rebuilt from those programs alone and run through
   LD2's own heads, the layer gives the network's top move 41% of the time (5% for a layer that knows nothing, 70% for
   the dictionary itself), but almost none of its evaluation (R2 0.06 against 0.99). The translations keep move logic,
   not judgement. All 22 definitions tested in the second network replicate.

## What it did not find

- **No new chess concept.** The unexplained 22% of LD2's evaluation is still mostly unexplained: all translated
  variables together close 6.6% of it (+1.46% of variance), and the best five, chosen on training positions, +0.96%.
- **Raw channels cannot be defined well** (best 0.46 with four programs; substitution restores at most 4%); the sparse
  dictionary was needed.
- **The cheap model at volume added nothing here**: 45 usable proposals from 18 live calls, none better than the
  exhaustive search.
- **Look-ahead in the convolutional networks is weak**, and in LD2 not significant under the matched control.

## How sure we are, and where to doubt

- Pass lines came from controls. Three measurement changes were made after the known-answer cases or controls showed
  a flaw, before the real network was scored (DIARY 04:32, 04:37, 04:42), and two translation errors were fixed (04:51).
- Finding 3 is at the margin, and came after a declared second step: single programs failed substitution (best 30%),
  so compositions of three programs were tried on four candidates and one passed. The line for compositions came from
  the untrained dictionary. Its knowledge criterion is met only through the look-ahead clause.
- Finding 2 compares networks of different strength and age as well as architecture; the transformer t1 is a newer
  network. Architecture is the most likely explanation, not the only one.
- Finding 4's effect is small, and the fresh-games test is the strictly untouched one.

## Compared with prior work

- Jenner et al. 2024 (learned look-ahead in Leela's transformer): reproduced as a known answer; the comparison with
  convolutional Leela networks is not in their paper, which relies on the transformer's per-square tokens.
- McGrath et al. 2022 (AlphaZero concepts) probed for supplied human concepts; Rosetta defines the network's own
  variables with programs it searches for, tests them by substitution, and grows the vocabulary.
- The project's own record (Kepler, docs/kepler/FINDINGS.md section 4): 59 one-shot concept programs and 8 channel rules
  were all rejected on the same residual; Rosetta's variable-first route is what produced definitions that pass.
