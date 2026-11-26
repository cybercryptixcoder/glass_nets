# Rosetta: intent

*Set by Shreyas on Nov 26, 2026 (message of 04:08 ET): build Rosetta, the most advanced system of the project, then
point it at targets and make discoveries. Three hours. This file states what Rosetta is for; the code serves it. It
does not change `docs/glass/INTENT.md` (that needs Shreyas's approval); it is the sprint's own intent.*

## The question

Can one engine translate a trained network's own internal variables into short, executable, named concepts, verified
by intervention, and in doing so explain knowledge the network has that our textbook vocabulary does not?

- **Yes** means: on networks with sealed answers it recovers the planted concept blind and rejects the must-fail
  networks; on a real network (Leela LD2) it defines at least one internal variable that is not a restatement of the
  classical library, that the network causally uses, that closes part of the unexplained evaluation on fresh
  positions, that replicates in a second network, and that carries knowledge (predicts game results beyond the
  classical library) or is a recognisable piece of the network's own look-ahead.
- **No** means: definitions do not beat the control line, or only restate the classical library.

## What Rosetta is (the build, from docs/review/HANDOFF.md, BRIEF.md and NEXT_BUILD.md section 3.3)

A residual-driven loop, domain-agnostic in its core, with a domain library plugged in:

1. **Map.** What the known vocabulary explains of the network's output, and the residual it leaves.
2. **Variables.** The network's own units that carry the residual: read inside the network, kept only if the network
   uses them (ablation changes the output).
3. **Definitions.** Short programs in a restricted language over the domain's primitives (no weights, a budget of
   terms), searched exhaustively over small compositions and proposed at volume by a cheap model, scored on held-out
   fidelity to the unit's own activity. The pass line comes from controls (the same search on an untrained network and
   on shuffled activity), never from the results it judges.
4. **Substitution.** The unit's activity is replaced by its program inside the network; the network's behaviour must
   be restored. This is the test a disguised copy cannot game: the program is in the restricted language.
5. **Composition and the residual curve.** Accepted definitions join the vocabulary; the residual is re-measured.
6. **Vocabulary growth.** Accepted definitions become named primitives available to the next round.
7. **Confirmation.** Fresh positions, a second network (replication), outside truth (game results), and a
   teachability test (a fresh model predicts the unit from the plain-English definition alone).

The chess library includes look-ahead primitives built from the network's own best line (its policy's top move and
the replies), so the box can recognise search inside the network, not only static features.
