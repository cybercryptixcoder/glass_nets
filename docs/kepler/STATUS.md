# Kepler: status and next steps

*Branch `kepler` (from `decompiler`). Intent: `docs/kepler/INTENT.md`. Diary: `docs/kepler/DIARY.md`. Research:
`docs/kepler/RESEARCH.md`. Ledger entries: K-series in `docs/glass/CHANGES.md`.*

**If you are a session picking this up:** read this file, then the diary's last entry, then continue from the first
unchecked item. Push after every step.

## Plan

1. [x] Branch, intent, ledger entry K1 (09:27).
2. [x] Fluid target: exact spectral simulator with hidden parameters (`kepler/targets/fluid/sim.py`).
3. [x] Fluid video model trained (8,000 steps; 10.5% error on the one-step change).
4. [x] Research (`docs/kepler/RESEARCH.md`): prior work on concept discovery, symbolic distillation, world models, chess interpretability.
5. [x] Engine core (`kepler/core/`: law.py, probe.py, residual.py, scientist.py): concept atlas (probes with controls), law extraction (sparse regression on the model's implied
   dynamics), causal tests (concept-swap interventions), findings ledger.
6. [x] K1 known-answer test ALL PASS (10:09); blind verdict PASS (10:26); law by regime, amplitude and scale.
7. [x] Fluid mechanism (learned Biot-Savart, advection built in blocks 3-6; velocity swap partly confounded): where velocity lives inside the model, its learned Biot-Savart kernel, law vs. lookup out of
   distribution.
8. [x] Chess target (LD2 + 744706 ported; 20,000 positions; 61 concepts; K3 known tests: a FAIL as registered, b and c PASS): a strong Leela network loaded in PyTorch, positions from Lichess, chess concept library.
9. [x] Chess discovery on LD2: round 1 0/24, round 2 0/15, minimal pairs (K9) 0/20; calculation feature accepted (classical
   search, not new); unit channel 22 causal (32% of the quiet residual); behaviour vs humans (rook pawns: the known
   AlphaZero idea, recovered).
10. [x] Lab page `/kepler` and `docs/kepler/FINDINGS.md`.
11. [x] K10: Leela transformer t1-256x10 ported and verified (all four registered checks pass; the bug was lc0's hidden
    FFN activation rule).
12. [x] K11 (14:14-15:02): strategy rules from t1 against human decisions.
    - Textbook principles recovered.
    - 5 rules confirmed on fresh games across three networks.
    - Cause tests: one partial (enemy king's place and levers).
    - Real games (40,000 new positions): no rule adds to human play beyond the network's own moves.
    - **"When lost, trade queens" (counter-textbook) works in play against a strong opponent: +0.066 [0.017, 0.119].**
    - `docs/kepler/RESULTS.md` written.
13. [x] Wrap-up (15:02): FINDINGS, DIARY, RESULTS.md, ledger K10-K11j, lab page card and sheets, summary JSON, all pushed.

## Next (if a later session continues)
1. The queen-trade strategy:
   - play-outs with search (lc0 with nodes > 1) and more positions;
   - an opponent-strength sweep (sampling temperature 0, 0.3, 0.6, 1) to test the "only against strong opposition"
     scope properly.
2. Replicate the unregistered positive (trade offers when behind, +0.12 in real games) on new games before believing it.
3. Run the same pipeline on a bigger transformer (BT4) if one can be loaded; a teaching test with people.
