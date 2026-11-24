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
9. [~] Chess discovery: round 1 0/24, calculation feature accepted, units causal (channel 22: 32%), behaviour (rook pawns), round 2 running: concept atlas, value decomposition into known concepts, the unexplained part, candidate new
   concepts with causal tests and example positions.
10. [x] Lab page `/kepler` and `docs/kepler/FINDINGS.md`.

## Also running
- K2 smoke video model (dye only; velocity hidden) on the Colab T4: error on the one-step change 2.0% at step 6,750
  of 10,000. Analysis script ready: `scripts/kepler/dye_law.py` (known, then verdict).

## Next (if the session continues)
1. Finish scientist round 2; then a round 3 aimed at channel 22 with its causal and counterfactual evidence.
2. A cleaner control for the fluid velocity swap (swap at a site before velocity forms; wrong-velocity predictions).
3. Bigger Leela nets on Colab (transformer T-series) to see if the knowledge gap shrinks or moves.
