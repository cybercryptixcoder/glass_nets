# Kepler: status and next steps

*Branch `kepler` (from `decompiler`). Intent: `docs/kepler/INTENT.md`. Diary: `docs/kepler/DIARY.md`. Research:
`docs/kepler/RESEARCH.md`. Ledger entries: K-series in `docs/glass/CHANGES.md`.*

**If you are a session picking this up:** read this file, then the diary's last entry, then continue from the first
unchecked item. Push after every step.

## Plan

1. [x] Branch, intent, ledger entry K1 (09:27).
2. [x] Fluid target: exact spectral simulator with hidden parameters (`kepler/targets/fluid/sim.py`).
3. [ ] Fluid video model trained on its frames (`scripts/kepler/fluid_build.py`, running since 09:30).
4. [ ] Research: prior work on concept discovery, symbolic distillation, world models, chess interpretability.
5. [ ] Engine core: concept atlas (probes with controls), law extraction (sparse regression on the model's implied
   dynamics), causal tests (concept-swap interventions), findings ledger.
6. [ ] Known-answer test for the law extractor (K1 (a)–(c)), then the held-out verdict on the trained model.
7. [ ] Fluid mechanism: where velocity lives inside the model, its learned Biot-Savart kernel, law vs. lookup out of
   distribution.
8. [ ] Chess target: a strong Leela network loaded in PyTorch, positions from Lichess, chess concept library.
9. [ ] Chess discovery: concept atlas, value decomposition into known concepts, the unexplained part, candidate new
   concepts with causal tests and example positions.
10. [ ] Lab page for findings (cards), final report.
