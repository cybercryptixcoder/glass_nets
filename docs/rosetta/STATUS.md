# Rosetta: status

Branch `rosetta` (cut from `review-next-build`). Ledger row R1 in docs/glass/CHANGES.md. Intent: docs/rosetta/INTENT.md.

## Done (Nov 26, 04:09 to 07:09 ET)

- Engine: `rosetta/core.py` (program search, restricted expressions), `rosetta/engine.py` (definitions at the square and
  position level, compositions, control lines, substitution), `rosetta/sae.py` (variable finder), `rosetta/llm.py`
  (cheap-model proposer and teachability, live through Luminal).
- Chess library with look-ahead: `rosetta/chess_lib.py`, `rosetta/line.py`, `rosetta/build_prims.py`.
- Known answers: `rosetta/yardstick.py`, `rosetta/run_yardstick.py`, sealed rules in `rosetta/sealed/`.
- LD2, raw channels: `rosetta/run_ld2.py`; variables: `rosetta/run_feats.py`; whole layer: `rosetta/dictionary.py`;
  residual curve: `rosetta/residual_curve.py`; fresh games: `rosetta/fresh_knowledge.py`; battery: `rosetta/battery.py`,
  `rosetta/compose_sub.py`, `rosetta/sub_check.py`; replication: `rosetta/replicate.py`, `rosetta/rep_compose.py`;
  look-ahead: `rosetta/probe_lookahead.py`, `rosetta/lookahead_causal.py`, `rosetta/lookahead_causal2.py`,
  `rosetta/t1_lookahead.py`; teaching: `rosetta/teach_dict.py`.
- Results: `results/rosetta/*.json`; write-ups: RESULTS.md, FINDINGS.md, DISCOVERY.md (plain language), DIARY.md.

## Data outside the repo (~/kdata, gitignored)

`~/kdata/chess/` (Kepler's networks, positions and activations), `~/kdata/rosetta/` (primitives, lines, sparse codes,
yardstick networks, fresh-games cache). Rebuild: `python3 rosetta/line.py LD2`, `python3 rosetta/build_prims.py LD2`,
same for 744706, `python3 rosetta/yardstick.py`, `python3 rosetta/sae.py LD2 10`, then the drivers above.

## Results in one line each

- Known answers: 3 of 3 planted rules recovered blind.
- Look-ahead: all three Leela networks look ahead mostly along forcing lines; the transformer's is about three times as
  strong (confirmed on fresh positions, ledger R2, two of three predictions).
- A capture concept passes all six pre-registered checks in LD2 (at the margin) and can be substituted in all three
  networks (50% to 73%).
- Translated variables carry about a fifth of LD2's extra knowledge about game results, on fresh games, spread thinly.
- No new chess concept; the textbook gap is mostly unexplained (translations close about 7%).

## Next (in order of value)

1. Translate the transformer t1 layer by layer (it is the network that looks ahead) and show its look-ahead as an
   explicit step of a glass model.
2. A teachability test with a reader that sees the board (rendered positions), and then human players.
3. Look-ahead across more networks of both architectures and different strengths, to separate design from strength
   (R2 showed the clean split by architecture does not hold for 744706).
4. Compositions for every variable of the dictionary, to raise the glass model's evaluation (now R2 0.06).
