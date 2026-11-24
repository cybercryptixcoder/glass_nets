# Kepler: results in one page (Nov 24, 2026)

**The ask:** one interpretability engine that makes real discoveries inside trained networks. Two targets:
- how a video model does fluid dynamics;
- what a chess network knows that people don't, stated so a person can learn it.

**Branch:** `kepler`. Details: `FINDINGS.md`. Day log: `DIARY.md`. Pre-registrations and verdicts: K1–K11 in
`docs/glass/CHANGES.md`. Evidence files: `results/kepler/`.

## What Kepler is

One engine with four levels:
1. **Concept atlas:** which variables a network computes, where, with untrained-copy and shuffled controls.
2. **Units:** which internal units carry them, tested by intervention.
3. **Mechanisms:** what the network does with them.
4. **Laws and strategies:** stated as equations or executable rules, with the part they leave unexplained measured.

Every claim is judged by tests written before the result existed. Each test has known answers that must pass and
controls that must fail.

## Fluid: the governing equation and the algorithm, read out of a video model (done earlier today)

- **The law.** From a trained fluid video model alone, Kepler picked the **exact 5 terms of the hidden law out of 38**
  and recovered their coefficients within about 2%:
  - advection −1.0009 (truth −1);
  - viscosity 0.0102 (0.010);
  - drag 0.0511 (0.050);
  - forcing 0.511 (0.5);
  - beta 0.602 (0.6).
  - It also found where the model's law breaks down: small scales and transients.
- **The mechanism.** The model rebuilds the velocity field from vorticity with a **learned Biot–Savart law**, formed in
  its middle blocks:
  - the far-field response matches 0.97, against 0.70 for an untrained network;
  - swapping in another flow's internal velocity changes the output exactly as the law predicts (0.86, against 0.28 for
    an unrelated flow).
- **The smoke model.** A model shown only smoke rebuilds the invisible velocity field inside (probe 0 → 0.89 across
  layers), then applies the exact transport law: −1.010 and κ 0.00307, against truth −1 and 0.003.

## Chess: strategies from a strong network, priced against real human games (this session)

**Port.** The Leela transformer t1-256x10 is ported and verified against lc0's code. The one bug: lc0 silently uses a
squared-ReLU feed-forward layer for old-format files.

**The search.** About 6,000 rules of the form "in situation C, play move type M", mined from 16,000 human decisions.
Each candidate had to:
- clear a significance line taken from a shuffled-label control;
- appear in all three independently trained networks (t1, LD2, 744706);
- hold in every rating band;
- be confirmed on 16,000 fresh positions from other games;
- face board-edit cause tests;
- be priced by t1's own value head **and by the real results of the human games**.

**Results.**
- **The miner rediscovers the textbook.** It finds the opening and endgame principles first: develop knights, push
  centre pawns, push flank pawns in endgames, attack the uncastled king. These are known answers.
- **Five rules confirmed on fresh games.**

  | Rule | Network vs humans | Do humans who follow it win more? |
  |---|---|---|
  | **Quiet position: play the pawn lever** (a pawn push that attacks an enemy pawn) | 1.4× (d-pawn 1.9×, a-pawn 2.5×) | **Yes: +0.08 score** [0.01, 0.15] |
  | **Enemy king still in the centre: play the lever** | 1.6× | **Yes: +0.13** [0.01, 0.24] |
  | Quiet position: leave a pushed pawn en prise | 1.6× | +0.19 [−0.00, 0.38], only 21 games |
  | Behind in material: offer trades | 1.6× (all of it in lost positions) | No: −0.02 [−0.13, 0.10] |
  | **Lost: trade queens** (the textbook says the opposite) | 2.0×; t1's draw chance per trade +4.0 points | **No: −0.04** [−0.16, 0.09] |

- **Two different kinds of knowledge.**
  - **Pawn levers:** humans play fewer of the networks' levers at almost every rating level (fewest in the middle and
    top bands of the fresh games). Those who do play them score better. This is a lesson that transfers to people.
  - **Queen trades when lost:** this inverts the textbook rule ("when behind, avoid trades, especially of queens"). It is
    right by the network's own judgement, but in real games it buys humans nothing. This is behaviour that pays only
    against a superhuman opponent.
- **Cause tests failed for these rules.** Board edits that switch the situation off (material, lostness) do not switch
  the behaviour off. The rules describe where the networks' choices differ from humans'. They are not a switch inside
  the network that the situation flips.

## What is a first, and what is not

- **Plausibly a first** (prior work in `RESEARCH.md`):
  - reading a governing PDE and a causally verified internal algorithm (learned Biot–Savart) out of a video model, blind
    and pre-registered;
  - an automated pipeline that turns a superhuman network's play into conditional strategy rules confirmed on fresh games
    across three networks, and **prices each rule twice (the network's value head and real game outcomes)**. That double
    pricing is what separates transferable lessons from superhuman-only behaviour. Prior work looked inside AlphaZero
    (McGrath 2022), taught its concepts to grandmasters by hand (Schut 2025), or modelled human moves by rating (Maia).
- **Not new:**
  - the lever (Kmoch, 1959), now measured as under-played by humans and worth points;
  - the networks' extra pawn sacrifices and rook-pawn pushes (known AlphaZero style).
  - No chess rule passed the cause test, so no "new strategy the network uses because of X" is claimed.
- **Negative results kept:**
  - 59 LLM-written concept programs (three scientist rounds, including minimal pairs) explained none of LD2's value
    residual;
  - classical concepts explain 78% of LD2's evaluation, and the rest is mostly tactical foresight.

## Reproduce

Data lives in `$KDATA` (default `~/kdata`): Lichess January 2013 games and puzzles, and the three network files.
```
python3 scripts/kepler/chess_t1_check.py            # K10 known answers for the transformer port
python3 scripts/kepler/chess_rules.py eval          # t1 / 744706 on the 16,000 K8 positions
MOVES=c python3 scripts/kepler/chess_rules.py evalc
MODE=b python3 scripts/kepler/chess_rules.py mine   # K11b; add MOVES=c for K11c
python3 scripts/kepler/chess_rules.py fresh         # 16,000 fresh positions, three networks
python3 scripts/kepler/chess_rules.py confirm       # K11d
python3 scripts/kepler/chess_rules.py freshq        # K11e (queen trades by evaluation, draw mechanism)
python3 scripts/kepler/chess_rules.py trades        # b1 cause / worth / mechanism
python3 scripts/kepler/chess_rules.py lostcause     # K11f
```
Fluid: `scripts/kepler/fluid_build.py`, `fluid_known_test.py`, `fluid_law.py`, `fluid_mechanism.py`, `dye_law.py`.
