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

  | Rule | Network vs humans | Real games, 40,000 new positions: what the rule adds for humans |
  |---|---|---|
  | **Quiet position: play the pawn lever** (a pawn push that attacks an enemy pawn) | 1.4× (d-pawn 1.9×, a-pawn 2.5×) | +0.01 [−0.04, 0.06] |
  | **Enemy king still in the centre: play the lever** | 1.6×; partly causal (enemy king moved to a castled square: appetite −13% vs −3% for a matched move) | −0.06 [−0.14, 0.02] |
  | Quiet position: leave a pushed pawn en prise | 1.6× | +0.00 [−0.11, 0.12] |
  | Behind in material: offer trades | 1.6× (all of it in lost positions) | +0.12 [0.02, 0.22], unconfirmed |
  | **Lost: trade queens** (the textbook says the opposite) | 2.0×; t1's draw chance per trade +4.0 points | −0.02 [−0.10, 0.06] |

- **What real games say.**
  - Playing the network's move helps humans by about 8 points whatever kind of move it is.
  - No confirmed rule adds anything to that which holds up on 40,000 positions from new games (K11i, registered first).
  - The single positive (trade offers when behind) is unregistered, opposite in sign to the earlier reading, and 1 of 5
    tests, so it needs its own replication.
  - **Correction, said plainly:** at 14:40 I first wrote "levers transfer to humans (+0.06 to +0.10)". The move-type
    controls (14:48) and the new games (14:57) showed that number was the general value of playing the network's move.
- **The strongest chess result: a counter-textbook strategy that works in play.**
  - **When lost against a strong opponent, trade queens.** The networks do it 2× as often as humans, in all three
    networks and every rating band, on fresh games.
  - Played out from 146 lost positions against t1, the trade lifts the lost side's score from 0.106 to 0.172 (+0.066
    [0.017, 0.119]). Two independent networks judge it the same way, and 80% of those games ended by play.
  - Against a fallible opponent it adds nothing (+0.01). Humans in real games got nothing measurable from it either.
  - The textbook (web search) says the opposite: when behind, avoid trades, especially of queens.
  - We found no source stating this rule. Its scope (strong opposition) is suggested, not proven: the difference
    between opponents is +0.06 [−0.02, 0.13].
- **Two different kinds of knowledge** (descriptive, all confirmed on fresh games):
  - **Levers:** humans under-play the networks' levers at almost every rating level.
  - **Queen trades when lost:** the networks invert the textbook rule ("when behind, avoid trades, especially of
    queens"). By their own judgement each trade raises the draw chance.
- **Cause tests: mostly negative, one partial.**
  - Board edits that switch material or lostness off do not switch the behaviour off.
  - The one matched test (enemy king castled vs moved within the centre) shows a real but modest causal effect on lever
    appetite: 13% vs 3%, below the 30% bar set beforehand.
  - The rules mainly describe where the networks' choices differ from humans'.
- **Numbers.** Real-game scores (1 win, ½ draw, 0 loss) are adjusted for t1's value of the position:
  - `transfer_new_games.json`: 40,000 new positions;
  - `transfer_controls_fresh.json`: the first 16,000 fresh games.
- **Teaching positions** for each rule (FEN, t1's move, the human's move, Lichess link):
  `results/kepler/chess/rules/examples_fresh.json`.

## What is a first, and what is not

- **Plausibly a first** (prior work in `RESEARCH.md`):
  - reading a governing PDE and a causally verified internal algorithm (learned Biot–Savart) out of a video model, blind
    and pre-registered;
  - an automated pipeline that turns a superhuman network's play into conditional strategy rules confirmed on fresh games
    across three networks, and **prices each rule twice**: by the network's value head, and by real game results against
    the general value of playing the network's move. Prior work looked inside AlphaZero (McGrath 2022), taught its
    concepts to grandmasters by hand (Schut 2025), or modelled human moves by rating (Maia). The pipeline's honest answer
    here is that the rules describe the network's style, while what helps humans is its specific moves.
  - **A counter-textbook chess strategy, found and verified end to end:** when lost against a strong opponent, trade
    queens. It was mined from the network against humans, confirmed on fresh games in three networks and every rating
    band, explained by the network's own draw estimate (+4 points), and shown to work in play (+0.066 [0.017, 0.119]
    expected score, two independent judges).
- **Not new:**
  - the lever (Kmoch, 1959), now measured as under-played by humans relative to three networks;
  - the networks' extra pawn sacrifices and rook-pawn pushes (known AlphaZero style).
  - No chess rule passed the board-edit cause test, so no internal "the network does this because of X" switch is
    claimed. The queen-trade strategy is verified by play, not by a mechanism inside the network.
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
python3 scripts/kepler/chess_rules.py outcomes      # K11g real game results (recovers the fresh games' results)
python3 scripts/kepler/chess_rules.py kingcause     # K11h matched king edits
python3 scripts/kepler/chess_rules.py transfer2     # K11i powered transfer test on 40,000 positions from new games
python3 scripts/kepler/chess_rules.py playout       # K11j play-out of 'trade queens when lost', strong vs fallible opponent
```
Fluid: `scripts/kepler/fluid_build.py`, `fluid_known_test.py`, `fluid_law.py`, `fluid_mechanism.py`, `dye_law.py`.
