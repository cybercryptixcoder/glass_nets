# Mechanism atlas: zoo_2nd_argmax_8_5 (original)

The network's weights are split EXACTLY into rank-1 mechanisms (SPD). Each mechanism has a readable trigger (when it is needed), scored on held-out inputs (its confidence), and an effect (what removing it changes).

## Whole-network certificate

- Rebuilt from its mechanisms, each switched on by its trigger where the trigger's confidence is >= 0.9 (otherwise always on): keeps **92.3%** of the network's decisions; the task rule alone keeps 87.9%.
- Every trigger used (no confidence bar): 73.7%. Mechanisms switched by their own learned importances: 87.5%. All alive mechanisms on: 100.0%.

## Findings

1. The weights split exactly into 22 mechanisms; 11 have a trusted trigger (held-out confidence >= 0.9), 3 are always needed.
2. 11 of the 11 trusted triggers use a task variable (not only neuron states). Most used: x_t (6), x_t minus the previous 2nd (6), step t (5), x_t minus the previous max (5), steps since the max (4).
3. 10 mechanism(s) (W_x.2, W_x.12, W_h.8, W_h.9, W_h.15, W_h.16, W_o.0, W_o.2, W_o.14, W_o.16) keep the network from answering rank 3 instead of rank 2: removing them where needed mostly flips the answer from rank 2 to rank 3.
3. 3 mechanism(s) (W_h.0, W_h.4, W_h.18) keep the network from answering rank 1 instead of rank 2: removing them where needed mostly flips the answer from rank 2 to rank 1.
4. Rebuilt from mechanisms switched by trusted triggers: 92.3% of decisions (task rule 87.9%).
5. Shuffled-weights control: 20 mechanisms, 11 trusted triggers, 9 using a task variable (trained: 22, 11, 11); functional groups: 0 mechanisms (trained: 13).

## In plain English

The network solves its task through 22 mechanisms, 11 of which have trusted triggers (confidence ≥ 0.9) and 3 of which are always needed. Those trusted triggers mostly depend on task variables—most often x_t, x_t minus the previous 2nd, step t, x_t minus the previous max, and steps since the max. Some mechanisms keep the network from answering rank 3 instead of rank 2 (W_x.2, W_x.12, W_h.8, W_h.9, W_h.15, W_h.16, W_o.0, W_o.2, W_o.14, W_o.16), while others keep it from answering rank 1 instead of rank 2 (W_h.0, W_h.4, W_h.18). Rebuilding from mechanisms switched by trusted triggers reproduces 92.3% of decisions, close to the 87.9% task rule. The facts do not explain how the remaining mechanisms, or the specific trigger conditions of several listed mechanisms, combine to produce the final answer.

*(written by the reader from the facts above; checked only for numbers and names)*

## W_x: 4 mechanisms (4 with a trusted trigger)

Alive mechanisms reproduce W_x to a relative error of 0.005.

| mechanism | in use | trigger (when it is needed) | confidence | removing it where needed | specificity |
|---|---|---|---|---|---|
| W_x.4 | 100% | always | 1.00 | changes the output by 0.71 and flips 28% of decisions; lowers class 0, raises class 3; flips go rank 2 → rank 3 (39%), rank 2 → rank 1 (30%) | only where needed |
| W_x.12 | 100% | always | 1.00 | changes the output by 1.14 and flips 38% of decisions; lowers class 0, raises class 1; flips go rank 2 → rank 3 (41%), rank 2 → rank 1 (28%) | only where needed |
| W_x.17 | 100% | always | 1.00 | changes the output by 0.75 and flips 27% of decisions; lowers class 1, raises class 2; flips go rank 2 → rank 3 (37%), rank 2 → rank 1 (27%) | only where needed |
| W_x.2 | 84% | ON when x_t > 0.20 (50% of its ON cases)<br>ON when x_t <= -0.22 (50% of its ON cases) | 1.00 | changes the output by 0.54 and flips 20% of decisions; lowers class 3, raises class 1; flips go rank 2 → rank 3 (41%), rank 2 → rank 1 (23%) | 20x |

## W_h: 11 mechanisms (8 with a trusted trigger)

Alive mechanisms reproduce W_h to a relative error of 0.017.

| mechanism | in use | trigger (when it is needed) | confidence | removing it where needed | specificity |
|---|---|---|---|---|---|
| W_h.5 | 70% | ON when step t > 1.50 AND x_t minus the previous max > -1.48 (75% of its ON cases)<br>ON when step t > 1.50 AND x_t minus the previous max <= -1.48 AND x_t > -1.16 (17% of its ON cases)<br>+ 1 rarer case(s), 8% of its ON cases | 0.83 | changes the output by 0.71 and flips 26% of decisions; lowers class 3, raises class 2; flips go rank 2 → rank 1 (38%), rank 2 → rank 3 (32%) | 148x |
| W_h.0 | 61% | ON when neuron 2 active in the state before this step AND x_t minus the previous max <= 0.28 (66% of its ON cases)<br>ON when not (neuron 2 active in the state before this step) AND step t > 3.50 AND neuron 6 active in the state before this step (21% of its ON cases)<br>+ 4 rarer case(s), 13% of its ON cases | 0.93 | changes the output by 0.98 and flips 35% of decisions; lowers class 2, raises class 1; flips go rank 2 → rank 1 (41%), rank 2 → rank 3 (36%) | 97x |
| W_h.11 | 55% | ON when neuron 0 active in the state before this step AND steps since the min > 1.50 (33% of its ON cases)<br>ON when neuron 0 active in the state before this step AND steps since the min <= 1.50 AND x_t minus the previous max <= -0.54 (25% of its ON cases)<br>ON when not (neuron 0 active in the state before this step) AND neuron 2 active in the state before this step AND not (neuron 1 active in the state before this step) (16% of its ON cases)<br>ON when not (neuron 0 active in the state before this step) AND neuron 2 active in the state before this step AND neuron 1 active in the state before this step AND step t > 3.50 (13% of its ON cases)<br>+ 2 rarer case(s), 12% of its ON cases | 0.89 | changes the output by 0.71 and flips 25% of decisions; lowers class 3, raises class 2; flips go rank 2 → rank 3 (34%), rank 2 → rank 4 (17%) | 75x |
| W_h.17 | 54% | ON when neuron 5 active in the state before this step (88% of its ON cases)<br>+ 3 rarer case(s), 12% of its ON cases | 0.93 | changes the output by 0.98 and flips 40% of decisions; lowers class 4, raises class 1; flips go rank 2 → rank 3 (39%), rank 2 → rank 1 (21%) | 85x |
| W_h.8 | 49% | ON when neuron 2 active in the state before this step AND distance of x_t's rank from the middle <= 1.25 (78% of its ON cases)<br>ON when neuron 2 active in the state before this step AND distance of x_t's rank from the middle > 1.25 AND not (neuron 1 active in the state before this step) (9% of its ON cases)<br>+ 2 rarer case(s), 12% of its ON cases | 0.90 | changes the output by 0.67 and flips 21% of decisions; lowers class 4, raises class 1; flips go rank 2 → rank 3 (40%), rank 2 → rank 1 (34%) | 62x |
| W_h.18 | 47% | ON when step t > 3.50 (74% of its ON cases)<br>ON when 2.50 < step t <= 3.50 AND neuron 0 active in the state before this step AND neuron 6 active in the state before this step (17% of its ON cases)<br>+ 1 rarer case(s), 9% of its ON cases | 0.93 | changes the output by 0.68 and flips 30% of decisions; lowers class 3, raises class 1; flips go rank 2 → rank 1 (40%), rank 2 → rank 3 (28%) | 75x |
| W_h.15 | 31% | ON when step t > 3.50 AND neuron 6 active in the state before this step (100% of its ON cases)<br>+ 1 rarer case(s), 0% of its ON cases | 0.94 | changes the output by 0.56 and flips 20% of decisions; lowers class 1, raises class 3; flips go rank 2 → rank 3 (57%), rank 2 → rank 1 (11%) | 62x |
| W_h.4 | 28% | ON when neuron 7 active in the state before this step AND steps since the max <= 1.50 (96% of its ON cases)<br>+ 1 rarer case(s), 4% of its ON cases | 0.99 | changes the output by 0.87 and flips 28% of decisions; lowers class 4, raises class 1; flips go rank 2 → rank 1 (46%), rank 2 → rank 3 (34%) | 266x |
| W_h.16 | 20% | ON when step t > 4.50 (71% of its ON cases)<br>ON when 3.50 < step t <= 4.50 AND not (neuron 3 active in the state before this step) AND steps since the min <= 2.50 (18% of its ON cases)<br>+ 2 rarer case(s), 12% of its ON cases | 0.92 | changes the output by 0.50 and flips 23% of decisions; lowers class 0, raises class 3; flips go rank 2 → rank 3 (42%), rank 2 → rank 1 (19%) | 36x |
| W_h.9 | 12% | ON when 1.50 < steps since the max <= 3.00 AND not (neuron 0 active in the state before this step) AND neuron 2 active in the state before this step (76% of its ON cases)<br>ON when steps since the max <= 1.50 AND steps since the 2nd > 1.50 AND not (neuron 0 active in the state before this step) AND x_t > 0.64 (18% of its ON cases)<br>+ 2 rarer case(s), 6% of its ON cases | 0.93 | changes the output by 0.26 and flips 6% of decisions; lowers class 2, raises class 1; flips go rank 2 → rank 3 (43%), rank 3 → rank 2 (32%) | 19x |
| W_h.1 | 10% | ON when 2.50 < steps since the max <= 3.50 AND not (neuron 2 active in the state before this step) (60% of its ON cases)<br>ON when steps since the max <= 2.50 AND steps since the 2nd > 2.50 AND not (neuron 2 active in the state before this step) (31% of its ON cases)<br>+ 1 rarer case(s), 9% of its ON cases | 0.95 | changes the output by 0.24 and flips 7% of decisions; lowers class 1, raises class 4; flips go rank 2 → rank 3 (36%), rank 2 → rank 1 (22%) | 16x |

## W_o: 7 mechanisms (2 with a trusted trigger)

Alive mechanisms reproduce W_o to a relative error of 0.028.

| mechanism | in use | trigger (when it is needed) | confidence | removing it where needed | specificity |
|---|---|---|---|---|---|
| W_o.0 | 83% | ON when x_t minus the previous min > 1.00 AND neuron 1 active in the final state (51% of its ON cases)<br>ON when x_t minus the previous min <= 1.00 AND x_t minus the previous max <= -1.11 AND 1.50 < steps since the 2nd <= 2.50 (18% of its ON cases)<br>ON when x_t minus the previous min > 1.00 AND not (neuron 1 active in the final state) AND neuron 2 active in the final state AND not (neuron 0 active in the final state) (14% of its ON cases)<br>+ 3 rarer case(s), 17% of its ON cases | 0.62 | changes the output by 1.13 and flips 40% of decisions; lowers class 1, raises class 2; flips go rank 2 → rank 3 (42%), rank 2 → rank 1 (27%) | 24x |
| W_o.10 | 82% | ON when steps since the 2nd <= 0.50 AND steps since the min <= 3.50 (39% of its ON cases)<br>ON when steps since the 2nd > 3.50 AND distance of x_t's rank from the middle <= 1.50 (25% of its ON cases)<br>ON when steps since the 2nd > 3.50 AND distance of x_t's rank from the middle > 1.50 AND neuron 4 active in the final state (16% of its ON cases)<br>+ 3 rarer case(s), 19% of its ON cases | 0.66 | changes the output by 1.09 and flips 41% of decisions; lowers class 4, raises class 3; flips go rank 2 → rank 3 (39%), rank 2 → rank 1 (26%) | 18x |
| W_o.1 | 81% | ON when x_t minus the previous max <= -0.02 AND not (neuron 4 active in the final state) (37% of its ON cases)<br>ON when x_t minus the previous max <= -1.50 AND neuron 4 active in the final state AND x_t > -1.24 (27% of its ON cases)<br>ON when -1.50 < x_t minus the previous max <= -0.02 AND neuron 4 active in the final state AND x_t minus the previous min > 1.17 (24% of its ON cases)<br>+ 3 rarer case(s), 12% of its ON cases | 0.65 | changes the output by 0.94 and flips 44% of decisions; lowers class 3, raises class 1; flips go rank 2 → rank 3 (39%), rank 2 → rank 1 (21%) | 20x |
| W_o.14 | 65% | ON when x_t minus the previous max <= -0.52 AND neuron 3 active in the final state AND x_t minus the previous 2nd <= -0.07 (87% of its ON cases)<br>+ 2 rarer case(s), 13% of its ON cases | 0.95 | changes the output by 1.19 and flips 30% of decisions; lowers class 1, raises class 4; flips go rank 2 → rank 3 (44%), rank 2 → rank 4 (19%) | 23x |
| W_o.5 | 64% | ON when neuron 4 active in the final state AND steps since the max <= 2.50 AND x_t minus the previous min > 1.20 (35% of its ON cases)<br>ON when neuron 4 active in the final state AND 2.50 < steps since the max <= 3.50 (30% of its ON cases)<br>ON when neuron 4 active in the final state AND steps since the max <= 2.50 AND x_t minus the previous min <= 1.20 AND x_t minus the previous 2nd <= -0.50 (16% of its ON cases)<br>+ 3 rarer case(s), 20% of its ON cases | 0.65 | changes the output by 0.72 and flips 30% of decisions; lowers class 4, raises class 0; flips go rank 2 → rank 3 (27%), rank 2 → rank 1 (26%) | 12x |
| W_o.16 | 22% | ON when neuron 2 active in the final state AND 1.50 < steps since the min <= 2.50 (40% of its ON cases)<br>ON when not (neuron 2 active in the final state) AND x_t minus the previous max <= -0.30 AND steps since the max <= 2.50 AND neuron 0 active in the final state (35% of its ON cases)<br>ON when neuron 2 active in the final state AND steps since the min > 2.50 AND x_t > 0.51 (22% of its ON cases)<br>+ 1 rarer case(s), 3% of its ON cases | 0.79 | changes the output by 0.52 and flips 9% of decisions; lowers class 1, raises class 2; flips go rank 2 → rank 3 (51%), rank 3 → rank 2 (15%) | 7x |
| W_o.2 | 18% | ON when x_t minus the previous max > 0.01 AND x_t > 0.11 (97% of its ON cases)<br>+ 1 rarer case(s), 3% of its ON cases | 0.98 | changes the output by 1.00 and flips 27% of decisions; lowers class 0, raises class 4; flips go rank 2 → rank 3 (42%), rank 2 → rank 1 (27%) | 73x |
