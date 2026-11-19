# Mechanism atlas: zoo_argmedian_8_7 (original)

The network's weights are split EXACTLY into rank-1 mechanisms (SPD). Each mechanism has a readable trigger (when it is needed), scored on held-out inputs (its confidence), and an effect (what removing it changes).

## Whole-network certificate

- Rebuilt from its mechanisms, each switched on by its trigger where the trigger is trusted (confidence >= 0.9 on clear-cut cases; otherwise always on), and switched off only where the trigger gives it under a 5% chance of being needed: keeps **86.5%** of the network's decisions; the task rule alone keeps 49.5%.
- The mechanisms alone (all on, the small leftover residual removed): 92.4%. Every trigger used (no confidence bar): 35.6%. Mechanisms switched by their own learned importances: 62.8%. Earlier gates, for the record: v1 86.2%, v2 94.4%.

## Findings

1. The weights split exactly into 20 mechanisms; 12 have a trusted trigger (held-out confidence >= 0.9 on clear-cut cases), 2 are always needed.
1b. 10 of the 18 mechanisms that are not always on have a task-level explanation (a trigger in task variables only, confidence >= 0.9 on clear-cut cases).
2. 12 of the 12 trusted triggers use a task variable (not only neuron states). Most used: step t (12), x_t (5), x_t minus the previous max (4), x_t minus the previous min (2), steps since the max (2).
4. Rebuilt from mechanisms switched by trusted triggers: 86.5% of decisions (task rule 49.5%).

## W_x: 2 mechanisms (2 with a trusted trigger)

Alive mechanisms reproduce W_x to a relative error of 0.000.

| mechanism | in use | trigger (when it is needed) | confidence (clear-cut) | removing it where needed | specificity |
|---|---|---|---|---|---|
| W_x.4 | 100% | always | 1.00 | changes the output by 0.51 and flips 21% of decisions; lowers class 1, raises class 5; flips go rank 4 → rank 3 (13%), rank 4 → rank 5 (13%) | only where needed |
| W_x.11 | 100% | always | 1.00 | changes the output by 0.50 and flips 21% of decisions; lowers class 1, raises class 5; flips go rank 4 → rank 3 (13%), rank 4 → rank 5 (13%) | only where needed |

## W_h: 13 mechanisms (12 with a trusted trigger)

Alive mechanisms reproduce W_h to a relative error of 0.000.

| mechanism | in use | trigger (when it is needed) | confidence (clear-cut) | removing it where needed | specificity |
|---|---|---|---|---|---|
| W_h.19 | 85% | task-level: ON when step t > 1.50 | 0.99 | changes the output by 1.07 and flips 47% of decisions; lowers class 6, raises class 4; flips go rank 4 → rank 5 (21%), rank 4 → rank 6 (10%) | 889x |
| W_h.10 | 83% | task-level: ON when step t > 2.50 (96% of its ON cases)<br>+ 2 rarer case(s), 4% of its ON cases (clear-cut confidence 0.95) | 0.97 | changes the output by 0.72 and flips 38% of decisions; lowers class 6, raises class 4; flips go rank 4 → rank 3 (20%), rank 4 → rank 5 (12%) | 428x |
| W_h.8 | 74% | task-level: ON when step t > 3.50 (85% of its ON cases)<br>+ 2 rarer case(s), 15% of its ON cases (clear-cut confidence 0.96) | 0.97 | changes the output by 1.29 and flips 54% of decisions; lowers class 6, raises class 3; flips go rank 4 → rank 5 (23%), rank 4 → rank 6 (11%) | 76x |
| W_h.17 | 67% | task-level: ON when step t > 3.50 (85% of its ON cases)<br>+ 2 rarer case(s), 15% of its ON cases (clear-cut confidence 0.97) | 0.98 | changes the output by 2.18 and flips 73% of decisions; lowers class 5, raises class 2; flips go rank 4 → rank 3 (18%), rank 4 → rank 2 (15%) | 302x |
| W_h.3 | 66% | task-level: ON when step t > 3.50 (85% of its ON cases)<br>+ 2 rarer case(s), 15% of its ON cases (clear-cut confidence 0.96) | 0.96 | changes the output by 1.87 and flips 64% of decisions; lowers class 5, raises class 2; flips go rank 4 → rank 5 (20%), rank 4 → rank 6 (12%) | 86x |
| W_h.6 | 63% | ON when neuron 5 active in the state before this step AND step t > 3.50 (81% of its ON cases)<br>+ 1 rarer case(s), 19% of its ON cases | 0.92 | changes the output by 0.54 and flips 31% of decisions; lowers class 3, raises class 5; flips go rank 4 → rank 3 (18%), rank 4 → rank 5 (13%) | 68x |
| W_h.4 | 61% | task-level: ON when step t > 3.50 (96% of its ON cases)<br>+ 1 rarer case(s), 4% of its ON cases (clear-cut confidence 0.93) | 0.93 | changes the output by 0.85 and flips 44% of decisions; lowers class 6, raises class 5; flips go rank 4 → rank 3 (19%), rank 4 → rank 5 (10%) | 46x |
| W_h.12 | 54% | task-level: ON when step t > 3.50 (99% of its ON cases)<br>+ 2 rarer case(s), 1% of its ON cases | 0.97 | changes the output by 1.37 and flips 56% of decisions; lowers class 4, raises class 2; flips go rank 4 → rank 5 (20%), rank 4 → rank 6 (11%) | 60x |
| W_h.9 | 53% | task-level: ON when step t > 4.50 (73% of its ON cases)<br>ON when step t <= 4.50 AND x_t minus the previous min > 0.03 AND steps since the min <= 1.50 (27% of its ON cases) (clear-cut confidence 0.91) | 0.93 | changes the output by 0.91 and flips 49% of decisions; lowers class 6, raises class 2; flips go rank 4 → rank 3 (23%), rank 5 → rank 3 (10%) | 33x |
| W_h.18 | 51% | ON when step t > 3.50 AND neuron 2 [≈ relu(+0.19*max +0.11*2nd)] active in the state before this step (81% of its ON cases)<br>+ 2 rarer case(s), 19% of its ON cases | 0.91 | changes the output by 0.35 and flips 19% of decisions; lowers class 2, raises class 6; flips go rank 4 → rank 5 (17%), rank 4 → rank 3 (17%) | 36x |
| W_h.15 | 31% | task-level: ON when step t > 4.50 (100% of its ON cases)<br>+ 1 rarer case(s), 0% of its ON cases (clear-cut confidence 0.93) | 0.94 | changes the output by 1.01 and flips 49% of decisions; lowers class 6, raises class 2; flips go rank 4 → rank 3 (25%), rank 5 → rank 3 (10%) | 32x |
| W_h.16 | 16% | ON when step t > 3.50 AND steps since the min > 2.50 (56% of its ON cases)<br>ON when step t > 3.50 AND steps since the min <= 2.50 AND steps since the max > 2.50 (37% of its ON cases)<br>+ 4 rarer case(s), 8% of its ON cases | 0.79 | changes the output by 0.42 and flips 28% of decisions; lowers class 6, raises class 3; flips go rank 4 → rank 3 (20%), rank 5 → rank 3 (12%) | 9x |
| W_h.11 | 6% | task-level: ON when step t > 5.50 AND steps since the max <= 4.50 AND steps since the min <= 3.50 (55% of its ON cases)<br>ON when step t > 5.50 AND steps since the max > 4.50 (27% of its ON cases)<br>+ 3 rarer case(s), 18% of its ON cases (clear-cut confidence 0.92) | 0.92 | changes the output by 0.76 and flips 56% of decisions; lowers class 4, raises class 3; flips go rank 4 → rank 3 (18%), rank 4 → rank 5 (11%) | 10x |

## W_o: 5 mechanisms (0 with a trusted trigger)

Alive mechanisms reproduce W_o to a relative error of 0.048.

| mechanism | in use | trigger (when it is needed) | confidence (clear-cut) | removing it where needed | specificity |
|---|---|---|---|---|---|
| W_o.19 | 96% | ON when steps since the min <= 2.50 AND x_t minus the previous min > 0.65 (30% of its ON cases)<br>ON when steps since the min > 3.50 AND steps since the max <= 2.50 (30% of its ON cases)<br>ON when steps since the min <= 2.50 AND x_t minus the previous min <= 0.65 AND x_t minus the previous 2nd <= -1.07 (23% of its ON cases)<br>+ 4 rarer case(s), 16% of its ON cases | 0.84 | changes the output by 1.58 and flips 41% of decisions; lowers class 1, raises class 6; flips go rank 4 → rank 3 (15%), rank 4 → rank 5 (14%) | 21x |
| W_o.6 | 85% | ON when distance of x_t's rank from the middle <= 1.50 AND neuron 7 active in the final state AND steps since the min <= 3.50 (51% of its ON cases)<br>ON when distance of x_t's rank from the middle <= 1.50 AND neuron 7 active in the final state AND steps since the min > 3.50 AND steps since the max <= 3.50 (30% of its ON cases)<br>+ 3 rarer case(s), 18% of its ON cases | 0.70 | changes the output by 1.32 and flips 52% of decisions; lowers class 6, raises class 5; flips go rank 4 → rank 5 (15%), rank 4 → rank 3 (15%) | 19x |
| W_o.2 | 82% | ON when x_t minus the running median > -1.06 AND x_t minus the previous min > 1.86 (61% of its ON cases)<br>ON when x_t minus the running median > -1.06 AND x_t minus the previous min <= 1.86 AND x_t minus the previous max <= -2.38 (15% of its ON cases)<br>ON when x_t minus the running median <= -1.06 AND x_t minus the previous min > -0.59 AND steps since the max <= 5.50 (13% of its ON cases)<br>+ 2 rarer case(s), 11% of its ON cases | 0.59 | changes the output by 0.97 and flips 60% of decisions; lowers class 4, raises class 6; flips go rank 4 → rank 3 (15%), rank 4 → rank 5 (14%) | 15x |
| W_o.11 | 77% | ON when steps since the min > 1.50 AND x_t minus the previous max <= -0.54 AND steps since the max > 1.50 AND x_t minus the previous 2nd <= -1.10 (35% of its ON cases)<br>ON when steps since the min <= 1.50 AND x_t minus the previous min > 0.67 (30% of its ON cases)<br>ON when steps since the min > 1.50 AND x_t minus the previous max <= -0.54 AND steps since the max <= 1.50 (25% of its ON cases)<br>+ 2 rarer case(s), 10% of its ON cases | 0.66 | changes the output by 1.14 and flips 56% of decisions; lowers class 3, raises class 5; flips go rank 4 → rank 5 (13%), rank 4 → rank 3 (13%) | 19x |
| W_o.17 | 68% | ON when 3.50 < steps since the min <= 4.50 (42% of its ON cases)<br>ON when 1.50 < steps since the min <= 2.50 (37% of its ON cases)<br>ON when steps since the min <= 1.50 AND 1.50 < steps since the max <= 2.50 (13% of its ON cases)<br>+ 1 rarer case(s), 8% of its ON cases | 0.69 | changes the output by 0.86 and flips 44% of decisions; lowers class 2, raises class 4; flips go rank 4 → rank 3 (16%), rank 4 → rank 5 (15%) | 13x |
