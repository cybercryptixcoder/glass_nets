# Mechanism atlas: zoo_argmedian_8_7 (original)

The network's weights are split EXACTLY into rank-1 mechanisms (SPD). Each mechanism has a readable trigger (when it is needed), scored on held-out inputs (its confidence), and an effect (what removing it changes).

## Whole-network certificate

- Rebuilt from its mechanisms, each switched on by its trigger where the trigger's confidence is >= 0.9 (otherwise always on): keeps **88.4%** of the network's decisions; the task rule alone keeps 49.5%.
- Every trigger used (no confidence bar): 36.1%. Mechanisms switched by their own learned importances: 78.8%. All alive mechanisms on: 100.0%.

## Findings

1. The weights split exactly into 20 mechanisms; 7 have a trusted trigger (held-out confidence >= 0.9), 2 are always needed.
2. 7 of the 7 trusted triggers use a task variable (not only neuron states). Most used: step t (7), x_t (3), x_t minus the previous max (2), x_t minus the previous 2nd (2), steps since the 2nd (2).
4. Rebuilt from mechanisms switched by trusted triggers: 88.4% of decisions (task rule 49.5%).

## W_x: 2 mechanisms (2 with a trusted trigger)

Alive mechanisms reproduce W_x to a relative error of 0.005.

| mechanism | in use | trigger (when it is needed) | confidence | removing it where needed | specificity |
|---|---|---|---|---|---|
| W_x.4 | 100% | always | 1.00 | changes the output by 1.26 and flips 53% of decisions; lowers class 6, raises class 5; flips go rank 4 → rank 5 (15%), rank 4 → rank 3 (13%) | only where needed |
| W_x.11 | 100% | always | 1.00 | changes the output by 0.87 and flips 39% of decisions; lowers class 6, raises class 2; flips go rank 4 → rank 3 (17%), rank 4 → rank 5 (14%) | only where needed |

## W_h: 13 mechanisms (7 with a trusted trigger)

Alive mechanisms reproduce W_h to a relative error of 0.060.

| mechanism | in use | trigger (when it is needed) | confidence | removing it where needed | specificity |
|---|---|---|---|---|---|
| W_h.19 | 85% | ON when step t > 1.50 | 0.98 | changes the output by 2.19 and flips 67% of decisions; lowers class 6, raises class 2; flips go rank 4 → rank 5 (15%), rank 4 → rank 6 (13%) | 1435x |
| W_h.10 | 83% | ON when neuron 6 active in the state before this step AND step t > 2.50 (86% of its ON cases)<br>+ 4 rarer case(s), 14% of its ON cases | 0.95 | changes the output by 2.23 and flips 60% of decisions; lowers class 6, raises class 2; flips go rank 4 → rank 3 (19%), rank 4 → rank 2 (14%) | 508x |
| W_h.8 | 74% | ON when step t > 3.50 (82% of its ON cases)<br>+ 3 rarer case(s), 18% of its ON cases | 0.91 | changes the output by 1.22 and flips 51% of decisions; lowers class 6, raises class 3; flips go rank 4 → rank 5 (21%), rank 4 → rank 3 (11%) | 80x |
| W_h.17 | 67% | ON when step t > 3.50 AND neuron 0 active in the state before this step (81% of its ON cases)<br>+ 2 rarer case(s), 19% of its ON cases | 0.95 | changes the output by 1.38 and flips 50% of decisions; lowers class 5, raises class 2; flips go rank 4 → rank 3 (15%), rank 4 → rank 5 (13%) | 204x |
| W_h.3 | 66% | ON when step t > 3.50 (81% of its ON cases)<br>+ 1 rarer case(s), 19% of its ON cases | 0.94 | changes the output by 1.38 and flips 44% of decisions; lowers class 5, raises class 6; flips go rank 4 → rank 5 (16%), rank 4 → rank 3 (13%) | 142x |
| W_h.6 | 63% | ON when neuron 5 active in the state before this step AND step t > 3.50 (81% of its ON cases)<br>+ 1 rarer case(s), 19% of its ON cases | 0.87 | changes the output by 0.63 and flips 36% of decisions; lowers class 3, raises class 0; flips go rank 4 → rank 3 (18%), rank 4 → rank 5 (15%) | 65x |
| W_h.4 | 61% | ON when step t > 3.50 (86% of its ON cases)<br>+ 2 rarer case(s), 14% of its ON cases | 0.89 | changes the output by 1.24 and flips 48% of decisions; lowers class 6, raises class 3; flips go rank 4 → rank 3 (19%), rank 4 → rank 5 (12%) | 77x |
| W_h.12 | 54% | ON when step t > 3.50 (99% of its ON cases)<br>+ 2 rarer case(s), 1% of its ON cases | 0.95 | changes the output by 1.43 and flips 54% of decisions; lowers class 4, raises class 2; flips go rank 4 → rank 5 (20%), rank 4 → rank 3 (11%) | 59x |
| W_h.9 | 53% | ON when step t <= 4.50 AND neuron 4 active in the state before this step AND not (neuron 1 active in the state before this step) (30% of its ON cases)<br>ON when step t > 5.50 AND neuron 4 active in the state before this step (25% of its ON cases)<br>ON when 4.50 < step t <= 5.50 (23% of its ON cases)<br>ON when step t > 5.50 AND not (neuron 4 active in the state before this step) AND neuron 2 active in the state before this step (21% of its ON cases) | 0.90 | changes the output by 0.69 and flips 35% of decisions; lowers class 4, raises class 2; flips go rank 4 → rank 3 (20%), rank 4 → rank 5 (11%) | 53x |
| W_h.18 | 51% | ON when step t > 3.50 AND neuron 2 active in the state before this step (81% of its ON cases)<br>+ 2 rarer case(s), 19% of its ON cases | 0.84 | changes the output by 0.57 and flips 32% of decisions; lowers class 2, raises class 6; flips go rank 4 → rank 5 (18%), rank 4 → rank 3 (16%) | 38x |
| W_h.15 | 31% | ON when step t > 4.50 AND neuron 2 active in the state before this step (99% of its ON cases)<br>+ 2 rarer case(s), 1% of its ON cases | 0.92 | changes the output by 0.48 and flips 29% of decisions; lowers class 3, raises class 6; flips go rank 4 → rank 3 (21%), rank 5 → rank 3 (10%) | 36x |
| W_h.16 | 16% | ON when step t > 3.50 AND steps since the min > 2.50 (56% of its ON cases)<br>ON when step t > 3.50 AND steps since the min <= 2.50 AND steps since the max > 2.50 (37% of its ON cases)<br>+ 4 rarer case(s), 8% of its ON cases | 0.78 | changes the output by 0.19 and flips 14% of decisions; lowers class 4, raises class 6; flips go rank 4 → rank 3 (20%), rank 4 → rank 5 (11%) | 9x |
| W_h.11 | 6% | ON when step t > 5.50 AND steps since the max <= 4.50 AND steps since the min <= 3.50 (55% of its ON cases)<br>ON when step t > 5.50 AND steps since the max > 4.50 AND neuron 2 active in the state before this step (26% of its ON cases)<br>+ 3 rarer case(s), 18% of its ON cases | 0.89 | changes the output by 0.15 and flips 15% of decisions; lowers class 2, raises class 6; flips go rank 4 → rank 3 (18%), rank 3 → rank 4 (12%) | 13x |

## W_o: 5 mechanisms (0 with a trusted trigger)

Alive mechanisms reproduce W_o to a relative error of 0.048.

| mechanism | in use | trigger (when it is needed) | confidence | removing it where needed | specificity |
|---|---|---|---|---|---|
| W_o.19 | 96% | ON when steps since the min <= 2.50 AND x_t minus the previous min > 0.65 (30% of its ON cases)<br>ON when steps since the min > 3.50 AND steps since the max <= 2.50 (30% of its ON cases)<br>ON when steps since the min <= 2.50 AND x_t minus the previous min <= 0.65 AND x_t minus the previous 2nd <= -1.07 (23% of its ON cases)<br>+ 4 rarer case(s), 16% of its ON cases | 0.81 | changes the output by 1.60 and flips 40% of decisions; lowers class 1, raises class 6; flips go rank 4 → rank 3 (15%), rank 4 → rank 5 (14%) | 21x |
| W_o.6 | 85% | ON when distance of x_t's rank from the middle <= 1.50 AND neuron 7 active in the final state AND steps since the min <= 3.50 (51% of its ON cases)<br>ON when distance of x_t's rank from the middle <= 1.50 AND neuron 7 active in the final state AND steps since the min > 3.50 AND steps since the max <= 3.50 (30% of its ON cases)<br>+ 3 rarer case(s), 18% of its ON cases | 0.68 | changes the output by 1.34 and flips 52% of decisions; lowers class 6, raises class 5; flips go rank 4 → rank 5 (15%), rank 4 → rank 3 (15%) | 19x |
| W_o.2 | 82% | ON when x_t minus the running median > -1.06 AND x_t minus the previous min > 1.86 (61% of its ON cases)<br>ON when x_t minus the running median > -1.06 AND x_t minus the previous min <= 1.86 AND x_t minus the previous max <= -2.38 (15% of its ON cases)<br>ON when x_t minus the running median <= -1.06 AND x_t minus the previous min > -0.59 AND steps since the max <= 5.50 (13% of its ON cases)<br>+ 2 rarer case(s), 11% of its ON cases | 0.56 | changes the output by 0.98 and flips 60% of decisions; lowers class 4, raises class 6; flips go rank 4 → rank 3 (15%), rank 4 → rank 5 (14%) | 15x |
| W_o.11 | 77% | ON when steps since the min > 1.50 AND x_t minus the previous max <= -0.54 AND steps since the max > 1.50 AND x_t minus the previous 2nd <= -1.10 (35% of its ON cases)<br>ON when steps since the min <= 1.50 AND x_t minus the previous min > 0.67 (30% of its ON cases)<br>ON when steps since the min > 1.50 AND x_t minus the previous max <= -0.54 AND steps since the max <= 1.50 (25% of its ON cases)<br>+ 2 rarer case(s), 10% of its ON cases | 0.63 | changes the output by 1.15 and flips 56% of decisions; lowers class 3, raises class 5; flips go rank 4 → rank 3 (13%), rank 4 → rank 5 (13%) | 19x |
| W_o.17 | 68% | ON when 3.50 < steps since the min <= 4.50 (42% of its ON cases)<br>ON when 1.50 < steps since the min <= 2.50 (37% of its ON cases)<br>ON when steps since the min <= 1.50 AND 1.50 < steps since the max <= 2.50 (13% of its ON cases)<br>+ 1 rarer case(s), 8% of its ON cases | 0.65 | changes the output by 0.88 and flips 45% of decisions; lowers class 2, raises class 4; flips go rank 4 → rank 3 (16%), rank 4 → rank 5 (15%) | 13x |
