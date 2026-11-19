# Mechanism atlas: zoo_2nd_argmax_8_5_s1 (original)

The network's weights are split EXACTLY into rank-1 mechanisms (SPD). Each mechanism has a readable trigger (when it is needed), scored on held-out inputs (its confidence), and an effect (what removing it changes).

## Whole-network certificate

- Rebuilt from its mechanisms, each switched on by its trigger where the trigger is trusted (confidence >= 0.9 on clear-cut cases; otherwise always on), and switched off only where the trigger gives it under a 5% chance of being needed: keeps **91.2%** of the network's decisions; the task rule alone keeps 86.7%.
- The mechanisms alone (all on, the small leftover residual removed): 94.8%. Every trigger used (no confidence bar): 61.8%. Mechanisms switched by their own learned importances: 85.6%. Earlier gates, for the record: v1 87.6%, v2 91.1%.

## Findings

1. The weights split exactly into 27 mechanisms; 14 have a trusted trigger (held-out confidence >= 0.9 on clear-cut cases), 7 are always needed.
1b. 13 of the 20 mechanisms that are not always on have a task-level explanation (a trigger in task variables only, confidence >= 0.9 on clear-cut cases).
2. 14 of the 14 trusted triggers use a task variable (not only neuron states). Most used: step t (9), x_t (7), steps since the max (7), steps since the 2nd (6), x_t minus the previous max (3).
3. 5 mechanism(s) (W_h.1, W_h.3, W_h.5, W_h.10, W_h.14) keep the network from answering rank 1 instead of rank 2: removing them where needed mostly flips the answer from rank 2 to rank 1.
3. 2 mechanism(s) (W_h.15, W_h.16) keep the network from answering rank 3 instead of rank 2: removing them where needed mostly flips the answer from rank 2 to rank 3.
4. Rebuilt from mechanisms switched by trusted triggers: 91.2% of decisions (task rule 86.7%).

## W_x: 10 mechanisms (10 with a trusted trigger)

Alive mechanisms reproduce W_x to a relative error of 0.014.

| mechanism | in use | trigger (when it is needed) | confidence (clear-cut) | removing it where needed | specificity |
|---|---|---|---|---|---|
| W_x.5 | 100% | always | 1.00 | changes the output by 0.83 and flips 32% of decisions; lowers class 4, raises class 0; flips go rank 2 → rank 3 (34%), rank 2 → rank 1 (28%) | only where needed |
| W_x.7 | 100% | always | 1.00 | changes the output by 1.02 and flips 32% of decisions; lowers class 3, raises class 4; flips go rank 2 → rank 3 (33%), rank 2 → rank 1 (31%) | only where needed |
| W_x.8 | 100% | always | 1.00 | changes the output by 1.07 and flips 39% of decisions; lowers class 4, raises class 0; flips go rank 2 → rank 1 (32%), rank 2 → rank 3 (30%) | only where needed |
| W_x.11 | 100% | always | 1.00 | changes the output by 0.59 and flips 25% of decisions; lowers class 4, raises class 0; flips go rank 2 → rank 3 (32%), rank 2 → rank 1 (29%) | only where needed |
| W_x.12 | 100% | always | 1.00 | changes the output by 1.17 and flips 40% of decisions; lowers class 4, raises class 0; flips go rank 2 → rank 3 (32%), rank 2 → rank 1 (28%) | only where needed |
| W_x.15 | 100% | always | 1.00 | changes the output by 1.25 and flips 41% of decisions; lowers class 4, raises class 0; flips go rank 2 → rank 1 (31%), rank 2 → rank 3 (31%) | only where needed |
| W_x.19 | 100% | always | 1.00 | changes the output by 0.25 and flips 12% of decisions; lowers class 4, raises class 3; flips go rank 2 → rank 1 (32%), rank 2 → rank 3 (31%) | only where needed |
| W_x.14 | 98% | task-level: ON when x_t > 0.02 (50% of its ON cases)<br>ON when x_t <= -0.04 (50% of its ON cases) | 1.00 | changes the output by 0.85 and flips 34% of decisions; lowers class 0, raises class 1; flips go rank 2 → rank 3 (31%), rank 2 → rank 1 (28%) | only where needed |
| W_x.1 | 72% | task-level: ON when x_t > 0.35 (51% of its ON cases)<br>ON when x_t <= -0.38 (49% of its ON cases) | 1.00 | changes the output by 0.48 and flips 23% of decisions; lowers class 0, raises class 2; flips go rank 2 → rank 3 (33%), rank 2 → rank 1 (28%) | only where needed |
| W_x.0 | 51% | task-level: ON when x_t > 0.61 (53% of its ON cases)<br>ON when x_t <= -0.69 (47% of its ON cases) | 1.00 | changes the output by 0.24 and flips 12% of decisions; lowers class 3, raises class 2; flips go rank 2 → rank 3 (40%), rank 2 → rank 1 (18%) | 7x |

## W_h: 12 mechanisms (11 with a trusted trigger)

Alive mechanisms reproduce W_h to a relative error of 0.091.

| mechanism | in use | trigger (when it is needed) | confidence (clear-cut) | removing it where needed | specificity |
|---|---|---|---|---|---|
| W_h.0 | 77% | task-level: ON when step t > 1.50 | 0.96 | changes the output by 2.18 and flips 62% of decisions; lowers class 4, raises class 0; flips go rank 2 → rank 1 (26%), rank 2 → rank 3 (26%) | 830x |
| W_h.9 | 74% | task-level: ON when step t > 1.50 | 0.94 | changes the output by 1.05 and flips 43% of decisions; lowers class 4, raises class 1; flips go rank 2 → rank 1 (39%), rank 2 → rank 3 (30%) | 291x |
| W_h.14 | 74% | task-level: ON when step t > 2.50 (92% of its ON cases)<br>+ 2 rarer case(s), 8% of its ON cases | 0.93 | changes the output by 1.30 and flips 41% of decisions; lowers class 0, raises class 2; flips go rank 2 → rank 1 (44%), rank 2 → rank 3 (24%) | 107x |
| W_h.13 | 70% | ON when step t > 1.50 | 0.90 | changes the output by 0.58 and flips 30% of decisions; lowers class 3, raises class 1; flips go rank 2 → rank 3 (35%), rank 2 → rank 1 (24%) | 124x |
| W_h.1 | 52% | ON when x_t minus the previous max <= -0.00 AND not (neuron 4 active in the state before this step) AND steps since the max <= 3.50 (40% of its ON cases)<br>ON when x_t minus the previous max <= -0.00 AND neuron 4 active in the state before this step AND 1.50 < steps since the max <= 3.50 (39% of its ON cases)<br>ON when x_t minus the previous max > -0.00 AND neuron 7 active in the state before this step AND not (neuron 4 active in the state before this step) (11% of its ON cases)<br>+ 2 rarer case(s), 11% of its ON cases | 0.93 | changes the output by 0.64 and flips 30% of decisions; lowers class 4, raises class 3; flips go rank 2 → rank 1 (56%), rank 2 → rank 3 (23%) | 122x |
| W_h.16 | 44% | task-level: ON when step t > 2.50 AND steps since the max <= 1.50 AND steps since the 2nd > 1.50 (41% of its ON cases)<br>ON when step t > 2.50 AND steps since the max > 1.50 AND x_t <= -0.23 AND x_t minus the previous max <= -0.54 (29% of its ON cases)<br>ON when step t > 2.50 AND steps since the max > 1.50 AND x_t > -0.23 (28% of its ON cases)<br>+ 3 rarer case(s), 2% of its ON cases (clear-cut confidence 0.90) | 0.91 | changes the output by 1.98 and flips 56% of decisions; lowers class 1, raises class 4; flips go rank 2 → rank 3 (44%), rank 2 → rank 4 (22%) | 121x |
| W_h.3 | 44% | task-level: ON when step t > 3.50 AND steps since the max <= 1.50 (40% of its ON cases)<br>ON when step t > 2.50 AND steps since the max > 1.50 AND x_t <= -0.23 AND x_t minus the previous max <= -0.54 (31% of its ON cases)<br>ON when step t > 2.50 AND steps since the max > 1.50 AND x_t > -0.23 (30% of its ON cases) (clear-cut confidence 0.92) | 0.95 | changes the output by 1.31 and flips 47% of decisions; lowers class 0, raises class 1; flips go rank 2 → rank 1 (41%), rank 2 → rank 3 (30%) | 109x |
| W_h.18 | 34% | task-level: ON when step t > 2.50 AND steps since the max <= 1.50 (71% of its ON cases)<br>ON when step t > 2.50 AND steps since the max > 2.50 (29% of its ON cases) (clear-cut confidence 0.91) | 0.98 | changes the output by 0.97 and flips 39% of decisions; lowers class 3, raises class 2; flips go rank 2 → rank 3 (30%), rank 2 → rank 4 (21%) | 84x |
| W_h.5 | 28% | task-level: ON when step t > 3.50 AND steps since the max <= 2.50 (63% of its ON cases)<br>ON when step t > 3.50 AND steps since the max > 2.50 AND steps since the 2nd <= 2.50 (26% of its ON cases)<br>+ 3 rarer case(s), 10% of its ON cases (clear-cut confidence 0.91) | 0.91 | changes the output by 0.34 and flips 16% of decisions; lowers class 4, raises class 1; flips go rank 2 → rank 1 (49%), rank 2 → rank 3 (15%) | 31x |
| W_h.10 | 23% | task-level: ON when steps since the max <= 1.50 AND steps since the 2nd > 1.50 AND x_t minus the running median > 0.00 AND x_t > 0.20 (31% of its ON cases)<br>ON when 1.50 < steps since the max <= 2.50 AND x_t > -0.29 (28% of its ON cases)<br>ON when 1.50 < steps since the max <= 2.50 AND x_t <= -0.29 AND x_t minus the previous max <= -0.54 (26% of its ON cases)<br>+ 1 rarer case(s), 15% of its ON cases (clear-cut confidence 0.96) | 0.97 | changes the output by 0.99 and flips 45% of decisions; lowers class 4, raises class 0; flips go rank 2 → rank 1 (44%), rank 2 → rank 3 (24%) | 110x |
| W_h.6 | 22% | task-level: ON when step t > 3.50 AND steps since the max <= 2.50 AND steps since the min <= 3.50 AND steps since the 2nd > 1.50 (47% of its ON cases)<br>ON when step t > 3.50 AND steps since the max > 2.50 AND steps since the 2nd <= 2.50 (36% of its ON cases)<br>+ 3 rarer case(s), 17% of its ON cases (clear-cut confidence 0.93) | 0.93 | changes the output by 0.50 and flips 25% of decisions; lowers class 4, raises class 0; flips go rank 2 → rank 1 (35%), rank 2 → rank 3 (34%) | 28x |
| W_h.15 | 12% | task-level: ON when steps since the max > 2.50 AND x_t minus the previous max > -1.29 AND x_t > -0.91 (39% of its ON cases)<br>ON when steps since the max > 2.50 AND x_t minus the previous max <= -1.29 AND x_t > -1.25 (31% of its ON cases)<br>ON when steps since the max <= 2.50 AND steps since the 2nd > 2.50 AND x_t minus the running median > 0.15 AND x_t > 0.49 (19% of its ON cases)<br>+ 2 rarer case(s), 10% of its ON cases (clear-cut confidence 0.98) | 0.99 | changes the output by 0.42 and flips 19% of decisions; lowers class 2, raises class 0; flips go rank 2 → rank 3 (48%), rank 2 → rank 1 (20%) | 35x |

## W_o: 5 mechanisms (0 with a trusted trigger)

Alive mechanisms reproduce W_o to a relative error of 0.011.

| mechanism | in use | trigger (when it is needed) | confidence (clear-cut) | removing it where needed | specificity |
|---|---|---|---|---|---|
| W_o.6 | 92% | ON when x_t minus the previous max <= -1.13 AND x_t minus the previous min <= 2.01 (77% of its ON cases)<br>ON when x_t minus the previous max > -1.13 AND steps since the min <= 1.50 AND neuron 1 active in the final state (14% of its ON cases)<br>+ 2 rarer case(s), 9% of its ON cases | 0.80 | changes the output by 1.80 and flips 51% of decisions; lowers class 2, raises class 1; flips go rank 2 → rank 1 (37%), rank 2 → rank 3 (32%) | 30x |
| W_o.19 | 86% | ON when steps since the 2nd > 2.50 AND x_t minus the running median > -0.71 (57% of its ON cases)<br>ON when steps since the 2nd <= 0.50 AND x_t minus the running median > 0.13 (30% of its ON cases)<br>+ 2 rarer case(s), 13% of its ON cases | 0.74 | changes the output by 1.27 and flips 51% of decisions; lowers class 4, raises class 3; flips go rank 2 → rank 3 (34%), rank 2 → rank 1 (27%) | 19x |
| W_o.9 | 83% | ON when x_t minus the previous max <= -0.84 AND x_t minus the previous min <= 0.90 AND x_t minus the previous 2nd <= -0.41 (56% of its ON cases)<br>ON when x_t minus the previous max <= -0.84 AND x_t minus the previous min > 0.90 AND steps since the max > 1.50 (19% of its ON cases)<br>ON when x_t minus the previous max > -0.84 AND x_t minus the previous 2nd > 1.44 AND neuron 0 active in the final state AND x_t > 1.28 (7% of its ON cases)<br>+ 4 rarer case(s), 18% of its ON cases | 0.85 | changes the output by 1.42 and flips 46% of decisions; lowers class 3, raises class 2; flips go rank 2 → rank 3 (32%), rank 2 → rank 1 (25%) | 24x |
| W_o.1 | 78% | ON when x_t minus the previous min > 1.39 AND neuron 5 active in the final state (55% of its ON cases)<br>ON when 0.66 < x_t minus the previous min <= 1.39 AND x_t minus the running median > -1.46 AND x_t minus the previous 2nd <= -0.06 (23% of its ON cases)<br>ON when x_t minus the previous min > 1.39 AND not (neuron 5 active in the final state) AND x_t minus the previous max > -1.41 (13% of its ON cases)<br>+ 3 rarer case(s), 9% of its ON cases | 0.66 | changes the output by 0.84 and flips 50% of decisions; lowers class 3, raises class 1; flips go rank 2 → rank 3 (32%), rank 2 → rank 1 (27%) | 16x |
| W_o.16 | 76% | ON when x_t minus the previous min > 1.41 AND steps since the 2nd <= 3.50 AND x_t minus the previous max <= 0.52 (50% of its ON cases)<br>ON when x_t minus the previous min <= 1.41 AND x_t minus the previous 2nd <= -1.25 AND steps since the 2nd > 2.50 (18% of its ON cases)<br>ON when 0.59 < x_t minus the previous min <= 1.41 AND x_t minus the previous 2nd > -1.25 AND x_t minus the previous max <= -1.16 (16% of its ON cases)<br>+ 3 rarer case(s), 16% of its ON cases | 0.69 | changes the output by 0.81 and flips 50% of decisions; lowers class 1, raises class 2; flips go rank 2 → rank 1 (34%), rank 2 → rank 3 (27%) | 17x |
