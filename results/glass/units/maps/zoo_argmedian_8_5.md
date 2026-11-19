# Unit map: zoo_argmedian_8_5

Coverage after the first step (variance-weighted share of neuron-steps fitted at R² ≥ 0.9 on held-out inputs): history units 0.01, raw-input units 0.13. Shuffled-weights control: history 0.00, raw 0.01.

## Per neuron (most common label over its explained steps)

| neuron | label |
|---|---|
| 0 | unexplained |
| 1 | raw input: relu(+1.67*x_t -1.57*x_prev) |
| 2 | raw input: relu(+1.64*x_t -0.66*x_prev) |
| 3 | raw input: relu(+0.25*x_t -0.10*x_prev) |
| 4 | raw input: relu(+2.67*x_t -1.06*x_prev) |
| 5 | unexplained |
| 6 | unexplained |
| 7 | raw input: relu(-1.36*x_t +0.87*x_prev) |

## Every neuron-step

| step | neuron | unit | fit (held-out R²) | sufficiency | importance |
|---|---|---|---|---|---|
| 1 | 0 | raw input: relu(-0.17*x_t) | 1.000 | 100.0% | 11.7% |
| 1 | 1 | raw input: relu(+1.51*x_t) | 1.000 | 100.0% | 55.6% |
| 1 | 2 | raw input: relu(+1.50*x_t) | 1.000 | 100.0% | 53.7% |
| 1 | 3 | raw input: relu(+0.23*x_t) | 1.000 | 100.0% | 14.0% |
| 1 | 4 | raw input: relu(+2.43*x_t) | 1.000 | 100.0% | 68.3% |
| 1 | 5 | raw input: relu(-0.03*x_t) | 1.000 | 100.0% | 3.1% |
| 1 | 6 | raw input: relu(+0.22*x_t) | 1.000 | 100.0% | 7.4% |
| 1 | 7 | raw input: relu(-1.16*x_t) | 1.000 | 100.0% | 38.7% |
| 2 | 0 | comparator: relu(max - x_t) | 0.626 | 39.3% | 70.7% |
| 2 | 1 | raw input: relu(+1.67*x_t -1.57*x_prev) | 0.987 | 95.9% | 37.3% |
| 2 | 2 | raw input: relu(+1.64*x_t -0.66*x_prev) | 0.983 | 83.2% | 52.3% |
| 2 | 3 | raw input: relu(+0.25*x_t -0.10*x_prev) | 0.988 | 98.7% | 18.4% |
| 2 | 4 | raw input: relu(+2.67*x_t -1.06*x_prev) | 0.982 | 82.5% | 62.1% |
| 2 | 5 | comparator: relu(max - x_t) | 0.620 | 59.3% | 48.9% |
| 2 | 6 | relu(-2.03*x_t +4.48*max) | 0.518 | 30.4% | 70.5% |
| 2 | 7 | raw input: relu(-1.36*x_t +0.87*x_prev) | 0.953 | 90.8% | 48.8% |
| 3 | 0 | relu(+4.97*max_prev -2.63*2nd_prev) | 0.771 | 40.9% | 64.5% |
| 3 | 1 | raw input: relu(+1.76*x_t -1.49*x_prev) | 0.962 | 93.4% | 30.9% |
| 3 | 2 | raw input: relu(+1.88*x_t -0.56*x_prev) | 0.876 | 92.5% | 30.3% |
| 3 | 3 | relu(-1.81*x_prev +3.48*max_prev) | 0.306 | 44.1% | 61.7% |
| 3 | 4 | filter: relu(-1.09*x_1 -0.72*x_2 +2.82*x_3) | 0.953 | 90.1% | 53.2% |
| 3 | 5 | relu(-0.61*min +1.02*max_prev) | 0.627 | 58.2% | 51.4% |
| 3 | 6 | relu(+5.08*max_prev -3.50*2nd_prev) | 0.673 | 31.7% | 69.7% |
| 3 | 7 | comparator: relu(max - x_t) | 0.958 | 91.4% | 50.4% |
| 4 | 0 | relu(-2.25*min +3.81*max_prev) | 0.637 | 57.7% | 49.5% |
| 4 | 1 | raw input: relu(+1.68*x_t -1.56*x_prev) | 0.960 | 91.3% | 34.6% |
| 4 | 2 | filter: relu(-0.80*x_1 +0.29*x_2 -0.51*x_3 +1.79*x_4) | 0.888 | 92.8% | 32.7% |
| 4 | 3 | relu(-1.32*min +1.80*max_prev) | 0.258 | 54.9% | 48.9% |
| 4 | 4 | filter: relu(-0.67*x_1 -0.86*x_2 -0.74*x_3 +2.74*x_4) | 0.936 | 87.8% | 53.3% |
| 4 | 5 | relu(-1.28*min +1.61*max_prev) | 0.515 | 69.2% | 38.9% |
| 4 | 6 | relu(-2.56*min +3.30*max_prev) | 0.406 | 53.2% | 55.9% |
| 4 | 7 | filter: relu(+0.33*x_1 +0.44*x_2 +0.71*x_3 -1.48*x_4) | 0.825 | 83.3% | 48.8% |
| 5 | 0 | relu(-3.45*min +5.14*max_prev) | 0.496 | 84.6% | 20.8% |
| 5 | 1 | raw input: relu(+1.62*x_t -1.52*x_prev) | 0.968 | 87.5% | 39.2% |
| 5 | 2 | filter: relu(-0.57*x_1 -0.62*x_2 +0.35*x_3 -0.44*x_4 +1.60*x_5) | 0.912 | 90.8% | 39.2% |
| 5 | 3 | relu(+1.41*max -0.92*min) | 0.138 | 57.9% | 42.5% |
| 5 | 4 | filter: relu(-0.54*x_1 -0.59*x_2 -0.70*x_3 -0.69*x_4 +2.56*x_5) | 0.946 | 87.1% | 59.9% |
| 5 | 5 | relu(-1.00*min +1.38*max_prev) | 0.379 | 80.1% | 24.5% |
| 5 | 6 | relu(-3.94*min +2.78*x_prev) | 0.324 | 59.7% | 45.3% |
| 5 | 7 | comparator: relu(2nd - x_t) | 0.859 | 87.0% | 29.9% |

*Sufficiency: decisions kept when only this neuron-step is replaced by its unit formula. Importance: decisions lost when it is set to its mean. Both are on held-out inputs. Units are approximate in dense networks (UNITS_PROGRAM, first-sweep conclusion), so a high fit does not imply exact sufficiency.*
