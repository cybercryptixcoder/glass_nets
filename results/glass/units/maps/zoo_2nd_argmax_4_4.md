# Unit map: zoo_2nd_argmax_4_4

Coverage after the first step (variance-weighted share of neuron-steps fitted at R² ≥ 0.9 on held-out inputs): history units 0.00, raw-input units 0.02. Shuffled-weights control: history 0.00, raw 0.01.

## Per neuron (most common label over its explained steps)

| neuron | label |
|---|---|
| 0 | raw input: relu(+1.19*x_t -0.79*x_prev) |
| 1 | unexplained |
| 2 | raw input: relu(-2.29*x_t +0.51*x_prev) |
| 3 | relu(+0.56*max +0.31*2nd) |

## Every neuron-step

| step | neuron | unit | fit (held-out R²) | sufficiency | importance |
|---|---|---|---|---|---|
| 1 | 0 | raw input: relu(+1.00*x_t) | 1.000 | 100.0% | 51.3% |
| 1 | 1 | raw input: relu(-2.22*x_t) | 1.000 | 100.0% | 66.3% |
| 1 | 2 | raw input: relu(-2.04*x_t) | 1.000 | 100.0% | 60.1% |
| 1 | 3 | raw input: relu(+0.39*x_t) | 1.000 | 100.0% | 23.8% |
| 2 | 0 | raw input: relu(+1.19*x_t -0.79*x_prev) | 0.946 | 92.2% | 40.3% |
| 2 | 1 | comparator: relu(max - x_t) | 0.899 | 75.9% | 62.4% |
| 2 | 2 | raw input: relu(-2.29*x_t +0.51*x_prev) | 0.967 | 87.2% | 55.0% |
| 2 | 3 | relu(+0.56*max +0.31*2nd) | 0.962 | 94.9% | 34.7% |
| 3 | 0 | filter: relu(+3.41*x_1 -1.99*x_2 +1.74*x_3) | 0.791 | 72.2% | 52.7% |
| 3 | 1 | comparator: relu(max - x_t) | 0.715 | 61.2% | 56.9% |
| 3 | 2 | filter: relu(+0.39*x_1 +0.78*x_2 -2.29*x_3) | 0.938 | 86.6% | 60.5% |
| 3 | 3 | relu(-1.83*x_prev +3.38*max_prev) | 0.705 | 51.3% | 59.0% |
| 4 | 0 | relu(+4.56*max -3.08*x_prev) | 0.366 | 53.2% | 59.3% |
| 4 | 1 | relu(-2.38*x_t +10.48*max_prev) | 0.331 | 43.0% | 60.3% |
| 4 | 2 | filter: relu(+4.70*x_1 -0.84*x_2 +1.61*x_3 -3.02*x_4) | 0.554 | 55.5% | 61.6% |
| 4 | 3 | relu(-2.99*x_prev +5.82*max_prev) | 0.358 | 15.4% | 83.5% |

*Sufficiency: decisions kept when only this neuron-step is replaced by its unit formula. Importance: decisions lost when it is set to its mean. Both are on held-out inputs. Units are approximate in dense networks (UNITS_PROGRAM, first-sweep conclusion), so a high fit does not imply exact sufficiency.*
