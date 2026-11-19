# Unit map: zoo_argmedian_8_7

Coverage after the first step (variance-weighted share of neuron-steps fitted at R² ≥ 0.9 on held-out inputs): history units 0.04, raw-input units 0.11. Shuffled-weights control: history 0.00, raw 0.00.

Precision requirement ε95 = 0.006309573444801929: hidden-state noise (as a share of each neuron's spread) at which 5% of decisions flip. Substitution-grade units (relative error ≤ ε95): 8 of 56 neuron-steps.

## Per neuron (most common label over its explained steps)

| neuron | label |
|---|---|
| 0 | unexplained |
| 1 | comparator: relu(x_t - min) |
| 2 | relu(+0.19*max +0.11*2nd) |
| 3 | raw input: relu(-1.57*x_t +1.51*x_prev) |
| 4 | raw input: relu(-1.99*x_t +0.55*x_prev) |
| 5 | unexplained |
| 6 | unexplained |
| 7 | unexplained |

## Every neuron-step

| step | neuron | unit | fit (held-out R²) | relative error | sufficiency | importance |
|---|---|---|---|---|---|---|
| 1 | 0 | raw input: relu(+0.03*x_t) | 1.000 | 0.000 | 100.0% | 1.5% |
| 1 | 1 | raw input: relu(+1.65*x_t) | 1.000 | 0.000 | 100.0% | 26.9% |
| 1 | 2 | raw input: relu(+0.14*x_t) | 1.000 | 0.000 | 100.0% | 5.2% |
| 1 | 3 | raw input: relu(-1.30*x_t) | 1.000 | 0.000 | 100.0% | 24.8% |
| 1 | 4 | raw input: relu(-1.85*x_t) | 1.000 | 0.000 | 100.0% | 42.6% |
| 1 | 5 | raw input: relu(+0.12*x_t) | 1.000 | 0.000 | 100.0% | 2.4% |
| 1 | 6 | raw input: relu(-0.24*x_t) | 1.000 | 0.000 | 100.0% | 12.3% |
| 1 | 7 | raw input: relu(+0.14*x_t) | 1.000 | 0.000 | 100.0% | 3.0% |
| 2 | 0 | comparator: relu(x_t - 2nd) | 0.581 | 0.647 | 73.9% | 36.4% |
| 2 | 1 | relu(+1.87*x_t -0.89*2nd) | 0.974 | 0.162 | 95.0% | 39.9% |
| 2 | 2 | relu(+0.19*max +0.11*2nd) | 0.961 | 0.197 | 98.5% | 9.0% |
| 2 | 3 | raw input: relu(-1.57*x_t +1.51*x_prev) | 0.960 | 0.200 | 96.9% | 21.3% |
| 2 | 4 | raw input: relu(-1.99*x_t +0.55*x_prev) | 0.988 | 0.108 | 97.9% | 28.5% |
| 2 | 5 | comparator: relu(x_t - 2nd) | 0.435 | 0.752 | 78.0% | 34.5% |
| 2 | 6 | relu(+0.85*max -1.22*2nd) | 0.122 | 0.937 | 39.1% | 62.8% |
| 2 | 7 | comparator: relu(x_t - 2nd) | 0.478 | 0.722 | 68.1% | 45.7% |
| 3 | 0 | relu(+0.59*max -0.98*2nd_prev) | 0.662 | 0.581 | 76.9% | 40.8% |
| 3 | 1 | comparator: relu(x_t - min) | 0.975 | 0.157 | 93.2% | 49.7% |
| 3 | 2 | filter: relu(-0.07*x_1 +0.19*x_2 +0.15*x_3) | 0.902 | 0.313 | 98.5% | 8.3% |
| 3 | 3 | raw input: relu(-1.52*x_t +1.47*x_prev) | 0.961 | 0.197 | 98.1% | 15.1% |
| 3 | 4 | filter: relu(+0.49*x_1 +0.41*x_2 -2.04*x_3) | 0.978 | 0.150 | 94.3% | 41.1% |
| 3 | 5 | relu(+2.60*max_prev -3.46*2nd_prev) | 0.514 | 0.697 | 50.4% | 65.1% |
| 3 | 6 | relu(+2.39*max_prev -2.37*2nd_prev) | 0.520 | 0.693 | 45.1% | 71.2% |
| 3 | 7 | relu(+1.69*max -1.54*x_prev) | 0.388 | 0.782 | 77.3% | 29.0% |
| 4 | 0 | relu(+0.40*max -0.87*min) | 0.501 | 0.707 | 72.1% | 39.2% |
| 4 | 1 | comparator: relu(x_t - min) | 0.937 | 0.252 | 87.6% | 49.3% |
| 4 | 2 | relu(+1.32*max -1.31*min) | 0.131 | 0.932 | 40.0% | 61.0% |
| 4 | 3 | raw input: relu(-1.46*x_t +1.57*x_prev) | 0.970 | 0.174 | 96.3% | 25.6% |
| 4 | 4 | filter: relu(+0.41*x_1 +0.36*x_2 +0.49*x_3 -2.05*x_4) | 0.968 | 0.178 | 92.2% | 47.3% |
| 4 | 5 | relu(+2.02*max -2.53*min) | 0.483 | 0.719 | 45.6% | 61.5% |
| 4 | 6 | relu(-2.70*min +2.70*max_prev) | 0.670 | 0.574 | 42.1% | 67.3% |
| 4 | 7 | relu(+1.43*max -1.36*x_prev) | 0.432 | 0.754 | 72.4% | 35.5% |
| 5 | 0 | relu(+1.42*max -1.86*min) | 0.234 | 0.875 | 50.2% | 50.0% |
| 5 | 1 | comparator: relu(x_t - min) | 0.908 | 0.303 | 85.2% | 48.9% |
| 5 | 2 | relu(+1.54*max -1.47*min) | 0.183 | 0.904 | 48.3% | 53.1% |
| 5 | 3 | raw input: relu(-1.44*x_t +1.64*x_prev) | 0.971 | 0.172 | 96.2% | 27.5% |
| 5 | 4 | filter: relu(+0.37*x_1 +0.34*x_2 +0.36*x_3 +0.49*x_4 -2.05*x_5) | 0.965 | 0.186 | 91.4% | 50.3% |
| 5 | 5 | relu(-2.38*min +1.91*max_prev) | 0.457 | 0.737 | 47.8% | 58.6% |
| 5 | 6 | relu(-3.22*min +3.52*max_prev) | 0.674 | 0.571 | 49.3% | 54.5% |
| 5 | 7 | relu(+2.12*2nd -1.29*x_prev) | 0.387 | 0.783 | 69.2% | 37.4% |
| 6 | 0 | relu(+1.23*max -1.61*min) | 0.178 | 0.907 | 64.2% | 36.6% |
| 6 | 1 | filter: relu(-0.33*x_1 -0.33*x_2 -0.39*x_3 -0.33*x_4 -0.54*x_5 +2.12*x_6) | 0.830 | 0.412 | 83.3% | 40.9% |
| 6 | 2 | relu(+1.25*max -1.30*min) | 0.173 | 0.909 | 53.3% | 47.4% |
| 6 | 3 | raw input: relu(-1.39*x_t +1.61*x_prev) | 0.975 | 0.157 | 94.4% | 32.8% |
| 6 | 4 | filter: relu(+0.36*x_1 +0.32*x_2 +0.34*x_3 +0.35*x_4 +0.48*x_5 -2.01*x_6) | 0.965 | 0.186 | 90.9% | 51.2% |
| 6 | 5 | relu(-1.90*min +1.65*max_prev) | 0.270 | 0.854 | 54.8% | 47.5% |
| 6 | 6 | relu(-2.59*min +3.09*max_prev) | 0.501 | 0.706 | 63.0% | 41.8% |
| 6 | 7 | relu(-1.32*x_prev +2.00*2nd_prev) | 0.474 | 0.725 | 68.7% | 39.2% |
| 7 | 0 | relu(+0.94*max -1.53*min) | 0.169 | 0.912 | 75.7% | 25.0% |
| 7 | 1 | filter: relu(-0.35*x_1 -0.27*x_2 -0.35*x_3 -0.35*x_4 -0.33*x_5 -0.60*x_6 +2.10*x_7) | 0.825 | 0.419 | 84.5% | 35.4% |
| 7 | 2 | relu(+1.14*max -1.03*min) | 0.089 | 0.954 | 66.0% | 34.9% |
| 7 | 3 | raw input: relu(-1.38*x_t +1.51*x_prev) | 0.967 | 0.181 | 93.4% | 33.5% |
| 7 | 4 | filter: relu(+0.34*x_1 +0.31*x_2 +0.34*x_3 +0.34*x_4 +0.37*x_5 +0.42*x_6 -1.99*x_7) | 0.958 | 0.205 | 87.9% | 51.0% |
| 7 | 5 | relu(+1.34*max -1.71*min) | 0.191 | 0.900 | 60.2% | 40.8% |
| 7 | 6 | relu(-2.17*min +2.72*max_prev) | 0.336 | 0.815 | 76.2% | 28.9% |
| 7 | 7 | relu(+1.74*2nd -1.26*x_prev) | 0.468 | 0.729 | 64.0% | 39.8% |

*Sufficiency: decisions kept when only this neuron-step is replaced by its unit formula. Importance: decisions lost when it is set to its mean. Both are on held-out inputs. Units are approximate in dense networks (UNITS_PROGRAM, first-sweep conclusion), so a high fit does not imply exact sufficiency.*
