# Unit map: zoo_2nd_argmax_8_5

Coverage after the first step (variance-weighted share of neuron-steps fitted at R² ≥ 0.9 on held-out inputs): history units 0.23, raw-input units 0.10. Shuffled-weights control: history 0.08, raw 0.08.

Precision requirement ε95 = 0.012589254117941675: hidden-state noise (as a share of each neuron's spread) at which 5% of decisions flip. Substitution-grade units (relative error ≤ ε95): 8 of 40 neuron-steps.

## Per neuron (most common label over its explained steps)

| neuron | label |
|---|---|
| 0 | raw input: relu(+0.62*x_t -4.82*x_prev) |
| 1 | raw input: relu(-6.24*x_t) |
| 2 | relu(+4.23*max +4.76*x_prev) |
| 3 | comparator: relu(max - x_t) |
| 4 | filter: relu(+10.40*x_1 -2.14*x_2 -3.08*x_3 +0.08*x_4) |
| 5 | comparator: relu(max - x_t) |
| 6 | unexplained |
| 7 | comparator: relu(x_t - 2nd) |

## Every neuron-step

| step | neuron | unit | fit (held-out R²) | relative error | sufficiency | importance |
|---|---|---|---|---|---|---|
| 1 | 0 | raw input: relu(+0.64*x_t) | 1.000 | 0.000 | 100.0% | 1.6% |
| 1 | 1 | raw input: relu(-6.42*x_t) | 1.000 | 0.000 | 100.0% | 19.2% |
| 1 | 2 | raw input: relu(+0.42*x_t) | 1.000 | 0.000 | 100.0% | 2.1% |
| 1 | 3 | raw input: relu(-6.57*x_t) | 1.000 | 0.000 | 100.0% | 22.4% |
| 1 | 4 | raw input: relu(+0.05*x_t) | 1.000 | 0.000 | 100.0% | 0.3% |
| 1 | 5 | raw input: relu(-4.91*x_t) | 1.000 | 0.000 | 100.0% | 17.4% |
| 1 | 6 | raw input: relu(-0.05*x_t) | 1.000 | 0.000 | 100.0% | 0.2% |
| 1 | 7 | raw input: relu(+7.00*x_t) | 1.000 | 0.000 | 100.0% | 39.8% |
| 2 | 0 | raw input: relu(+0.62*x_t -4.82*x_prev) | 1.000 | 0.014 | 99.9% | 14.9% |
| 2 | 1 | raw input: relu(-6.24*x_t) | 0.997 | 0.056 | 99.1% | 20.4% |
| 2 | 2 | relu(+4.23*max +4.76*x_prev) | 0.914 | 0.294 | 93.8% | 33.7% |
| 2 | 3 | comparator: relu(max - x_t) | 0.932 | 0.261 | 92.0% | 35.3% |
| 2 | 4 | comparator: relu(x_t - 2nd) | 0.509 | 0.701 | 90.2% | 14.0% |
| 2 | 5 | comparator: relu(max - x_t) | 0.937 | 0.251 | 93.5% | 28.9% |
| 2 | 6 | relu(+3.34*max +2.31*x_prev) | 0.754 | 0.496 | 86.1% | 31.9% |
| 2 | 7 | raw input: relu(+5.57*x_t -1.75*x_prev) | 0.930 | 0.264 | 92.2% | 40.3% |
| 3 | 0 | relu(-7.34*x_prev +7.82*max_prev) | 0.909 | 0.301 | 89.1% | 33.6% |
| 3 | 1 | filter: relu(+4.88*x_1 -0.07*x_2 -7.90*x_3) | 0.890 | 0.331 | 87.2% | 36.2% |
| 3 | 2 | relu(+7.39*x_prev -6.32*2nd_prev) | 0.913 | 0.294 | 92.2% | 33.1% |
| 3 | 3 | raw input: relu(-8.27*x_t +5.09*x_prev) | 0.878 | 0.349 | 86.6% | 43.1% |
| 3 | 4 | relu(+0.40*max -2.74*2nd_prev) | 0.788 | 0.460 | 93.5% | 15.7% |
| 3 | 5 | comparator: relu(max - x_t) | 0.957 | 0.209 | 93.6% | 32.0% |
| 3 | 6 | relu(+0.54*x_prev +5.97*max_prev) | 0.892 | 0.328 | 88.5% | 32.0% |
| 3 | 7 | comparator: relu(x_t - max_prev) | 0.949 | 0.227 | 95.2% | 33.1% |
| 4 | 0 | relu(-5.48*x_prev +5.85*max_prev) | 0.819 | 0.425 | 86.9% | 35.4% |
| 4 | 1 | comparator: relu(max - x_t) | 0.832 | 0.409 | 81.0% | 49.3% |
| 4 | 2 | filter: relu(+3.11*x_1 -6.28*x_2 +4.89*x_3 +0.76*x_4) | 0.600 | 0.632 | 78.5% | 36.5% |
| 4 | 3 | comparator: relu(max - x_t) | 0.795 | 0.452 | 72.0% | 53.2% |
| 4 | 4 | filter: relu(+10.40*x_1 -2.14*x_2 -3.08*x_3 +0.08*x_4) | 0.907 | 0.305 | 88.0% | 39.3% |
| 4 | 5 | comparator: relu(max - x_t) | 0.760 | 0.490 | 80.2% | 35.8% |
| 4 | 6 | relu(+9.10*max_prev) | 0.799 | 0.449 | 81.5% | 38.0% |
| 4 | 7 | comparator: relu(x_t - 2nd) | 0.962 | 0.196 | 95.7% | 25.4% |
| 5 | 0 | relu(-4.50*x_prev +4.66*max_prev) | 0.721 | 0.528 | 84.2% | 34.4% |
| 5 | 1 | comparator: relu(max - x_t) | 0.824 | 0.419 | 68.5% | 61.1% |
| 5 | 2 | filter: relu(+0.04*x_1 +2.17*x_2 -5.23*x_3 +4.13*x_4 +0.61*x_5) | 0.532 | 0.684 | 74.8% | 36.4% |
| 5 | 3 | comparator: relu(max - x_t) | 0.808 | 0.438 | 59.7% | 59.5% |
| 5 | 4 | filter: relu(-2.10*x_1 +8.86*x_2 -2.05*x_3 -3.32*x_4 +0.14*x_5) | 0.875 | 0.353 | 88.5% | 36.9% |
| 5 | 5 | comparator: relu(max - x_t) | 0.800 | 0.447 | 78.5% | 47.1% |
| 5 | 6 | relu(+8.12*max_prev -2.99*2nd_prev) | 0.512 | 0.699 | 76.5% | 31.1% |
| 5 | 7 | comparator: relu(x_t - 2nd) | 0.957 | 0.207 | 97.1% | 10.7% |

*Sufficiency: decisions kept when only this neuron-step is replaced by its unit formula. Importance: decisions lost when it is set to its mean. Both are on held-out inputs. Units are approximate in dense networks (UNITS_PROGRAM, first-sweep conclusion), so a high fit does not imply exact sufficiency.*
