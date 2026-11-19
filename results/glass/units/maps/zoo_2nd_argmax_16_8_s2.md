# Unit map: zoo_2nd_argmax_16_8_s2

Coverage after the first step (variance-weighted share of neuron-steps fitted at R² ≥ 0.9 on held-out inputs): history units 0.23, raw-input units 0.03. Shuffled-weights control: history 0.00, raw 0.00.

Precision requirement ε95 = 0.006309573444801929: hidden-state noise (as a share of each neuron's spread) at which 5% of decisions flip. Substitution-grade units (relative error ≤ ε95): 16 of 128 neuron-steps.

## Per neuron (most common label over its explained steps)

| neuron | label |
|---|---|
| 0 | unexplained |
| 1 | relu(+0.29*max +0.19*x_prev) |
| 2 | raw input: relu(-0.26*x_prev) |
| 3 | unexplained |
| 4 | unexplained |
| 5 | unexplained |
| 6 | unexplained |
| 7 | raw input: relu(-8.26*x_t) |
| 8 | comparator: relu(2nd - x_t) |
| 9 | comparator: relu(max - x_t) |
| 10 | relu(-2.60*x_prev +5.68*max_prev) |
| 11 | raw input: relu(+9.29*x_prev) |
| 12 | unexplained |
| 13 | relu(+7.88*max_prev) |
| 14 | comparator: relu(max - x_t) |
| 15 | comparator: relu(x_t - 2nd_prev) |

## Every neuron-step

| step | neuron | unit | fit (held-out R²) | relative error | sufficiency | importance |
|---|---|---|---|---|---|---|
| 1 | 0 | raw input: relu(-0.04*x_t) | 1.000 | 0.000 | 100.0% | 0.1% |
| 1 | 1 | raw input: relu(+0.12*x_t) | 1.000 | 0.000 | 100.0% | 0.2% |
| 1 | 2 | raw input: relu(-0.02*x_t) | 1.000 | 0.000 | 100.0% | 0.0% |
| 1 | 3 | raw input: relu(+0.04*x_t) | 1.000 | 0.000 | 100.0% | 0.0% |
| 1 | 4 | raw input: relu(+0.31*x_t) | 1.000 | 0.000 | 100.0% | 0.2% |
| 1 | 5 | raw input: relu(-0.07*x_t) | 1.000 | 0.000 | 100.0% | 0.1% |
| 1 | 6 | raw input: relu(+0.16*x_t) | 1.000 | 0.000 | 100.0% | 0.1% |
| 1 | 7 | raw input: relu(-8.13*x_t) | 1.000 | 0.000 | 100.0% | 23.3% |
| 1 | 8 | raw input: relu(-6.67*x_t) | 1.000 | 0.000 | 100.0% | 44.2% |
| 1 | 9 | raw input: relu(-10.51*x_t) | 1.000 | 0.000 | 100.0% | 43.1% |
| 1 | 10 | raw input: relu(+0.00*x_t) | 1.000 | 0.000 | 100.0% | 0.0% |
| 1 | 11 | raw input: relu(-0.23*x_t) | 1.000 | 0.000 | 100.0% | 0.5% |
| 1 | 12 | raw input: relu(+0.06*x_t) | 1.000 | 0.000 | 100.0% | 0.1% |
| 1 | 13 | raw input: relu(-0.12*x_t) | 1.000 | 0.000 | 100.0% | 0.3% |
| 1 | 14 | raw input: relu(-7.77*x_t) | 1.000 | 0.000 | 100.0% | 18.4% |
| 1 | 15 | raw input: relu(+6.65*x_t) | 1.000 | 0.000 | 100.0% | 29.3% |
| 2 | 0 | comparator: relu(max - x_t) | 0.331 | 0.818 | 97.8% | 2.3% |
| 2 | 1 | relu(+0.29*max +0.19*x_prev) | 0.911 | 0.298 | 99.8% | 1.0% |
| 2 | 2 | raw input: relu(-0.26*x_prev) | 0.993 | 0.081 | 100.0% | 1.0% |
| 2 | 3 | relu(+0.09*max -0.13*x_prev) | 0.795 | 0.453 | 100.0% | 0.2% |
| 2 | 4 | relu(+2.74*max +1.95*x_prev) | 0.857 | 0.378 | 98.2% | 5.3% |
| 2 | 5 | comparator: relu(max - x_t) | 0.648 | 0.593 | 96.4% | 5.8% |
| 2 | 6 | raw input: relu(+0.07*x_t -0.01*x_prev) | 0.513 | 0.698 | 99.9% | 0.0% |
| 2 | 7 | raw input: relu(-8.26*x_t) | 0.999 | 0.038 | 99.0% | 30.2% |
| 2 | 8 | raw input: relu(-6.74*x_t) | 0.999 | 0.024 | 99.0% | 50.5% |
| 2 | 9 | comparator: relu(max - x_t) | 0.935 | 0.255 | 87.1% | 53.6% |
| 2 | 10 | comparator: relu(max - x_t) | 0.526 | 0.689 | 92.1% | 9.2% |
| 2 | 11 | raw input: relu(+9.29*x_prev) | 0.935 | 0.255 | 92.3% | 26.6% |
| 2 | 12 | raw input: relu(+0.02*x_t) | 0.453 | 0.740 | 100.0% | 0.0% |
| 2 | 13 | raw input: relu(+6.50*x_prev) | 0.892 | 0.328 | 86.3% | 32.5% |
| 2 | 14 | comparator: relu(max - x_t) | 0.927 | 0.271 | 86.3% | 37.1% |
| 2 | 15 | raw input: relu(+6.50*x_t) | 0.996 | 0.065 | 97.9% | 36.3% |
| 3 | 0 | relu(-2.97*x_prev +5.06*max_prev) | 0.736 | 0.514 | 94.5% | 10.1% |
| 3 | 1 | filter: relu(+1.99*x_1 +1.03*x_2 +0.39*x_3) | 0.608 | 0.626 | 96.5% | 6.7% |
| 3 | 2 | filter: relu(+0.34*x_1 -0.48*x_2 -0.03*x_3) | 0.790 | 0.458 | 99.4% | 3.3% |
| 3 | 3 | relu(-3.36*x_prev +6.38*max_prev) | 0.698 | 0.550 | 93.8% | 10.4% |
| 3 | 4 | relu(+2.88*max +2.30*x_prev) | 0.831 | 0.412 | 97.4% | 7.6% |
| 3 | 5 | relu(-4.06*x_prev +11.19*max_prev) | 0.705 | 0.543 | 79.4% | 40.2% |
| 3 | 6 | filter: relu(+0.68*x_1 -0.24*x_2 +0.12*x_3) | 0.863 | 0.370 | 99.8% | 0.9% |
| 3 | 7 | filter: relu(+5.52*x_1 +0.03*x_2 -10.27*x_3) | 0.879 | 0.348 | 77.4% | 46.8% |
| 3 | 8 | filter: relu(+1.02*x_1 +1.02*x_2 -7.22*x_3) | 0.942 | 0.241 | 90.2% | 57.0% |
| 3 | 9 | comparator: relu(max - x_t) | 0.978 | 0.149 | 92.8% | 56.8% |
| 3 | 10 | relu(-2.60*x_prev +5.68*max_prev) | 0.937 | 0.252 | 96.2% | 13.3% |
| 3 | 11 | relu(+4.03*max +5.15*x_prev) | 0.817 | 0.427 | 85.5% | 31.1% |
| 3 | 12 | relu(+1.16*max +0.54*x_prev) | 0.077 | 0.961 | 91.5% | 10.4% |
| 3 | 13 | relu(+7.88*max_prev) | 0.970 | 0.174 | 91.8% | 38.0% |
| 3 | 14 | raw input: relu(-9.63*x_t +5.36*x_prev) | 0.892 | 0.329 | 79.5% | 44.2% |
| 3 | 15 | raw input: relu(+5.95*x_t -0.65*x_prev) | 0.949 | 0.225 | 95.4% | 38.6% |
| 4 | 0 | relu(-2.73*x_prev +6.30*max_prev) | 0.577 | 0.650 | 84.4% | 20.6% |
| 4 | 1 | relu(+2.61*max +1.16*2nd_prev) | 0.321 | 0.824 | 87.0% | 17.8% |
| 4 | 2 | filter: relu(+0.18*x_1 +0.38*x_2 -0.45*x_3 -0.02*x_4) | 0.733 | 0.517 | 99.0% | 5.0% |
| 4 | 3 | relu(-2.29*x_prev +4.73*max_prev) | 0.495 | 0.710 | 93.7% | 8.8% |
| 4 | 4 | relu(+1.78*2nd +4.98*max_prev) | 0.761 | 0.489 | 96.1% | 9.0% |
| 4 | 5 | relu(+8.90*max_prev +2.06*2nd_prev) | 0.576 | 0.651 | 70.8% | 43.8% |
| 4 | 6 | filter: relu(+3.52*x_1 +0.70*x_2 -1.55*x_3 +0.37*x_4) | 0.697 | 0.551 | 97.3% | 9.2% |
| 4 | 7 | filter: relu(+1.81*x_1 +4.97*x_2 +0.66*x_3 -10.62*x_4) | 0.819 | 0.425 | 72.5% | 50.1% |
| 4 | 8 | comparator: relu(2nd - x_t) | 0.956 | 0.209 | 88.5% | 58.3% |
| 4 | 9 | comparator: relu(max - x_t) | 0.991 | 0.094 | 93.3% | 62.8% |
| 4 | 10 | relu(-1.24*x_prev +4.99*max_prev) | 0.738 | 0.512 | 90.3% | 18.1% |
| 4 | 11 | relu(+1.93*x_prev +10.64*max_prev) | 0.640 | 0.600 | 71.4% | 47.7% |
| 4 | 12 | relu(-1.96*min +1.78*x_prev) | 0.215 | 0.886 | 88.4% | 14.1% |
| 4 | 13 | relu(+7.00*max_prev -2.13*2nd_prev) | 0.815 | 0.430 | 75.9% | 43.3% |
| 4 | 14 | comparator: relu(max - x_t) | 0.837 | 0.403 | 69.2% | 49.7% |
| 4 | 15 | filter: relu(-0.69*x_1 -0.66*x_2 -0.73*x_3 +5.36*x_4) | 0.928 | 0.268 | 92.5% | 39.1% |
| 5 | 0 | relu(-2.94*x_prev +7.93*max_prev) | 0.572 | 0.654 | 72.9% | 33.2% |
| 5 | 1 | relu(+3.90*2nd -2.22*min) | 0.188 | 0.901 | 76.9% | 26.3% |
| 5 | 2 | filter: relu(+0.57*x_1 +0.19*x_2 +0.31*x_3 -0.65*x_4 -0.01*x_5) | 0.592 | 0.639 | 97.0% | 9.5% |
| 5 | 3 | relu(-2.55*x_prev +7.23*max_prev) | 0.567 | 0.658 | 91.8% | 14.9% |
| 5 | 4 | relu(+1.24*x_prev +4.66*max_prev) | 0.586 | 0.643 | 94.6% | 12.7% |
| 5 | 5 | relu(+10.00*max_prev +5.15*2nd_prev) | 0.702 | 0.546 | 67.3% | 50.9% |
| 5 | 6 | relu(-1.90*x_prev +4.20*max_prev) | 0.476 | 0.724 | 91.6% | 13.6% |
| 5 | 7 | comparator: relu(max - x_t) | 0.842 | 0.397 | 63.3% | 57.2% |
| 5 | 8 | comparator: relu(2nd - x_t) | 0.976 | 0.153 | 92.2% | 61.1% |
| 5 | 9 | comparator: relu(max - x_t) | 0.993 | 0.083 | 89.5% | 67.1% |
| 5 | 10 | relu(-1.41*x_prev +5.29*max_prev) | 0.692 | 0.555 | 82.5% | 24.3% |
| 5 | 11 | relu(+9.72*max_prev +4.58*2nd_prev) | 0.585 | 0.644 | 66.1% | 46.6% |
| 5 | 12 | relu(-1.93*min +4.86*2nd_prev) | 0.444 | 0.745 | 83.7% | 23.4% |
| 5 | 13 | relu(+6.70*max_prev -2.20*2nd_prev) | 0.763 | 0.487 | 75.4% | 41.2% |
| 5 | 14 | comparator: relu(max - x_t) | 0.824 | 0.420 | 65.4% | 53.4% |
| 5 | 15 | comparator: relu(x_t - 2nd_prev) | 0.946 | 0.233 | 94.9% | 38.3% |
| 6 | 0 | relu(-3.02*x_prev +9.01*max_prev) | 0.533 | 0.683 | 66.0% | 42.5% |
| 6 | 1 | relu(+4.50*2nd -1.89*min) | 0.253 | 0.864 | 74.0% | 29.8% |
| 6 | 2 | relu(-0.53*x_prev +0.63*max_prev) | 0.487 | 0.716 | 94.5% | 12.8% |
| 6 | 3 | relu(-2.11*x_prev +6.12*max_prev) | 0.399 | 0.775 | 84.8% | 20.0% |
| 6 | 4 | relu(+3.70*max_prev +3.29*2nd_prev) | 0.701 | 0.547 | 94.3% | 16.1% |
| 6 | 5 | relu(+8.97*max_prev +6.89*2nd_prev) | 0.702 | 0.546 | 70.0% | 48.3% |
| 6 | 6 | relu(-2.39*x_prev +6.41*max_prev) | 0.442 | 0.747 | 87.2% | 19.6% |
| 6 | 7 | comparator: relu(max - x_t) | 0.867 | 0.365 | 59.2% | 61.3% |
| 6 | 8 | comparator: relu(2nd - x_t) | 0.987 | 0.113 | 95.3% | 60.5% |
| 6 | 9 | comparator: relu(max - x_t) | 0.992 | 0.090 | 82.3% | 69.6% |
| 6 | 10 | relu(-1.34*min +4.06*max_prev) | 0.515 | 0.697 | 77.2% | 31.1% |
| 6 | 11 | relu(+10.65*max_prev +6.41*2nd_prev) | 0.641 | 0.599 | 62.1% | 50.4% |
| 6 | 12 | relu(-2.17*min +4.09*2nd_prev) | 0.267 | 0.856 | 75.5% | 28.2% |
| 6 | 13 | relu(+5.95*max_prev -2.89*2nd_prev) | 0.506 | 0.703 | 68.3% | 44.5% |
| 6 | 14 | comparator: relu(max - x_t) | 0.852 | 0.385 | 57.1% | 59.7% |
| 6 | 15 | comparator: relu(x_t - 2nd_prev) | 0.963 | 0.193 | 96.7% | 33.9% |
| 7 | 0 | relu(-3.02*x_prev +10.33*max_prev) | 0.500 | 0.707 | 61.6% | 46.5% |
| 7 | 1 | relu(-2.14*min +4.90*2nd_prev) | 0.239 | 0.872 | 71.2% | 32.9% |
| 7 | 2 | relu(-0.68*x_prev +0.85*max_prev) | 0.381 | 0.787 | 90.4% | 16.3% |
| 7 | 3 | relu(-1.99*x_prev +7.42*max_prev) | 0.485 | 0.718 | 79.7% | 28.6% |
| 7 | 4 | relu(+1.02*x_prev +4.47*max_prev) | 0.347 | 0.808 | 83.8% | 20.5% |
| 7 | 5 | relu(+9.44*max_prev +7.67*2nd_prev) | 0.729 | 0.521 | 70.1% | 47.2% |
| 7 | 6 | relu(-2.10*x_prev +7.43*max_prev) | 0.410 | 0.768 | 84.8% | 20.9% |
| 7 | 7 | comparator: relu(max - x_t) | 0.863 | 0.370 | 50.6% | 66.4% |
| 7 | 8 | comparator: relu(2nd - x_t) | 0.993 | 0.083 | 95.7% | 59.4% |
| 7 | 9 | comparator: relu(max - x_t) | 0.988 | 0.107 | 76.6% | 73.2% |
| 7 | 10 | relu(-1.37*x_prev +5.35*max_prev) | 0.588 | 0.642 | 75.6% | 33.2% |
| 7 | 11 | relu(+10.35*max_prev +7.44*2nd_prev) | 0.666 | 0.578 | 64.2% | 49.7% |
| 7 | 12 | relu(-2.25*min +5.43*2nd_prev) | 0.301 | 0.836 | 73.1% | 34.4% |
| 7 | 13 | relu(+5.38*max_prev -2.63*2nd_prev) | 0.430 | 0.755 | 69.4% | 43.5% |
| 7 | 14 | comparator: relu(max - x_t) | 0.859 | 0.376 | 53.9% | 62.5% |
| 7 | 15 | comparator: relu(x_t - 2nd_prev) | 0.973 | 0.165 | 98.3% | 25.7% |
| 8 | 0 | relu(-3.04*x_prev +11.65*max_prev) | 0.439 | 0.749 | 57.2% | 50.4% |
| 8 | 1 | relu(+2.97*max +3.36*2nd_prev) | 0.223 | 0.881 | 67.1% | 35.7% |
| 8 | 2 | relu(-0.77*x_prev +1.08*max_prev) | 0.319 | 0.825 | 82.2% | 22.3% |
| 8 | 3 | relu(+9.93*max_prev -5.76*2nd_prev) | 0.388 | 0.782 | 67.0% | 41.2% |
| 8 | 4 | relu(+2.94*max +2.38*2nd) | 0.258 | 0.861 | 75.4% | 25.1% |
| 8 | 5 | relu(+9.34*max_prev +7.99*2nd_prev) | 0.747 | 0.503 | 74.1% | 45.1% |
| 8 | 6 | relu(-2.25*x_prev +8.98*max_prev) | 0.402 | 0.773 | 67.3% | 39.4% |
| 8 | 7 | comparator: relu(max - x_t) | 0.877 | 0.350 | 49.5% | 71.7% |
| 8 | 8 | comparator: relu(2nd - x_t) | 0.995 | 0.068 | 97.1% | 57.2% |
| 8 | 9 | comparator: relu(max - x_t) | 0.987 | 0.113 | 71.4% | 78.6% |
| 8 | 10 | relu(-1.50*x_prev +5.43*max_prev) | 0.504 | 0.705 | 71.8% | 37.0% |
| 8 | 11 | relu(+6.31*2nd +11.85*max_prev) | 0.617 | 0.619 | 62.4% | 49.1% |
| 8 | 12 | relu(-1.53*min +4.05*max_prev) | 0.219 | 0.884 | 64.9% | 38.5% |
| 8 | 13 | relu(+4.73*max_prev -2.66*2nd_prev) | 0.326 | 0.821 | 70.2% | 41.2% |
| 8 | 14 | comparator: relu(max - x_t) | 0.863 | 0.370 | 53.3% | 65.7% |
| 8 | 15 | comparator: relu(x_t - 2nd_prev) | 0.978 | 0.149 | 97.6% | 8.8% |

*Sufficiency: decisions kept when only this neuron-step is replaced by its unit formula. Importance: decisions lost when it is set to its mean. Both are on held-out inputs. Units are approximate in dense networks (UNITS_PROGRAM, first-sweep conclusion), so a high fit does not imply exact sufficiency.*
