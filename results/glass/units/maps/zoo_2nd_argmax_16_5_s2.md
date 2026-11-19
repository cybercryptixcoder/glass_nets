# Unit map: zoo_2nd_argmax_16_5_s2

Coverage after the first step (variance-weighted share of neuron-steps fitted at R² ≥ 0.9 on held-out inputs): history units 0.29, raw-input units 0.16. Shuffled-weights control: history 0.06, raw 0.09.

Precision requirement ε95 = 0.01: hidden-state noise (as a share of each neuron's spread) at which 5% of decisions flip. Substitution-grade units (relative error ≤ ε95): 19 of 80 neuron-steps.

## Per neuron (most common label over its explained steps)

| neuron | label |
|---|---|
| 0 | raw input: relu(+5.41*x_t -5.16*x_prev) |
| 1 | comparator: relu(max - x_t) |
| 2 | unexplained |
| 3 | comparator: relu(x_t - 2nd) |
| 4 | raw input: relu(-0.16*x_prev) |
| 5 | relu(-3.01*x_prev +13.02*max_prev) |
| 6 | comparator: relu(max - x_t) |
| 7 | relu(-17.62*max_prev) |
| 8 | filter: relu(-14.03*x_1 +6.60*x_2 +0.26*x_3) |
| 9 | unexplained |
| 10 | raw input: relu(-8.32*x_prev) |
| 11 | filter: relu(+8.04*x_1 +0.13*x_2 +0.03*x_3) |
| 12 | comparator: relu(max - x_t) |
| 13 | filter: relu(+2.33*x_1 -1.14*x_2 -0.25*x_3) |
| 14 | raw input: relu(-4.37*x_prev) |
| 15 | raw input: relu(+14.39*x_prev) |

## Every neuron-step

| step | neuron | unit | fit (held-out R²) | relative error | sufficiency | importance |
|---|---|---|---|---|---|---|
| 1 | 0 | raw input: relu(+5.49*x_t) | 1.000 | 0.000 | 100.0% | 15.4% |
| 1 | 1 | raw input: relu(-8.32*x_t) | 1.000 | 0.000 | 100.0% | 19.0% |
| 1 | 2 | raw input: relu(-7.21*x_t) | 1.000 | 0.000 | 100.0% | 18.5% |
| 1 | 3 | raw input: relu(+14.70*x_t) | 1.000 | 0.000 | 100.0% | 49.4% |
| 1 | 4 | raw input: relu(-0.00*x_t) | 1.000 | 0.000 | 100.0% | 0.0% |
| 1 | 5 | raw input: relu(+0.19*x_t) | 1.000 | 0.000 | 100.0% | 0.3% |
| 1 | 6 | raw input: relu(-6.63*x_t) | 1.000 | 0.000 | 100.0% | 12.5% |
| 1 | 7 | raw input: relu(-0.22*x_t) | 1.000 | 0.000 | 100.0% | 0.8% |
| 1 | 8 | raw input: relu(+0.17*x_t) | 1.000 | 0.000 | 100.0% | 0.3% |
| 1 | 9 | raw input: relu(-7.29*x_t) | 1.000 | 0.000 | 100.0% | 17.9% |
| 1 | 10 | raw input: relu(+0.07*x_t) | 1.000 | 0.000 | 100.0% | 0.4% |
| 1 | 11 | raw input: relu(+0.02*x_t) | 1.000 | 0.000 | 100.0% | 0.0% |
| 1 | 12 | raw input: relu(-6.04*x_t) | 1.000 | 0.000 | 100.0% | 11.7% |
| 1 | 13 | raw input: relu(-0.21*x_t) | 1.000 | 0.000 | 100.0% | 0.6% |
| 1 | 14 | raw input: relu(-0.03*x_t) | 1.000 | 0.000 | 100.0% | 0.2% |
| 1 | 15 | raw input: relu(-0.05*x_t) | 1.000 | 0.000 | 100.0% | 0.1% |
| 2 | 0 | raw input: relu(+5.41*x_t -5.16*x_prev) | 1.000 | 0.019 | 99.9% | 17.1% |
| 2 | 1 | comparator: relu(max - x_t) | 0.924 | 0.275 | 96.4% | 22.5% |
| 2 | 2 | relu(-3.62*x_t -5.21*max) | 0.858 | 0.377 | 97.0% | 12.6% |
| 2 | 3 | raw input: relu(+11.10*x_t -3.60*x_prev) | 0.904 | 0.311 | 88.5% | 59.9% |
| 2 | 4 | raw input: relu(-0.16*x_prev) | 1.000 | 0.001 | 100.0% | 0.3% |
| 2 | 5 | relu(+3.17*max +2.77*x_prev) | 0.866 | 0.366 | 97.7% | 10.1% |
| 2 | 6 | comparator: relu(max - x_t) | 0.928 | 0.268 | 95.8% | 21.2% |
| 2 | 7 | raw input: relu(-0.00*x_t) | 0.023 | 0.989 | 100.0% | 0.0% |
| 2 | 8 | relu(+4.15*max +3.31*x_prev) | 0.840 | 0.400 | 95.0% | 16.6% |
| 2 | 9 | relu(-3.69*x_t -5.44*max) | 0.865 | 0.368 | 97.2% | 10.0% |
| 2 | 10 | raw input: relu(-8.32*x_prev) | 1.000 | 0.010 | 99.9% | 19.9% |
| 2 | 11 | raw input: relu(+0.01*x_t) | 0.672 | 0.573 | 100.0% | 0.0% |
| 2 | 12 | comparator: relu(max - x_t) | 0.926 | 0.272 | 95.5% | 15.2% |
| 2 | 13 | relu(-0.05*x_t -0.02*max) | 0.349 | 0.807 | 99.9% | 0.1% |
| 2 | 14 | raw input: relu(-4.37*x_prev) | 1.000 | 0.007 | 100.0% | 9.7% |
| 2 | 15 | raw input: relu(+14.39*x_prev) | 0.980 | 0.141 | 97.4% | 26.6% |
| 3 | 0 | raw input: relu(+5.47*x_t -5.43*x_prev) | 0.999 | 0.024 | 99.7% | 21.0% |
| 3 | 1 | raw input: relu(-9.50*x_t +6.67*x_prev) | 0.913 | 0.295 | 92.9% | 31.1% |
| 3 | 2 | filter: relu(+0.71*x_1 -0.51*x_2 -4.11*x_3) | 0.439 | 0.749 | 91.3% | 13.3% |
| 3 | 3 | comparator: relu(x_t - max_prev) | 0.918 | 0.287 | 91.4% | 58.4% |
| 3 | 4 | relu(-4.34*max_prev) | 0.995 | 0.071 | 99.8% | 3.2% |
| 3 | 5 | relu(-3.01*x_prev +13.02*max_prev) | 0.918 | 0.286 | 94.8% | 20.5% |
| 3 | 6 | comparator: relu(max - x_t) | 0.984 | 0.126 | 96.9% | 26.6% |
| 3 | 7 | filter: relu(-0.05*x_1 -0.01*x_2 -0.03*x_3) | 0.160 | 0.917 | 99.7% | 0.4% |
| 3 | 8 | filter: relu(-14.03*x_1 +6.60*x_2 +0.26*x_3) | 0.903 | 0.311 | 96.2% | 14.4% |
| 3 | 9 | filter: relu(+0.84*x_1 -0.43*x_2 -4.10*x_3) | 0.439 | 0.749 | 94.0% | 9.2% |
| 3 | 10 | relu(-11.24*x_prev +7.37*max_prev) | 0.879 | 0.347 | 92.3% | 32.0% |
| 3 | 11 | filter: relu(+8.04*x_1 +0.13*x_2 +0.03*x_3) | 1.000 | 0.007 | 100.0% | 13.3% |
| 3 | 12 | comparator: relu(max - x_t) | 0.999 | 0.026 | 99.8% | 18.6% |
| 3 | 13 | filter: relu(+2.33*x_1 -1.14*x_2 -0.25*x_3) | 0.961 | 0.197 | 99.4% | 4.5% |
| 3 | 14 | filter: relu(+1.82*x_1 -2.81*x_2 -0.02*x_3) | 0.949 | 0.225 | 99.6% | 5.5% |
| 3 | 15 | relu(+2.18*x_prev +12.04*max_prev) | 0.964 | 0.190 | 95.9% | 27.9% |
| 4 | 0 | raw input: relu(+5.49*x_t -5.43*x_prev) | 0.999 | 0.035 | 99.6% | 27.0% |
| 4 | 1 | filter: relu(+4.86*x_1 -1.38*x_2 +5.49*x_3 -10.30*x_4) | 0.697 | 0.550 | 83.8% | 34.8% |
| 4 | 2 | filter: relu(+5.83*x_1 +0.71*x_2 -0.92*x_3 -6.26*x_4) | 0.690 | 0.557 | 88.9% | 32.5% |
| 4 | 3 | comparator: relu(x_t - 2nd) | 0.955 | 0.213 | 87.8% | 49.9% |
| 4 | 4 | relu(-10.52*max_prev -1.66*2nd_prev) | 0.970 | 0.174 | 99.4% | 3.8% |
| 4 | 5 | filter: relu(-17.63*x_1 +7.51*x_2 +4.52*x_3 +0.99*x_4) | 0.746 | 0.504 | 90.7% | 16.9% |
| 4 | 6 | comparator: relu(max - x_t) | 0.997 | 0.059 | 98.9% | 20.5% |
| 4 | 7 | relu(+0.86*x_prev -1.60*max_prev) | 0.203 | 0.893 | 99.5% | 0.7% |
| 4 | 8 | filter: relu(+7.59*x_1 -15.87*x_2 +5.26*x_3 +0.41*x_4) | 0.725 | 0.524 | 87.1% | 22.0% |
| 4 | 9 | comparator: relu(2nd - x_t) | 0.542 | 0.677 | 92.1% | 13.6% |
| 4 | 10 | relu(-9.39*x_prev +8.93*max_prev) | 0.982 | 0.134 | 97.9% | 23.2% |
| 4 | 11 | relu(-1.97*x_prev +6.91*max_prev) | 0.749 | 0.501 | 94.0% | 16.3% |
| 4 | 12 | comparator: relu(max - x_t) | 0.802 | 0.445 | 91.4% | 21.9% |
| 4 | 13 | relu(-1.82*x_prev +2.08*max_prev) | 0.588 | 0.642 | 97.0% | 7.8% |
| 4 | 14 | filter: relu(+0.69*x_1 +2.33*x_2 -3.34*x_3 -0.04*x_4) | 0.883 | 0.342 | 99.1% | 5.5% |
| 4 | 15 | relu(+1.75*x_prev +11.26*max_prev) | 0.946 | 0.233 | 95.5% | 23.4% |
| 5 | 0 | raw input: relu(+5.58*x_t -5.55*x_prev) | 0.973 | 0.166 | 98.6% | 17.2% |
| 5 | 1 | comparator: relu(max - x_t) | 0.752 | 0.498 | 72.9% | 41.3% |
| 5 | 2 | filter: relu(+0.95*x_1 +5.24*x_2 +0.73*x_3 -0.66*x_4 -6.78*x_5) | 0.653 | 0.589 | 73.9% | 42.4% |
| 5 | 3 | comparator: relu(x_t - 2nd) | 0.978 | 0.149 | 90.8% | 25.1% |
| 5 | 4 | filter: relu(+3.07*x_1 -0.67*x_2 -1.56*x_3 -1.60*x_4 +0.07*x_5) | 0.698 | 0.550 | 98.5% | 1.5% |
| 5 | 5 | filter: relu(+9.03*x_1 -19.16*x_2 +6.81*x_3 +4.82*x_4 +1.38*x_5) | 0.600 | 0.633 | 80.0% | 22.3% |
| 5 | 6 | comparator: relu(max - x_t) | 0.996 | 0.064 | 98.5% | 30.6% |
| 5 | 7 | relu(-17.62*max_prev) | 0.911 | 0.299 | 100.0% | 0.5% |
| 5 | 8 | filter: relu(+5.79*x_1 +6.06*x_2 -16.04*x_3 +4.49*x_4 +0.09*x_5) | 0.551 | 0.670 | 80.3% | 24.7% |
| 5 | 9 | filter: relu(+5.87*x_1 +0.76*x_2 +0.77*x_3 -0.70*x_4 -6.84*x_5) | 0.595 | 0.637 | 72.4% | 42.8% |
| 5 | 10 | relu(-8.93*x_prev +9.18*max_prev) | 0.993 | 0.082 | 99.5% | 16.8% |
| 5 | 11 | filter: relu(-10.56*x_1 +5.14*x_2 +5.27*x_3 +0.29*x_4 +0.11*x_5) | 0.764 | 0.486 | 94.6% | 13.9% |
| 5 | 12 | filter: relu(-0.51*x_1 +0.01*x_2 +4.15*x_3 +3.92*x_4 -7.37*x_5) | 0.783 | 0.465 | 88.0% | 32.3% |
| 5 | 13 | relu(-2.05*x_prev +2.82*max_prev) | 0.475 | 0.724 | 99.0% | 2.9% |
| 5 | 14 | filter: relu(+0.48*x_1 +0.38*x_2 +2.64*x_3 -3.81*x_4 -0.11*x_5) | 0.849 | 0.389 | 99.0% | 8.5% |
| 5 | 15 | filter: relu(-11.99*x_1 +2.47*x_2 +4.40*x_3 +6.15*x_4 -0.07*x_5) | 0.418 | 0.763 | 86.4% | 17.3% |

*Sufficiency: decisions kept when only this neuron-step is replaced by its unit formula. Importance: decisions lost when it is set to its mean. Both are on held-out inputs. Units are approximate in dense networks (UNITS_PROGRAM, first-sweep conclusion), so a high fit does not imply exact sufficiency.*
