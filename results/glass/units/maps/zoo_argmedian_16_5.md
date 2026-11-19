# Unit map: zoo_argmedian_16_5

Coverage after the first step (variance-weighted share of neuron-steps fitted at R² ≥ 0.9 on held-out inputs): history units 0.16, raw-input units 0.28. Shuffled-weights control: history 0.02, raw 0.01.

Precision requirement ε95 = 0.01: hidden-state noise (as a share of each neuron's spread) at which 5% of decisions flip. Substitution-grade units (relative error ≤ ε95): 22 of 80 neuron-steps.

## Per neuron (most common label over its explained steps)

| neuron | label |
|---|---|
| 0 | unexplained |
| 1 | raw input: relu(+6.78*x_t) |
| 2 | raw input: relu(-7.97*x_prev) |
| 3 | relu(-2.48*max_prev -15.76*2nd_prev) |
| 4 | raw input: relu(-11.28*x_t) |
| 5 | raw input: relu(+4.93*x_t) |
| 6 | raw input: relu(+0.20*x_t +0.04*x_prev) |
| 7 | raw input: relu(+10.90*x_t -10.84*x_prev) |
| 8 | raw input: relu(+0.13*x_t +0.03*x_prev) |
| 9 | unexplained |
| 10 | relu(-29.68*x_prev +23.00*max_prev) |
| 11 | raw input: relu(-10.31*x_t +10.37*x_prev) |
| 12 | raw input: relu(-10.80*x_t) |
| 13 | raw input: relu(-11.46*x_t) |
| 14 | unexplained |
| 15 | relu(+14.68*x_prev -16.06*2nd_prev) |

## Every neuron-step

| step | neuron | unit | fit (held-out R²) | relative error | sufficiency | importance |
|---|---|---|---|---|---|---|
| 1 | 0 | raw input: relu(-0.02*x_t) | 1.000 | 0.000 | 100.0% | 0.0% |
| 1 | 1 | raw input: relu(+6.76*x_t) | 1.000 | 0.000 | 100.0% | 3.8% |
| 1 | 2 | raw input: relu(+0.11*x_t) | 1.000 | 0.000 | 100.0% | 0.3% |
| 1 | 3 | raw input: relu(-0.06*x_t) | 1.000 | 0.000 | 100.0% | 0.1% |
| 1 | 4 | raw input: relu(-11.31*x_t) | 1.000 | 0.000 | 100.0% | 11.7% |
| 1 | 5 | raw input: relu(+4.97*x_t) | 1.000 | 0.000 | 100.0% | 4.0% |
| 1 | 6 | raw input: relu(+0.18*x_t) | 1.000 | 0.000 | 100.0% | 0.1% |
| 1 | 7 | raw input: relu(+10.83*x_t) | 1.000 | 0.000 | 100.0% | 24.2% |
| 1 | 8 | raw input: relu(+0.15*x_t) | 1.000 | 0.000 | 100.0% | 0.1% |
| 1 | 9 | raw input: relu(+0.09*x_t) | 1.000 | 0.000 | 100.0% | 0.1% |
| 1 | 10 | raw input: relu(-0.11*x_t) | 1.000 | 0.000 | 100.0% | 0.3% |
| 1 | 11 | raw input: relu(-10.30*x_t) | 1.000 | 0.000 | 100.0% | 21.3% |
| 1 | 12 | raw input: relu(-10.83*x_t) | 1.000 | 0.000 | 100.0% | 14.1% |
| 1 | 13 | raw input: relu(-11.48*x_t) | 1.000 | 0.000 | 100.0% | 4.2% |
| 1 | 14 | raw input: relu(-4.47*x_t) | 1.000 | 0.000 | 100.0% | 7.2% |
| 1 | 15 | raw input: relu(+0.05*x_t) | 1.000 | 0.000 | 100.0% | 0.0% |
| 2 | 0 | relu(+4.34*max +3.44*x_prev) | 0.789 | 0.460 | 93.2% | 18.9% |
| 2 | 1 | raw input: relu(+6.78*x_t) | 0.999 | 0.024 | 99.9% | 7.1% |
| 2 | 2 | raw input: relu(-7.97*x_prev) | 0.965 | 0.187 | 98.0% | 13.6% |
| 2 | 3 | relu(-5.26*2nd -4.40*x_prev) | 0.851 | 0.386 | 94.3% | 16.8% |
| 2 | 4 | raw input: relu(-11.28*x_t) | 1.000 | 0.007 | 99.9% | 10.8% |
| 2 | 5 | raw input: relu(+4.93*x_t) | 0.999 | 0.032 | 100.0% | 5.1% |
| 2 | 6 | raw input: relu(+0.20*x_t +0.04*x_prev) | 0.980 | 0.143 | 100.0% | 0.1% |
| 2 | 7 | raw input: relu(+10.90*x_t -10.84*x_prev) | 1.000 | 0.008 | 99.8% | 40.2% |
| 2 | 8 | raw input: relu(+0.13*x_t +0.03*x_prev) | 0.980 | 0.140 | 100.0% | 0.2% |
| 2 | 9 | raw input: relu(+0.06*x_t -0.03*x_prev) | 0.786 | 0.463 | 100.0% | 0.1% |
| 2 | 10 | relu(-14.20*2nd -12.25*x_prev) | 0.859 | 0.375 | 88.2% | 35.5% |
| 2 | 11 | raw input: relu(-10.31*x_t +10.37*x_prev) | 1.000 | 0.002 | 100.0% | 34.7% |
| 2 | 12 | raw input: relu(-10.80*x_t) | 1.000 | 0.007 | 99.8% | 15.9% |
| 2 | 13 | raw input: relu(-11.46*x_t) | 1.000 | 0.006 | 100.0% | 4.6% |
| 2 | 14 | comparator: relu(max - x_t) | 0.891 | 0.331 | 96.2% | 14.9% |
| 2 | 15 | relu(-6.19*x_t +12.48*max) | 0.625 | 0.613 | 83.1% | 27.2% |
| 3 | 0 | relu(+4.17*max +4.85*x_prev) | 0.810 | 0.436 | 88.2% | 25.3% |
| 3 | 1 | filter: relu(-6.96*x_1 -0.16*x_2 +8.51*x_3) | 0.879 | 0.348 | 96.2% | 16.9% |
| 3 | 2 | relu(-4.07*min -2.95*x_prev) | 0.490 | 0.714 | 88.0% | 17.5% |
| 3 | 3 | relu(-2.48*max_prev -15.76*2nd_prev) | 0.951 | 0.222 | 93.4% | 30.9% |
| 3 | 4 | raw input: relu(-11.35*x_t) | 0.999 | 0.024 | 99.8% | 12.9% |
| 3 | 5 | filter: relu(-0.44*x_1 -0.31*x_2 +4.61*x_3) | 0.982 | 0.136 | 99.6% | 5.1% |
| 3 | 6 | relu(+9.34*max_prev -14.35*2nd_prev) | 0.451 | 0.741 | 87.7% | 17.2% |
| 3 | 7 | raw input: relu(+10.88*x_t -10.86*x_prev) | 1.000 | 0.007 | 99.9% | 44.3% |
| 3 | 8 | relu(+3.52*max -1.92*max_prev) | -0.034 | 1.017 | 94.9% | 5.5% |
| 3 | 9 | relu(-6.37*x_prev +13.01*max_prev) | 0.596 | 0.635 | 83.4% | 23.4% |
| 3 | 10 | relu(-29.68*x_prev +23.00*max_prev) | 0.951 | 0.221 | 89.8% | 41.0% |
| 3 | 11 | raw input: relu(-10.34*x_t +10.50*x_prev) | 1.000 | 0.012 | 99.3% | 37.6% |
| 3 | 12 | filter: relu(+11.04*x_1 +0.06*x_2 -10.92*x_3) | 1.000 | 0.013 | 99.4% | 43.2% |
| 3 | 13 | raw input: relu(-11.49*x_t) | 0.999 | 0.026 | 99.9% | 8.1% |
| 3 | 14 | raw input: relu(-5.36*x_t +4.32*x_prev) | 0.871 | 0.359 | 95.3% | 15.3% |
| 3 | 15 | relu(+14.68*x_prev -16.06*2nd_prev) | 0.967 | 0.182 | 92.3% | 41.6% |
| 4 | 0 | filter: relu(-4.65*x_1 +0.06*x_2 +6.63*x_3 -0.09*x_4) | 0.621 | 0.616 | 82.8% | 35.8% |
| 4 | 1 | filter: relu(+0.02*x_1 -6.95*x_2 -0.17*x_3 +8.12*x_4) | 0.818 | 0.426 | 93.9% | 21.5% |
| 4 | 2 | relu(+3.78*x_t -14.66*min) | 0.374 | 0.791 | 72.6% | 34.7% |
| 4 | 3 | filter: relu(+24.65*x_1 -12.45*x_2 -11.68*x_3 -0.20*x_4) | 0.922 | 0.279 | 90.5% | 34.9% |
| 4 | 4 | filter: relu(+11.72*x_1 -0.14*x_2 -0.14*x_3 -11.39*x_4) | 1.000 | 0.015 | 99.0% | 46.1% |
| 4 | 5 | filter: relu(-1.17*x_1 -0.81*x_2 -0.89*x_3 +4.92*x_4) | 0.939 | 0.247 | 98.5% | 9.0% |
| 4 | 6 | relu(+4.36*max -9.69*min) | 0.167 | 0.913 | 79.0% | 24.2% |
| 4 | 7 | raw input: relu(+10.91*x_t -10.82*x_prev) | 1.000 | 0.013 | 99.4% | 48.7% |
| 4 | 8 | relu(+8.85*max +3.15*2nd_prev) | 0.230 | 0.878 | 84.8% | 17.0% |
| 4 | 9 | relu(-9.63*min +12.12*max_prev) | 0.343 | 0.810 | 66.7% | 41.2% |
| 4 | 10 | relu(-24.98*x_prev +18.80*max_prev) | 0.905 | 0.308 | 83.2% | 47.5% |
| 4 | 11 | raw input: relu(-10.32*x_t +10.51*x_prev) | 1.000 | 0.020 | 99.2% | 38.5% |
| 4 | 12 | filter: relu(-0.17*x_1 +11.12*x_2 -0.03*x_3 -10.92*x_4) | 1.000 | 0.018 | 99.1% | 50.0% |
| 4 | 13 | raw input: relu(-11.56*x_t) | 0.999 | 0.037 | 100.0% | 1.4% |
| 4 | 14 | raw input: relu(-5.13*x_t +4.06*x_prev) | 0.756 | 0.494 | 89.7% | 26.6% |
| 4 | 15 | relu(-9.30*min +8.74*x_prev) | 0.465 | 0.731 | 69.2% | 40.8% |
| 5 | 0 | filter: relu(-3.25*x_1 -5.89*x_2 -0.38*x_3 +7.24*x_4 +0.09*x_5) | 0.677 | 0.568 | 92.0% | 19.3% |
| 5 | 1 | filter: relu(-0.56*x_1 -0.33*x_2 -6.90*x_3 -0.19*x_4 +8.25*x_5) | 0.816 | 0.429 | 90.0% | 19.4% |
| 5 | 2 | filter: relu(+15.48*x_1 -10.35*x_2 +1.50*x_3 -6.01*x_4 +0.72*x_5) | 0.348 | 0.808 | 70.7% | 31.8% |
| 5 | 3 | filter: relu(-0.41*x_1 +24.70*x_2 -13.07*x_3 -11.29*x_4 -0.01*x_5) | 0.934 | 0.257 | 89.8% | 37.6% |
| 5 | 4 | filter: relu(-0.22*x_1 +11.84*x_2 -0.24*x_3 -0.13*x_4 -11.33*x_5) | 0.999 | 0.026 | 98.4% | 44.4% |
| 5 | 5 | filter: relu(-1.04*x_1 -1.49*x_2 -1.07*x_3 -0.73*x_4 +5.04*x_5) | 0.934 | 0.257 | 97.9% | 8.7% |
| 5 | 6 | relu(-7.18*min +7.84*max_prev) | 0.161 | 0.916 | 60.4% | 40.2% |
| 5 | 7 | raw input: relu(+10.86*x_t -10.77*x_prev) | 1.000 | 0.021 | 99.1% | 29.7% |
| 5 | 8 | relu(-9.15*min +10.05*max_prev) | 0.170 | 0.911 | 70.9% | 30.5% |
| 5 | 9 | relu(-9.06*min +10.02*max_prev) | 0.243 | 0.870 | 68.4% | 33.7% |
| 5 | 10 | relu(-22.96*x_prev +18.23*max_prev) | 0.862 | 0.372 | 77.0% | 52.0% |
| 5 | 11 | raw input: relu(-10.33*x_t +10.59*x_prev) | 0.999 | 0.027 | 98.2% | 41.5% |
| 5 | 12 | filter: relu(-0.10*x_1 -0.10*x_2 +11.11*x_3 -0.01*x_4 -10.90*x_5) | 0.999 | 0.023 | 98.5% | 45.1% |
| 5 | 13 | filter: relu(+12.01*x_1 -0.07*x_2 -0.10*x_3 -0.23*x_4 -11.66*x_5) | 0.999 | 0.029 | 98.7% | 44.7% |
| 5 | 14 | raw input: relu(-4.08*x_t +3.36*x_prev) | 0.679 | 0.567 | 99.3% | 1.9% |
| 5 | 15 | relu(-9.09*min +8.71*x_prev) | 0.395 | 0.778 | 83.4% | 22.1% |

*Sufficiency: decisions kept when only this neuron-step is replaced by its unit formula. Importance: decisions lost when it is set to its mean. Both are on held-out inputs. Units are approximate in dense networks (UNITS_PROGRAM, first-sweep conclusion), so a high fit does not imply exact sufficiency.*
