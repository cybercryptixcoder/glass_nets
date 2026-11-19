# Unit map: zoo_2nd_argmax_16_10

Coverage after the first step (variance-weighted share of neuron-steps fitted at R² ≥ 0.9 on held-out inputs): history units 0.37, raw-input units 0.06. Shuffled-weights control: history 0.00, raw 0.00.

Precision requirement ε95 = 0.006309573444801929: hidden-state noise (as a share of each neuron's spread) at which 5% of decisions flip. Substitution-grade units (relative error ≤ ε95): 16 of 160 neuron-steps.

## Per neuron (most common label over its explained steps)

| neuron | label |
|---|---|
| 0 | raw input: relu(+4.92*x_prev) |
| 1 | comparator: relu(max - x_t) |
| 2 | raw input: relu(+16.77*x_prev) |
| 3 | unexplained |
| 4 | comparator: relu(x_t - 2nd) |
| 5 | raw input: relu(+12.20*x_prev) |
| 6 | comparator: relu(max - x_t) |
| 7 | comparator: relu(max - x_t) |
| 8 | comparator: relu(2nd - x_t) |
| 9 | raw input: relu(-0.22*x_t +1.49*x_prev) |
| 10 | relu(+18.51*max_prev) |
| 11 | relu(+16.04*max_prev) |
| 12 | relu(+10.54*max_prev) |
| 13 | raw input: relu(+4.11*x_prev) |
| 14 | unexplained |
| 15 | unexplained |

## Every neuron-step

| step | neuron | unit | fit (held-out R²) | relative error | sufficiency | importance |
|---|---|---|---|---|---|---|
| 1 | 0 | raw input: relu(+0.07*x_t) | 1.000 | 0.000 | 100.0% | 0.0% |
| 1 | 1 | raw input: relu(-10.56*x_t) | 1.000 | 0.000 | 100.0% | 9.1% |
| 1 | 2 | raw input: relu(+0.01*x_t) | 1.000 | 0.000 | 100.0% | 0.0% |
| 1 | 3 | raw input: relu(-0.12*x_t) | 1.000 | 0.000 | 100.0% | 0.0% |
| 1 | 4 | raw input: relu(+10.16*x_t) | 1.000 | 0.000 | 100.0% | 24.2% |
| 1 | 5 | raw input: relu(-0.28*x_t) | 1.000 | 0.000 | 100.0% | 0.2% |
| 1 | 6 | raw input: relu(-11.00*x_t) | 1.000 | 0.000 | 100.0% | 13.5% |
| 1 | 7 | raw input: relu(-13.17*x_t) | 1.000 | 0.000 | 100.0% | 12.0% |
| 1 | 8 | raw input: relu(-12.31*x_t) | 1.000 | 0.000 | 100.0% | 12.1% |
| 1 | 9 | raw input: relu(-0.23*x_t) | 1.000 | 0.000 | 100.0% | 0.3% |
| 1 | 10 | raw input: relu(+0.06*x_t) | 1.000 | 0.000 | 100.0% | 0.0% |
| 1 | 11 | raw input: relu(+0.15*x_t) | 1.000 | 0.000 | 100.0% | 0.0% |
| 1 | 12 | raw input: relu(+0.30*x_t) | 1.000 | 0.000 | 100.0% | 0.1% |
| 1 | 13 | raw input: relu(-0.05*x_t) | 1.000 | 0.000 | 100.0% | 0.0% |
| 1 | 14 | raw input: relu(-1.32*x_t) | 1.000 | 0.000 | 100.0% | 0.6% |
| 1 | 15 | raw input: relu(-0.12*x_t) | 1.000 | 0.000 | 100.0% | 0.0% |
| 2 | 0 | raw input: relu(+4.92*x_prev) | 0.940 | 0.244 | 99.5% | 3.5% |
| 2 | 1 | comparator: relu(max - x_t) | 0.948 | 0.228 | 98.1% | 14.5% |
| 2 | 2 | raw input: relu(+16.77*x_prev) | 0.997 | 0.057 | 98.2% | 29.4% |
| 2 | 3 | relu(+1.33*max +1.02*x_prev) | 0.751 | 0.499 | 99.6% | 1.0% |
| 2 | 4 | raw input: relu(+7.75*x_t -2.48*x_prev) | 0.896 | 0.323 | 92.9% | 27.6% |
| 2 | 5 | raw input: relu(+12.20*x_prev) | 0.999 | 0.028 | 99.9% | 12.2% |
| 2 | 6 | raw input: relu(-10.80*x_t) | 0.999 | 0.031 | 99.6% | 18.6% |
| 2 | 7 | comparator: relu(max - x_t) | 0.931 | 0.263 | 94.1% | 27.8% |
| 2 | 8 | raw input: relu(-12.15*x_t) | 0.999 | 0.023 | 99.7% | 17.7% |
| 2 | 9 | raw input: relu(-0.22*x_t +1.49*x_prev) | 0.999 | 0.031 | 100.0% | 3.0% |
| 2 | 10 | relu(+6.83*max +5.58*x_prev) | 0.804 | 0.442 | 95.9% | 10.0% |
| 2 | 11 | relu(+6.84*max +5.37*x_prev) | 0.803 | 0.444 | 96.8% | 10.2% |
| 2 | 12 | relu(+5.44*max +4.10*x_prev) | 0.826 | 0.418 | 97.8% | 7.4% |
| 2 | 13 | raw input: relu(+4.11*x_prev) | 0.980 | 0.143 | 99.8% | 2.2% |
| 2 | 14 | relu(-4.09*2nd +6.83*x_prev) | 0.824 | 0.419 | 98.0% | 3.4% |
| 2 | 15 | relu(+0.65*max +0.63*x_prev) | 0.774 | 0.475 | 99.8% | 0.7% |
| 3 | 0 | filter: relu(-2.20*x_1 +3.50*x_2 +0.09*x_3) | 0.813 | 0.432 | 98.5% | 6.3% |
| 3 | 1 | raw input: relu(-12.75*x_t +6.47*x_prev) | 0.908 | 0.304 | 95.8% | 22.4% |
| 3 | 2 | relu(+16.09*max_prev) | 0.986 | 0.119 | 94.9% | 41.8% |
| 3 | 3 | relu(+2.07*max_prev -1.22*2nd_prev) | 0.723 | 0.526 | 99.6% | 1.0% |
| 3 | 4 | comparator: relu(x_t - 2nd) | 0.903 | 0.312 | 93.2% | 25.2% |
| 3 | 5 | raw input: relu(+11.47*x_prev) | 0.983 | 0.130 | 99.0% | 11.7% |
| 3 | 6 | filter: relu(+5.51*x_1 +0.56*x_2 -13.13*x_3) | 0.918 | 0.287 | 95.1% | 27.2% |
| 3 | 7 | comparator: relu(max - x_t) | 0.977 | 0.153 | 96.0% | 34.4% |
| 3 | 8 | filter: relu(+3.79*x_1 +1.56*x_2 -13.90*x_3) | 0.932 | 0.260 | 96.2% | 22.8% |
| 3 | 9 | relu(+66.46*x_prev -64.89*max_prev) | 0.970 | 0.174 | 99.6% | 4.4% |
| 3 | 10 | relu(+18.51*max_prev) | 0.959 | 0.202 | 99.0% | 8.7% |
| 3 | 11 | relu(+16.04*max_prev) | 0.962 | 0.196 | 98.1% | 13.9% |
| 3 | 12 | relu(+10.54*max_prev) | 0.947 | 0.230 | 98.7% | 7.5% |
| 3 | 13 | relu(-1.33*x_prev +5.74*max_prev) | 0.969 | 0.176 | 99.6% | 3.0% |
| 3 | 14 | raw input: relu(-2.44*x_t +5.99*x_prev) | 0.834 | 0.407 | 98.8% | 3.2% |
| 3 | 15 | relu(-1.31*x_prev +2.96*max_prev) | 0.842 | 0.398 | 99.4% | 1.5% |
| 4 | 0 | filter: relu(+2.73*x_1 -3.27*x_2 +3.06*x_3 +0.06*x_4) | 0.579 | 0.649 | 95.0% | 8.3% |
| 4 | 1 | comparator: relu(max - x_t) | 0.858 | 0.377 | 92.3% | 28.8% |
| 4 | 2 | relu(+17.63*max_prev) | 0.998 | 0.046 | 97.7% | 45.7% |
| 4 | 3 | relu(+2.65*max_prev) | 0.508 | 0.702 | 98.0% | 3.6% |
| 4 | 4 | comparator: relu(x_t - 2nd) | 0.941 | 0.243 | 96.5% | 21.3% |
| 4 | 5 | filter: relu(-0.38*x_1 -1.17*x_2 +11.15*x_3 -0.25*x_4) | 0.975 | 0.157 | 98.6% | 12.3% |
| 4 | 6 | comparator: relu(max - x_t) | 0.839 | 0.401 | 85.7% | 37.1% |
| 4 | 7 | comparator: relu(max - x_t) | 0.993 | 0.085 | 98.0% | 40.7% |
| 4 | 8 | comparator: relu(2nd - x_t) | 0.919 | 0.285 | 94.3% | 28.9% |
| 4 | 9 | relu(+122.50*x_prev -120.90*max_prev) | 0.976 | 0.155 | 99.6% | 4.8% |
| 4 | 10 | relu(+19.83*max_prev) | 0.963 | 0.193 | 98.1% | 11.2% |
| 4 | 11 | relu(+14.90*max_prev -3.82*2nd_prev) | 0.888 | 0.335 | 96.8% | 13.5% |
| 4 | 12 | relu(+1.20*x_prev +7.46*max_prev) | 0.640 | 0.600 | 89.5% | 14.7% |
| 4 | 13 | relu(+6.06*max_prev) | 0.871 | 0.359 | 98.3% | 5.7% |
| 4 | 14 | relu(+1.83*x_prev +4.92*max_prev) | 0.670 | 0.575 | 96.2% | 6.5% |
| 4 | 15 | relu(+1.63*max_prev -1.23*2nd_prev) | 0.335 | 0.816 | 99.1% | 1.3% |
| 5 | 0 | relu(+6.05*max_prev -2.21*2nd_prev) | 0.304 | 0.834 | 87.7% | 13.7% |
| 5 | 1 | comparator: relu(max - x_t) | 0.871 | 0.359 | 89.9% | 33.1% |
| 5 | 2 | relu(+18.03*max_prev) | 0.997 | 0.051 | 97.2% | 45.1% |
| 5 | 3 | relu(+1.89*max_prev) | 0.242 | 0.871 | 97.1% | 4.0% |
| 5 | 4 | comparator: relu(x_t - 2nd) | 0.970 | 0.174 | 98.2% | 19.8% |
| 5 | 5 | filter: relu(-1.30*x_1 -0.54*x_2 -1.13*x_3 +10.02*x_4 -0.20*x_5) | 0.953 | 0.217 | 97.8% | 14.7% |
| 5 | 6 | comparator: relu(max - x_t) | 0.870 | 0.360 | 78.2% | 46.8% |
| 5 | 7 | comparator: relu(max - x_t) | 0.995 | 0.068 | 95.9% | 48.4% |
| 5 | 8 | comparator: relu(2nd - x_t) | 0.951 | 0.220 | 93.6% | 36.1% |
| 5 | 9 | relu(+84.08*x_prev -82.54*max_prev) | 0.974 | 0.161 | 99.7% | 5.4% |
| 5 | 10 | relu(+18.69*max_prev -4.74*2nd_prev) | 0.876 | 0.352 | 96.5% | 12.5% |
| 5 | 11 | relu(+11.87*max_prev -4.48*2nd_prev) | 0.581 | 0.647 | 89.9% | 16.5% |
| 5 | 12 | relu(+4.35*max +1.91*x_prev) | 0.305 | 0.834 | 85.2% | 18.1% |
| 5 | 13 | relu(+6.76*max_prev) | 0.843 | 0.396 | 96.1% | 10.2% |
| 5 | 14 | relu(-1.30*x_t +7.77*max_prev) | 0.526 | 0.688 | 95.2% | 10.7% |
| 5 | 15 | relu(+1.00*max_prev -0.95*2nd_prev) | 0.199 | 0.895 | 99.4% | 1.0% |
| 6 | 0 | relu(-4.25*2nd +9.41*max_prev) | 0.303 | 0.835 | 82.1% | 21.3% |
| 6 | 1 | comparator: relu(max - x_t) | 0.865 | 0.367 | 80.9% | 47.2% |
| 6 | 2 | relu(+17.83*max_prev) | 0.988 | 0.108 | 95.6% | 42.6% |
| 6 | 3 | relu(+1.45*max_prev) | 0.133 | 0.931 | 96.1% | 4.4% |
| 6 | 4 | comparator: relu(x_t - 2nd) | 0.984 | 0.127 | 99.0% | 16.6% |
| 6 | 5 | filter: relu(-1.09*x_1 -1.29*x_2 -0.57*x_3 -1.12*x_4 +9.03*x_5 -0.15*x_6) | 0.933 | 0.259 | 96.5% | 17.0% |
| 6 | 6 | comparator: relu(max - x_t) | 0.894 | 0.325 | 77.2% | 51.0% |
| 6 | 7 | comparator: relu(max - x_t) | 0.989 | 0.107 | 87.3% | 55.7% |
| 6 | 8 | comparator: relu(2nd - x_t) | 0.948 | 0.229 | 94.0% | 42.2% |
| 6 | 9 | relu(+17.26*x_prev -15.77*max_prev) | 0.960 | 0.200 | 99.4% | 5.8% |
| 6 | 10 | relu(+15.71*max_prev -6.13*2nd_prev) | 0.598 | 0.634 | 92.8% | 14.4% |
| 6 | 11 | relu(+9.54*max_prev -5.30*2nd_prev) | 0.331 | 0.818 | 79.1% | 28.7% |
| 6 | 12 | relu(+4.08*max +1.66*x_prev) | 0.305 | 0.834 | 86.1% | 18.3% |
| 6 | 13 | relu(+6.88*max_prev -1.62*2nd_prev) | 0.776 | 0.473 | 95.7% | 10.6% |
| 6 | 14 | relu(+10.56*max_prev) | 0.482 | 0.719 | 90.5% | 16.3% |
| 6 | 15 | filter: relu(+1.20*x_1 +0.28*x_2 -0.59*x_3 -0.50*x_4 +0.02*x_5 -0.00*x_6) | 0.209 | 0.889 | 98.0% | 2.7% |
| 7 | 0 | relu(+12.30*max_prev -8.53*2nd_prev) | 0.350 | 0.806 | 79.5% | 24.8% |
| 7 | 1 | comparator: relu(max - x_t) | 0.863 | 0.370 | 75.0% | 52.6% |
| 7 | 2 | relu(+17.39*max_prev) | 0.962 | 0.194 | 93.8% | 38.9% |
| 7 | 3 | relu(+1.67*max_prev) | 0.151 | 0.922 | 95.1% | 5.7% |
| 7 | 4 | comparator: relu(x_t - 2nd) | 0.985 | 0.124 | 99.0% | 14.8% |
| 7 | 5 | filter: relu(-0.56*x_1 -1.13*x_2 -1.18*x_3 -0.86*x_4 -1.12*x_5 +8.65*x_6 -0.14*x_7) | 0.890 | 0.332 | 94.0% | 18.8% |
| 7 | 6 | comparator: relu(max - x_t) | 0.910 | 0.300 | 75.3% | 54.4% |
| 7 | 7 | comparator: relu(max - x_t) | 0.980 | 0.143 | 77.0% | 61.4% |
| 7 | 8 | comparator: relu(2nd - x_t) | 0.886 | 0.338 | 86.3% | 51.7% |
| 7 | 9 | relu(+12.73*x_prev -11.26*max_prev) | 0.948 | 0.227 | 99.6% | 7.0% |
| 7 | 10 | relu(+12.98*max_prev -7.13*2nd_prev) | 0.344 | 0.810 | 80.0% | 26.4% |
| 7 | 11 | relu(+2.17*x_prev +5.85*max_prev) | 0.263 | 0.859 | 78.3% | 25.1% |
| 7 | 12 | relu(+8.22*max_prev -4.44*2nd_prev) | 0.318 | 0.826 | 83.1% | 22.9% |
| 7 | 13 | relu(+6.62*max_prev -2.03*2nd_prev) | 0.709 | 0.539 | 95.2% | 12.4% |
| 7 | 14 | relu(+12.14*max_prev) | 0.493 | 0.712 | 87.1% | 20.2% |
| 7 | 15 | filter: relu(+3.36*x_1 +0.51*x_2 -0.18*x_3 -1.21*x_4 -1.58*x_5 -0.22*x_6 -0.07*x_7) | 0.389 | 0.782 | 97.1% | 4.3% |
| 8 | 0 | relu(+13.32*max_prev -9.39*2nd_prev) | 0.371 | 0.793 | 71.8% | 33.0% |
| 8 | 1 | comparator: relu(max - x_t) | 0.884 | 0.341 | 72.2% | 59.9% |
| 8 | 2 | relu(+17.01*max_prev) | 0.932 | 0.260 | 92.8% | 33.2% |
| 8 | 3 | relu(+2.39*max_prev) | 0.175 | 0.908 | 96.1% | 5.4% |
| 8 | 4 | comparator: relu(x_t - 2nd) | 0.983 | 0.130 | 99.3% | 7.7% |
| 8 | 5 | filter: relu(-0.38*x_1 -0.81*x_2 -1.09*x_3 -1.22*x_4 -1.00*x_5 -1.03*x_6 +8.37*x_7 -0.13*x_8) | 0.866 | 0.366 | 92.2% | 20.5% |
| 8 | 6 | comparator: relu(max - x_t) | 0.898 | 0.319 | 72.1% | 55.6% |
| 8 | 7 | comparator: relu(max - x_t) | 0.977 | 0.151 | 72.5% | 65.1% |
| 8 | 8 | comparator: relu(max - x_t) | 0.823 | 0.420 | 66.8% | 61.5% |
| 8 | 9 | relu(+16.52*x_prev -15.07*max_prev) | 0.953 | 0.216 | 99.6% | 7.3% |
| 8 | 10 | relu(-5.88*2nd +11.15*max_prev) | 0.269 | 0.855 | 73.5% | 31.4% |
| 8 | 11 | relu(+1.89*x_prev +6.13*max_prev) | 0.304 | 0.834 | 78.2% | 25.8% |
| 8 | 12 | relu(+11.71*max_prev -6.81*2nd_prev) | 0.286 | 0.845 | 74.1% | 32.0% |
| 8 | 13 | relu(-1.81*2nd +6.35*max_prev) | 0.629 | 0.609 | 91.9% | 16.2% |
| 8 | 14 | relu(+12.19*max_prev) | 0.502 | 0.706 | 84.6% | 24.9% |
| 8 | 15 | filter: relu(+6.53*x_1 +2.68*x_2 -0.37*x_3 -0.05*x_4 -1.61*x_5 -2.18*x_6 -0.41*x_7 -0.37*x_8) | 0.315 | 0.828 | 89.6% | 16.4% |
| 9 | 0 | relu(+13.45*max_prev -9.74*2nd_prev) | 0.342 | 0.811 | 71.2% | 35.7% |
| 9 | 1 | comparator: relu(max - x_t) | 0.881 | 0.345 | 68.1% | 63.3% |
| 9 | 2 | relu(+16.77*max_prev) | 0.918 | 0.286 | 92.9% | 30.6% |
| 9 | 3 | relu(+3.66*max_prev) | 0.191 | 0.900 | 88.3% | 15.9% |
| 9 | 4 | comparator: relu(x_t - 2nd) | 0.984 | 0.128 | 99.2% | 5.7% |
| 9 | 5 | relu(-6.12*2nd +13.51*x_prev) | 0.856 | 0.379 | 93.2% | 19.3% |
| 9 | 6 | comparator: relu(max - x_t) | 0.908 | 0.304 | 69.4% | 61.4% |
| 9 | 7 | comparator: relu(max - x_t) | 0.979 | 0.145 | 70.4% | 69.0% |
| 9 | 8 | comparator: relu(max - x_t) | 0.821 | 0.423 | 58.0% | 65.7% |
| 9 | 9 | relu(+16.60*x_prev -15.23*max_prev) | 0.920 | 0.283 | 99.3% | 5.7% |
| 9 | 10 | relu(+11.24*max_prev -7.02*2nd_prev) | 0.299 | 0.837 | 69.2% | 36.5% |
| 9 | 11 | relu(+10.56*max_prev -6.27*2nd_prev) | 0.332 | 0.818 | 73.7% | 30.5% |
| 9 | 12 | relu(-6.02*2nd +13.70*max_prev) | 0.264 | 0.858 | 67.1% | 36.8% |
| 9 | 13 | relu(+6.06*max_prev) | 0.527 | 0.688 | 90.2% | 18.6% |
| 9 | 14 | relu(-1.27*x_t +11.17*max_prev) | 0.456 | 0.738 | 85.2% | 24.4% |
| 9 | 15 | relu(+7.09*max_prev) | 0.185 | 0.903 | 77.6% | 26.7% |
| 10 | 0 | relu(+12.73*max_prev -8.89*2nd_prev) | 0.336 | 0.815 | 67.3% | 38.6% |
| 10 | 1 | comparator: relu(max - x_t) | 0.868 | 0.364 | 62.5% | 66.5% |
| 10 | 2 | relu(+16.79*max_prev) | 0.922 | 0.279 | 95.2% | 23.6% |
| 10 | 3 | relu(+2.84*max_prev +2.29*2nd_prev) | 0.199 | 0.895 | 87.1% | 19.0% |
| 10 | 4 | comparator: relu(x_t - 2nd) | 0.982 | 0.136 | 98.7% | 3.5% |
| 10 | 5 | relu(-6.12*2nd +13.66*x_prev) | 0.856 | 0.380 | 93.0% | 19.4% |
| 10 | 6 | comparator: relu(max - x_t) | 0.913 | 0.296 | 66.1% | 63.3% |
| 10 | 7 | comparator: relu(max - x_t) | 0.982 | 0.135 | 76.0% | 68.3% |
| 10 | 8 | comparator: relu(max - x_t) | 0.823 | 0.421 | 50.4% | 67.4% |
| 10 | 9 | relu(+18.86*x_prev -17.54*max_prev) | 0.896 | 0.322 | 99.6% | 1.7% |
| 10 | 10 | relu(-6.30*2nd +11.66*max_prev) | 0.296 | 0.839 | 62.5% | 42.6% |
| 10 | 11 | relu(-6.22*2nd +11.60*max_prev) | 0.302 | 0.836 | 71.4% | 32.2% |
| 10 | 12 | relu(+15.37*max_prev -6.64*2nd_prev) | 0.281 | 0.848 | 64.4% | 39.9% |
| 10 | 13 | relu(+6.52*max_prev) | 0.474 | 0.725 | 89.8% | 19.9% |
| 10 | 14 | relu(+12.41*max_prev -4.62*2nd_prev) | 0.357 | 0.802 | 71.4% | 33.8% |
| 10 | 15 | relu(+7.60*max_prev) | 0.201 | 0.894 | 76.1% | 28.1% |

*Sufficiency: decisions kept when only this neuron-step is replaced by its unit formula. Importance: decisions lost when it is set to its mean. Both are on held-out inputs. Units are approximate in dense networks (UNITS_PROGRAM, first-sweep conclusion), so a high fit does not imply exact sufficiency.*
