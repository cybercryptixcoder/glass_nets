# Unit map: zoo_2nd_argmax_8_8

Coverage after the first step (variance-weighted share of neuron-steps fitted at R² ≥ 0.9 on held-out inputs): history units 0.00, raw-input units 0.00. Shuffled-weights control: history 0.00, raw 0.47.

## Per neuron (most common label over its explained steps)

| neuron | label |
|---|---|
| 0 | unexplained |
| 1 | unexplained |
| 2 | raw input: relu(-1.34*x_t) |
| 3 | unexplained |
| 4 | unexplained |
| 5 | unexplained |
| 6 | unexplained |
| 7 | unexplained |

## Every neuron-step

| step | neuron | unit | fit (held-out R²) | sufficiency | importance |
|---|---|---|---|---|---|
| 1 | 0 | raw input: relu(-0.21*x_t) | 1.000 | 100.0% | 10.2% |
| 1 | 1 | raw input: relu(+0.03*x_t) | 1.000 | 100.0% | 1.5% |
| 1 | 2 | raw input: relu(-1.20*x_t) | 1.000 | 100.0% | 25.0% |
| 1 | 3 | raw input: relu(-0.15*x_t) | 1.000 | 100.0% | 8.2% |
| 1 | 4 | raw input: relu(-0.47*x_t) | 1.000 | 100.0% | 24.3% |
| 1 | 5 | raw input: relu(+0.00*x_t) | 1.000 | 100.0% | 0.4% |
| 1 | 6 | raw input: relu(+0.62*x_t) | 1.000 | 100.0% | 23.1% |
| 1 | 7 | raw input: relu(+0.00*x_t) | 1.000 | 100.0% | 0.2% |
| 2 | 0 | relu(+0.38*max -0.60*2nd) | 0.158 | 68.3% | 33.6% |
| 2 | 1 | raw input: relu(+0.01*x_t) | 0.262 | 99.8% | 0.4% |
| 2 | 2 | raw input: relu(-1.34*x_t) | 0.968 | 96.3% | 25.9% |
| 2 | 3 | relu(-0.28*2nd +0.11*x_prev) | 0.595 | 91.8% | 18.5% |
| 2 | 4 | relu(-1.00*2nd +0.33*x_prev) | 0.530 | 75.4% | 37.1% |
| 2 | 5 | relu(-0.97*2nd -0.50*x_prev) | 0.693 | 70.4% | 42.0% |
| 2 | 6 | comparator: relu(x_t - 2nd) | 0.751 | 79.8% | 40.7% |
| 2 | 7 | none | -9.000 | nan% | nan% |
| 3 | 0 | relu(-1.32*min +0.59*x_prev) | 0.526 | 70.3% | 40.2% |
| 3 | 1 | relu(+0.01*max +0.03*min) | 0.566 | 99.5% | 0.8% |
| 3 | 2 | relu(-1.23*x_t +0.36*max) | 0.986 | 96.8% | 34.2% |
| 3 | 3 | relu(+1.06*x_prev -2.19*2nd_prev) | 0.725 | 63.0% | 58.8% |
| 3 | 4 | relu(-0.23*x_t -0.49*min) | 0.380 | 82.1% | 24.7% |
| 3 | 5 | relu(-2.88*2nd_prev) | 0.793 | 63.9% | 53.8% |
| 3 | 6 | relu(+0.70*x_t -2.37*2nd_prev) | 0.742 | 56.0% | 59.9% |
| 3 | 7 | relu(+0.01*max -0.01*x_prev) | 0.158 | 98.9% | 1.4% |
| 4 | 0 | relu(-4.01*min +1.23*x_prev) | 0.366 | 38.3% | 67.1% |
| 4 | 1 | filter: relu(-3.01*x_1 +0.10*x_2 -0.10*x_3 +0.13*x_4) | 0.669 | 68.3% | 73.0% |
| 4 | 2 | relu(-1.16*x_t +0.45*max) | 0.960 | 94.0% | 34.5% |
| 4 | 3 | filter: relu(+0.18*x_1 -1.61*x_2 +0.05*x_3 -0.22*x_4) | 0.485 | 51.1% | 60.7% |
| 4 | 4 | comparator: relu(max - x_t) | 0.471 | 81.5% | 24.5% |
| 4 | 5 | relu(+1.29*x_t -5.29*min) | 0.515 | 35.6% | 70.7% |
| 4 | 6 | relu(+1.50*x_t -3.60*min) | 0.575 | 44.0% | 64.7% |
| 4 | 7 | filter: relu(+0.06*x_1 +0.02*x_2 -0.03*x_3 +0.00*x_4) | 0.578 | 98.4% | 3.0% |
| 5 | 0 | relu(-5.34*min +1.41*x_prev) | 0.335 | 35.6% | 71.4% |
| 5 | 1 | relu(+0.88*x_t -4.00*min) | 0.289 | 33.3% | 72.9% |
| 5 | 2 | relu(-1.17*x_t +0.52*max) | 0.946 | 92.5% | 35.8% |
| 5 | 3 | filter: relu(+0.29*x_1 +0.28*x_2 -1.67*x_3 +0.16*x_4 -0.20*x_5) | 0.484 | 49.5% | 56.1% |
| 5 | 4 | comparator: relu(max - x_t) | 0.525 | 81.7% | 25.5% |
| 5 | 5 | relu(+1.56*max -4.74*min) | 0.369 | 34.6% | 70.2% |
| 5 | 6 | relu(+1.57*x_t -4.62*min) | 0.501 | 43.2% | 63.0% |
| 5 | 7 | filter: relu(-7.91*x_1 -0.13*x_2 -0.18*x_3 -0.78*x_4 +0.00*x_5) | 0.646 | 69.6% | 51.2% |
| 6 | 0 | relu(+1.96*max -3.75*min) | 0.197 | 32.8% | 69.9% |
| 6 | 1 | relu(-8.97*min +1.66*x_prev) | 0.278 | 31.5% | 72.6% |
| 6 | 2 | relu(-1.17*x_t +0.58*max) | 0.938 | 89.3% | 39.2% |
| 6 | 3 | filter: relu(+0.38*x_1 +0.30*x_2 +0.31*x_3 -1.65*x_4 +0.18*x_5 -0.23*x_6) | 0.347 | 45.5% | 53.5% |
| 6 | 4 | comparator: relu(max - x_t) | 0.600 | 78.7% | 30.1% |
| 6 | 5 | relu(+2.27*max -4.89*min) | 0.322 | 36.0% | 67.1% |
| 6 | 6 | relu(-13.02*min) | 0.196 | 32.2% | 66.8% |
| 6 | 7 | filter: relu(+1.29*x_1 -8.91*x_2 +0.83*x_3 +0.46*x_4 -0.40*x_5 +0.17*x_6) | 0.591 | 60.3% | 50.7% |
| 7 | 0 | relu(-15.64*min) | 0.156 | 28.7% | 72.3% |
| 7 | 1 | relu(+2.26*max -9.13*min) | 0.239 | 31.4% | 73.0% |
| 7 | 2 | comparator: relu(max - x_t) | 0.919 | 89.0% | 38.7% |
| 7 | 3 | filter: relu(-6.11*x_1 +0.11*x_2 +0.04*x_3 +0.27*x_4 -1.73*x_5 +0.20*x_6 -0.12*x_7) | 0.285 | 17.2% | 69.8% |
| 7 | 4 | filter: relu(-21.67*x_1 -0.64*x_2 -0.65*x_3 -0.37*x_4 +0.34*x_5 +0.30*x_6 -1.21*x_7) | 0.602 | 45.1% | 80.0% |
| 7 | 5 | relu(+3.32*max -4.08*min) | 0.225 | 37.6% | 61.7% |
| 7 | 6 | relu(+3.47*x_t -17.49*min) | 0.188 | 34.4% | 65.0% |
| 7 | 7 | filter: relu(+1.81*x_1 +1.68*x_2 -9.22*x_3 +1.22*x_4 +0.48*x_5 -0.34*x_6 +0.11*x_7) | 0.568 | 61.7% | 52.0% |
| 8 | 0 | relu(+5.48*max -18.13*min) | 0.164 | 22.9% | 78.0% |
| 8 | 1 | relu(+3.47*max -9.54*min) | 0.191 | 30.4% | 73.6% |
| 8 | 2 | comparator: relu(2nd - x_t) | 0.937 | 91.3% | 35.4% |
| 8 | 3 | relu(-6.64*min) | 0.097 | 33.8% | 69.1% |
| 8 | 4 | filter: relu(+3.02*x_1 -24.62*x_2 +1.28*x_3 +0.78*x_4 +0.13*x_5 +0.66*x_6 +0.41*x_7 -1.18*x_8) | 0.548 | 33.4% | 75.2% |
| 8 | 5 | relu(+3.92*max -8.22*min) | 0.181 | 32.6% | 69.2% |
| 8 | 6 | relu(+5.85*max -17.79*min) | 0.219 | 32.2% | 71.4% |
| 8 | 7 | filter: relu(-0.97*x_1 +1.56*x_2 +1.48*x_3 -8.97*x_4 +1.35*x_5 +0.40*x_6 -0.31*x_7 +0.15*x_8) | 0.115 | 41.5% | 49.8% |

*Sufficiency: decisions kept when only this neuron-step is replaced by its unit formula. Importance: decisions lost when it is set to its mean. Both are on held-out inputs. Units are approximate in dense networks (UNITS_PROGRAM, first-sweep conclusion), so a high fit does not imply exact sufficiency.*
