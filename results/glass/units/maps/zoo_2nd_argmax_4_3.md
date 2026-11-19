# Unit map: zoo_2nd_argmax_4_3

Coverage after the first step (variance-weighted share of neuron-steps fitted at R² ≥ 0.9 on held-out inputs): history units 0.00, raw-input units 1.00. Shuffled-weights control: history 0.02, raw 0.00.

## Per neuron (most common label over its explained steps)

| neuron | label |
|---|---|
| 0 | raw input: relu(+3.95*x_t -4.92*x_prev) |
| 1 | raw input: relu(+4.68*x_t -3.71*x_prev) |
| 2 | raw input: relu(-4.70*x_t +3.73*x_prev) |
| 3 | raw input: relu(-3.95*x_t +4.92*x_prev) |

## Every neuron-step

| step | neuron | unit | fit (held-out R²) | sufficiency | importance |
|---|---|---|---|---|---|
| 1 | 0 | raw input: relu(+3.97*x_t) | 1.000 | 100.0% | 35.4% |
| 1 | 1 | raw input: relu(+4.70*x_t) | 1.000 | 100.0% | 52.3% |
| 1 | 2 | raw input: relu(-4.72*x_t) | 1.000 | 100.0% | 50.8% |
| 1 | 3 | raw input: relu(-3.97*x_t) | 1.000 | 100.0% | 35.9% |
| 2 | 0 | raw input: relu(+3.95*x_t -4.92*x_prev) | 1.000 | 99.8% | 52.8% |
| 2 | 1 | raw input: relu(+4.68*x_t -3.71*x_prev) | 1.000 | 99.7% | 51.6% |
| 2 | 2 | raw input: relu(-4.70*x_t +3.73*x_prev) | 1.000 | 99.5% | 52.0% |
| 2 | 3 | raw input: relu(-3.95*x_t +4.92*x_prev) | 1.000 | 99.7% | 52.6% |
| 3 | 0 | filter: relu(+0.85*x_1 -4.78*x_2 +3.77*x_3) | 0.994 | 95.8% | 55.6% |
| 3 | 1 | filter: relu(-0.87*x_1 -3.63*x_2 +4.50*x_3) | 0.995 | 96.4% | 56.9% |
| 3 | 2 | filter: relu(+0.87*x_1 +3.64*x_2 -4.51*x_3) | 0.994 | 96.0% | 56.0% |
| 3 | 3 | filter: relu(-0.81*x_1 +4.77*x_2 -3.79*x_3) | 0.993 | 96.2% | 56.6% |

*Sufficiency: decisions kept when only this neuron-step is replaced by its unit formula. Importance: decisions lost when it is set to its mean. Both are on held-out inputs. Units are approximate in dense networks (UNITS_PROGRAM, first-sweep conclusion), so a high fit does not imply exact sufficiency.*
