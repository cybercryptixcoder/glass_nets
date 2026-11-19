# Unit map: zoo_2nd_argmax_2_2

Coverage after the first step (variance-weighted share of neuron-steps fitted at R² ≥ 0.9 on held-out inputs): history units 0.00, raw-input units 1.00. Shuffled-weights control: history 0.18, raw 0.00.

Precision requirement ε95 = 0.1584893192461114: hidden-state noise (as a share of each neuron's spread) at which 5% of decisions flip. Substitution-grade units (relative error ≤ ε95): 3 of 4 neuron-steps.

## Per neuron (most common label over its explained steps)

| neuron | label |
|---|---|
| 0 | raw input: relu(+5.46*x_t -7.48*x_prev) |
| 1 | raw input: relu(-7.23*x_t +7.40*x_prev) |

## Every neuron-step

| step | neuron | unit | fit (held-out R²) | relative error | sufficiency | importance |
|---|---|---|---|---|---|---|
| 1 | 0 | raw input: relu(+4.17*x_t) | 1.000 | 0.000 | 100.0% | 13.1% |
| 1 | 1 | raw input: relu(-6.75*x_t) | 1.000 | 0.000 | 100.0% | 13.3% |
| 2 | 0 | raw input: relu(+5.46*x_t -7.48*x_prev) | 0.938 | 0.248 | 98.0% | 4.3% |
| 2 | 1 | raw input: relu(-7.23*x_t +7.40*x_prev) | 0.994 | 0.080 | 97.4% | 39.3% |

*Sufficiency: decisions kept when only this neuron-step is replaced by its unit formula. Importance: decisions lost when it is set to its mean. Both are on held-out inputs. Units are approximate in dense networks (UNITS_PROGRAM, first-sweep conclusion), so a high fit does not imply exact sufficiency.*
