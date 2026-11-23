# Recurrent Equilibrium Networks: Unconstrained Learning of Stable and Robust Dynamical Models

- **Key:** arxiv-2104.05942
- **Authors / organisation:** Max Revay; Ruigang Wang; Ian R. Manchester — Australian Centre for Field Robotics and Sydney Institute for Robotics and Intelligent Systems, University of Sydney
- **Date:** 2021-04-13
- **Link:** https://arxiv.org/abs/2104.05942
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Learning a model with feedback (a recurrent net, a state-space model, an autoregressive process) routinely produces a model that is unstable even when the system that generated the data is stable, and stability is not a convex constraint on matrices, let alone on a deep nonlinear model. The authors want a *model class* rather than a training penalty: a set of parameterised dynamical systems in which every member is stable and robust by construction, so that ordinary unconstrained gradient descent can be run on it without ever projecting back into a feasible set or solving a linear matrix inequality per step. Stability is taken in a strong form — contraction — which guarantees that all solutions of the model converge to one another regardless of initial condition and input, and robustness is taken as a prescribed incremental input-to-output gain. The larger goal is the one the whole constraint family shares: put the guarantee in the coordinate system, not in the loss. The paper sits directly between the group's guaranteed-bounded equilibrium net and the feedforward direct parameterisation, and is the trade-journal counterpart of the conference paper that first introduced the class.

## What they studied

A continuous-state but discrete-time dynamical model `x_{t+1} = f(x_t, u_t, theta)`, `y_t = g(x_t, u_t, theta)`, realised as a feedback interconnection of a *linear* system (an implicit linear layer with matrices `E, F, B1, B2, C1, D11, D12, C2, D21, D22` and a diagonal scaling `Lambda`) with a bank of static slope-restricted activation units. The nonlinearity is never unrolled into depth: the activations form an equilibrium equation `w = sigma(D w + b_w)` that has to be solved at each time step, so a layer of `q` neurons is a fixed point rather than a stack. Unit of analysis: the matrix block `W` of that feedback interconnection, plus the diagonal activation scale `Lambda` and the well-posedness weight `P`. Two model sets are defined: contracting RENs (C-REN), and Lipschitz-bounded RENs (LB-REN) at a user-chosen gain `gamma`.

## How they did it

They define the interconnection and assume the activation is piecewise differentiable and slope-restricted to the unit interval, which covers the rectifier, leaky variants and hard tanh. Rather than *constraining* `W`, they build it from a bank of free matrices (`A`, `l`, `B1`, `B2`, `C2`, `D12`, `D21`, `D22`, `S1`, `S2`): the free `A` is multiplied by its transpose to form a block Gram matrix, small symmetric perturbations `S1`, `S2` are subtracted from its symmetric parts, and `P` is read off its trailing block. `Lambda` is the exponential of the free vector `l`, which is what buys pointwise positivity for free. For the bounded set they re-derive `E`, `B1`, `C1`, `D11`, `P` by adding `1/gamma` times outer products of the input/output matrices, which is the standard way of turning a contraction certificate into a gain certificate by a change of storage function. The proof of each result is an error-dynamics argument: subtract two trajectories, apply the incremental sector constraint as a quadratic form, substitute a Schur complement of the construction's own LMI, and read off a decaying storage function. They then prove the class contains prior model sets as special cases by exhibiting explicit parameter choices — the earlier robust RNN at `D11 = 0`, the bounded equilibrium net at zero state dimension, dense feedforward nets as a strictly-block-lower-triangular activation coupling, Wiener and Hammerstein block models as one-sided couplings, and finite-impulse-response filters by fixing the transition matrix to a shift.

## What counts as an explanation here

A **parameterisation**: a map from an unconstrained coordinate vector to a matrix tuple, together with a theorem that every point of the image has the property. It is exact and total (the map is defined everywhere, and it is a direct parameterisation in the sense that the feasible set is the whole of the parameter space), and it predicts the guaranteed gain of any model it produces without running the model. It is not an explanation of a particular trained model: nothing here describes why a given network behaves as it does, and the certificate's own variables (the multiplier `Lambda`, the well-posedness weight `P`) are never read after training.

## How they judged success

Two system-identification benchmarks with matched parameter counts (F16 ground-vibration data, and a Wiener–Hammerstein system with heavy process noise), fitted by minimising simulation error with Adam. Success is a *trade-off curve*: normalised root-mean-square test error against the *observed* sensitivity of the model, that is, the maximum output difference over input difference found by gradient ascent. They plot the guaranteed upper bound as a vertical line on the same axes, so the question "how far above the true gain is the certificate" is answered directly by where the observed points sit relative to the line. Baselines are a plain RNN and an LSTM.

## What they found

On both benchmarks the REN sits at the best performance-versus-sensitivity trade-off, and the curve moves diagonally (better nominal error, worse sensitivity) as the prescribed gain is loosened. The REN's observed sensitivity was roughly a tenth of the RNN's on F16 and roughly a hundredth on the Wiener–Hammerstein data, at comparable or slightly better nominal error. Most relevant to this survey: at *small* prescribed gain the observed lower bound comes very close to the guaranteed upper bound, so the certificate is nearly tight exactly where the guarantee is doing the most work. The RNN's trajectories on the noisy benchmark are visibly erratic; the bounded model cannot produce that kind of response by construction, which is presented as the point of the exercise rather than as a metric.

## Limits and open problems they name

The activation constraint is the simple incremental sector; the authors remark that richer multiplier classes for repeated nonlinearities do not apply here because the channel-wise activation differences depend on the pre-activation values, which may differ across channels. The bounded set is lossy at the output: no direct feed-through is included, and the paper notes that a feed-through can only be recovered implicitly, by carrying the input in the state. Contraction is a sufficient and strong notion, so the class excludes stable-but-not-contracting systems, and the gain guarantee is for the *sequence-to-sequence* map of an infinite or truncated horizon — it bounds the accumulated effect of input differences, not a per-step sensitivity. Future work is elsewhere entirely: reinforcement learning and observer design.

## What we take from it

- **A framing (central).** The direct parameterisation makes "certificate variables are free parameters" literal: the multiplier is an exponentiated coordinate, the storage weight is a product of the parameter matrix. For our purposes this is the cleanest available statement that in this family the certificate is a *choice of coordinate*, and therefore that a reading of a guaranteed model must start from the coordinates, not from the trained matrix.
- **A warning.** The bench-marked claim is a trade-off *curve*, not a point — the model is only as tight as the prescribed gain, and the tightness improves as the gain tightens. Any table we produce that reports "the certified bound" for a constrained model must state which gain it was trained at, or the number is not comparable to anything.
- **A caution about what is guaranteed.** The guarantee here is a sequence gain with an initial-condition forgetfulness term, which is weaker than a per-input Lipschitz bound on a batch. Importing "Lipschitz bounded" from this paper into a claim about a static network would be a category error, and this is the same shape of condition-in-transit loss the head has recorded repeatedly.
- **A shortcut.** The construction's structure — triangular activation coupling for a feedforward stack, block-diagonal for independent channels, shift for a delay line — is a *graph* on the units, and it is written down explicitly. That graph is a candidate decomposition of the network for our reading, on the same principle the head drew from its per-output clique members.

## Tags

unit: whole-model
form: causal-abstraction
method: training-interpretable
eval: prediction/simulation
goal: control/editing
