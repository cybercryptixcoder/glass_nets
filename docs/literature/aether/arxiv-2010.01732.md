# Lipschitz-Bounded Equilibrium Networks

- **Key:** arxiv-2010.01732
- **Authors / organisation:** Max Revay; Ruigang Wang; Ian R. Manchester — Sydney Institute for Robotics and Intelligent Systems (SIRIS) and Australian Centre for Field Robotics (ACFR), University of Sydney
- **Date:** 2020-10-05
- **Link:** https://arxiv.org/abs/2010.01732
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Equilibrium (implicit) networks define the hidden state as the solution of a fixed-point equation rather than by stacking layers, and this family subsumes ordinary feed-forward and residual nets. Two properties are needed before such a model is usable: the equation must actually have a solution, and small input changes must produce bounded output changes. The authors want both properties guaranteed *by construction during training*, without projection steps or penalty terms that fight the optimiser. Their second and larger motivation is a repair of the literature: earlier Lipschitz certificates for equilibrium nets required strong assumptions on the hidden weight matrix, and they argue those assumptions can be weakened while the activation assumptions are simultaneously made more natural (monotone and slope-restricted). The paper therefore sits at the junction of two families the broader survey keeps meeting separately — the SDP certificate branch off a trained checkpoint, and the parameterisation branch that builds the guarantee into the weights.

## What they studied

Equilibrium networks with a single weight-tied hidden layer of the form z = σ(Wz + Ux + b), with output y = W_0 z + b_y; a scalar elementwise activation satisfying a monotone, slope-restricted-in-[0,1] assumption. Experiments are small: a fully connected MNIST model (80 hidden units, ReLU) across several prescribed Lipschitz constants, and a one-convolutional-layer SVHN model with 40 channels. Comparisons are against an unconstrained equilibrium net, the monotone-operator equilibrium net of Winston and Kolter, and Lipschitz-margin training. The unit of analysis is the whole-model input-output map, with weights as the carrier of the guarantee.

## How they did it

Two matrix conditions do the work. The first, a diagonal-plus-LMI condition on W alone, is equivalent to the existence of a positive diagonal scaling under which W behaves like a contraction in a weighted inner product; the second folds the input and output matrices and a target constant γ into a single matrix inequality. Their key move is to prove that each condition is *both necessary and sufficient* for W to be expressible as a specific closed-form function of unconstrained free variables — an unconstrained square matrix, a real vector exponentiated into the diagonal scaling, and a skew-symmetric term. Training then optimises those free variables with no feasibility machinery at all. To prove solutions exist, they offer two independent arguments: that the fixed point is the unique minimiser of a strongly convex objective built from a convex function whose proximal operator *is* the activation (they prove this correspondence both directions, and tabulate the pairing for ReLU, leaky ReLU, tanh, sigmoid, arctan, softplus), and that the fixed point is the equilibrium of a contracting neural ODE over an auxiliary variable, with contraction established by a Kalman-Yakubovich-Popov argument. Fixed points are computed by Peaceman-Rachford splitting on the fully connected model and by FISTA on the convolutional one; gradients flow by differentiating the implicit solution.

## What counts as an explanation here

Nothing in the explanatory register — this is a *construction*, not an interpretation. The understanding on offer is that the network's sensitivity is fully determined by a small set of certified quantities: a global Lipschitz constant, a diagonal metric, and a positive-definite "distance-to-constraint" matrix whose entries are computed from the weights alone. It is complete for the property it certifies and silent about the mechanism that produces any particular prediction. Notably, no quantity in the paper is attributed to individual units or features; the diagonal scaling is the only per-unit object and it appears only as a training coordinate.

## How they judged success

Nominal test error alongside an *observed* lower bound on the Lipschitz constant obtained by an L2 gradient-based attack, and an *approximation ratio* between that observed lower bound and the certified upper bound, reported per model. Robustness is test error at fixed perturbation radii (0, ≤5, ≤10 in L2 norm). Cost is measured as seconds per training epoch against the equilibrium-network baselines. Ablations run along the prescribed γ (from unbounded down to 0.2), against four Lipschitz-margin settings and one imported Lip-NN number from a prior paper.

## What they found

The certified and observed Lipschitz constants track each other closely on the fully connected MNIST models — the reported approximation ratios run from about 58% at γ=5 up to about 93% at γ=0.4, with most of the tight settings near 86–93%. Pushing γ down degrades robustness sharply at the small end: the γ=0.2 model has 7.41% nominal error and 42.6% error under radius-10 perturbation, against 1.81% and 95.4% for γ=5, so the trade curve is strongly non-monotone in the robustness column and the authors report only the gentler part of it as a win. The unconstrained equilibrium net has an observed constant around 239 with no accuracy advantage, and the monotone-operator baseline lands near 7.75 at comparable accuracy to the γ=5 model. Cost is the headline: epochs take 14.4 s for the baseline, 16.1 s unconstrained, 14.9 s and 14.8 s for the two bounded variants — essentially free, in explicit contrast to an earlier ADMM-based constrained training they cite as roughly fifty times slower. On SVHN the convolutional version cuts the observed sensitivity by more than an order of magnitude at a few points of nominal error, but the approximation ratio collapses to 40%, which the authors attribute to a deliberately restricted block-constant multiplier.

## Limits and open problems they name

Convolutional support is thin: one layer, one dataset, and a metric restricted to a block-constant pattern for tractability, which is also their own explanation for why the SVHN certificate is loose. Larger and more richly structured convolutional/deep architectures are deferred entirely. The authors note FISTA iteration counts rise during training because the norm governing its step size grows, and they take a certified Lip-NN figure on trust from another paper rather than reproducing it. The closing section lists multivariable activations, more general quadratic constraints, and applications such as quadratic-program argmins as future work.

## What we take from it

Three things. First, a **reusable instrument**: the pair (observed lower bound from an attack, certified upper bound from an LMI) reported as a ratio is exactly the interval discipline our own account needs, and this paper is one of the few in the head that prints it — its honest collapse to 40% on the convolutional case is the warning that a certificate's tightness is not portable across layer types even within one paper. Second, a **framing**: a fixed point of a weight-tied map is a much better host for a per-unit description than a deep stack, because the state has one coordinate system for all units and the sensitivity of the map is carried by a single matrix; if we ever want to attribute a whole-network gain to specific units, the implicit formulation removes the "which layer" problem at the outset. Third, a **warning about method choice**: their earlier-modern baseline note that a full peaked multiplier class claimed for repeated nonlinearities is genuinely invalid for *incremental* slope conditions is directly relevant to any account that reasons about pairwise unit coupling, since the counterexample rests on the increments depending on the operating point, not just the slope bounds.

## Tags

unit: whole-model | form: replacement-model | method: training-interpretable | eval: prediction/simulation | goal: safety/auditing
