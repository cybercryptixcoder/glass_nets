# Experimental Design for Cost-Aware Learning of Causal Graphs

- **Key:** arxiv-1810.11867
- **Authors / organisation:** Erik M. Lindgren, Murat Kocaoglu, Alexandros G. Dimakis, Sriram Vishwanath; University of Texas at Austin (Lindgren, Dimakis, Vishwanath) and IBM Research (Kocaoglu)
- **Date:** 2018-10-28 (NeurIPS 2018)
- **Link:** https://arxiv.org/abs/1810.11867
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

From observational data alone a causal DAG is pinned down only up to its essential graph, so some edge directions stay ambiguous until somebody runs experiments. Experiments cost money and different variables cost different amounts — the paper's motivating example is an automated scientist whose reagents have wildly different prices, so the same discovery can be made cheaply or expensively depending on which variables you choose to knock out. The authors take the design question in the restricted, non-adaptive setting where all interventions are chosen up front and run in parallel, and ask what the minimum-cost way to do it is and how well a greedy heuristic can perform. Two questions were open when they wrote it: whether the problem is hard, and whether the greedy colouring algorithm inherited from the previous work in this line has any approximation guarantee at all. Their contribution is a hardness proof plus guarantees, not a new learning method.

## What they studied

The unit is the design, not the network: a chordal undirected graph (the undirected component of the essential graph of some causal DAG under causal sufficiency), a positive integer cost per vertex standing for the price of intervening on that variable, and a set of intervention sets. The learning rule is the standard one — an edge gets its direction only if some intervention includes exactly one of its two endpoints. They note that the undirected components of an essential graph are separately learnable, so each component is a self-contained problem. There are no neural models and no estimation here: the whole analysis is combinatorial, over graph colourings and separating systems.

## How they did it

They start from the established equivalence between a set of interventions that orients every undirected edge and a *graph separating system* — a family of vertex subsets whose cuts cover all edges — and from the older result that the smallest such family has size equal to the ceiling of the log of the graph's chromatic number. A separating system is then re-read as a colouring of the vertices by binary vectors of length equal to the number of interventions, where the cost of the design is the sum of the per-vertex costs weighted by how many interventions hit that vertex. That reformulation turns the design question into a weighted colouring problem on a chordal graph. They prove hardness by reduction from a numerical three-dimensional matching problem into this weighted colouring problem on interval graphs, a subclass of chordal graphs the field usually expects to be easy, and the reduction is stated to hold even when every vertex costs one. They then modify the earlier greedy algorithm by adding a *quantization* step: colour a maximum-weight independent set with the free colour first, rescale the remaining vertex weights down to a cubic-size integer grid relative to the largest remaining weight, and then repeatedly colour the maximum-weight independent sets of what is left. To analyse it they fence the problem between two new classes of submodular optimization — a "submodular chain" and a "supermodular chain" — and borrow Wolsey's bound for submodular set cover to control how many colours the greedy run needs, and Nemhauser–Wolsey–Fisher style analysis to control the cost. For the sparse variant they build the design from a minimum (weighted) vertex cover: colour the induced subgraph on the cover optimally, then split its colour classes into pieces of at most the sparsity limit. Experiments follow on randomly generated chordal graphs with degree caps, heavy-tailed cost distributions drawn from a Pareto law, and comparisons against an integer-programming optimum and a naive sort-by-weight baseline.

## What counts as an explanation here

The object produced is a *design*: an explicit family of intervention sets, together with a claim about it. The claim is of two kinds — a multiplicative guarantee relative to the true cheapest design, and a statement about how many interventions it uses. It is a replacement for exhaustive search, not a model of the system; nothing here predicts a network's behaviour, and the "explanation" of why a design is good is the chain of inequalities, not a description. The most reusable piece is a structural one: a lower bound on any sparse design expressed in terms of the graph's minimum vertex cover, which is a floor on the problem rather than on the algorithm.

## How they judged success

Three measures, all relative: approximation ratio of design cost to the optimum, size ratio of the sparse design to the smallest sparse design, and wall-clock runtime. Baselines are an integer-programming optimum solved with a commercial solver and a heuristic that assigns the cheap colour to the heaviest independent set and then sorts the rest by weight. The guarantees are stated conditionally — the cost bound holds only when the allowed number of interventions exceeds the log of the chromatic number by a log-log term, which is the slack that lets the greedy run terminate on the quantized weights. For the sparse problems the guarantee is stated as an additive-over-optimal fractional form that collapses to near-perfect approximation when both the sparsity limit and the maximum degree grow slower than the cube root of the number of vertices, and there is a separate statement that the unweighted case is solved exactly.

## What they found

The minimum-cost design problem is NP-hard, and hard even on interval graphs with unit costs — so the ease of the size-only problem on chordal graphs does not carry over to cost. The quantized greedy algorithm is a two-plus-epsilon approximation with epsilon exponentially small in the intervention count plus an inverse polynomial, provided the allowed number of interventions clears the log-chromatic-count bar with a log-log slack. The sparse variant is near-optimal in size under the cube-root condition, and optimal in the unweighted case. In the experiments the greedy curve sits very close to the integer-programming optimum across vertex counts and graph densities, clearly above the weight-sorting baseline; on ten-thousand-vertex graphs with degree capped at twenty and five interventions the greedy run finished in five seconds against a hundred and twenty-eight for the solver. The trade-off experiment on ten-thousand-vertex graphs with ten-vertex interventions maps penalty strength against cost and intervention count, and their bound is reported as implying a floor of roughly five hundred interventions on average, close to what the algorithm actually spends.

## Limits and open problems they name

The setting is non-adaptive by construction — every intervention is chosen before any data arrives — so nothing here reacts to what an experiment reveals. Causal sufficiency is assumed throughout, so latent confounding is outside the scope of the design statement. Costs are positive integers attached to single variables, not to sets, and the pay-off is stated only in terms of identifying edge directions, with no account of sample size, noise, or how much data an intervention needs before a direction is actually recovered. Vertices of infinite cost are discussed as a special case and work only if they form an independent set; otherwise no finite-cost design exists. The guarantee for the cost bound is conditional on extra interventions being allowed beyond the minimum, which is a real restriction rather than a footnote.

## What we take from it

Three things. First, a *equivalence to exploit*: an intervention set that separates endpoints is the same object as a colouring, so any battery whose job is to distinguish neighbouring sites can be re-scored as a weighted colouring problem, which makes the minimum-cost version of our own probe plans a known-hard problem rather than an embarrassment. Second, a *measure*: the minimum vertex cover of the component is a computable floor on how few interventions any sparse design can use, so a probe plan can be reported as a fraction of that floor instead of as a bare count. Third, a *warning* about quantization: the guarantee only became available after the weights were coarsened, which is exactly the move we would resist if we were treating probe strengths as continuous design variables — coarsening was the thing that made the analysis tractable, and it is worth remembering when a clean theory keeps failing on a fiddly continuous parameterisation. What is absent and matters most for us is the predictive content: this design prices *which* variables to disturb, never *what to conclude* afterwards, and it is strictly non-adaptive.

## Tags
unit: whole-model
form: replacement-model
method: other
eval: benchmark
goal: evaluation-of-methods
