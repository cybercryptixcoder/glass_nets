# Post-selection inference for causal effects after causal discovery

- **Key:** arxiv-2405.06763
- **Authors / organisation:** Ting-Hsuan Chang, Zijian Guo, Daniel Malinsky; Department of Biostatistics, Columbia University (Chang, Malinsky) and Center for Data Science, Zhejiang University (Guo)
- **Date:** 2025-09-17 (v4; first posted 2024-05-10)
- **Link:** https://arxiv.org/abs/2405.06763
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Constraint-based discovery (here: PC) picks a graph by running many conditional-independence tests, and everyone then estimates an effect from that graph as though it were known. Two things go wrong. First, the same data has been used twice — to choose the model and to estimate inside it — so the reported interval is not valid. Second, and worse for them, if the selected graph is wrong then the thing being estimated is a functional of the *selected* graph, not the causal effect in the true one; a tight interval around the wrong quantity is not a fixable nuisance. Their goal is a confidence interval for the effect in the true graph that remains valid when the graph came from a statistical procedure on the same data. This is a step toward honest end-to-end uncertainty for a pipeline whose first stage is itself a sequence of noisy decisions.

## What they studied

Unobserved unit: the graph — a Markov equivalence class (CPDAG) output by a tiered PC-stable variant, and the DAGs inside it as the free part. Models: a linear-Gaussian SEM, plus a nonparametric arm using a kernel-based conditional-independence statistic. Data: simulated 10-node DAGs on a deliberately dense setting (about seven neighbours per node on average), a sparse companion, and the Sachs single-cell protein data (7,466 measurements, 11 proteins). Query: the average treatment effect of one designated exposure on one designated outcome that is known to be later in time.

## How they did it

Step one repeated the PC run many times, but instead of resampling rows it resampled each *test statistic*: every conditional-association estimate was perturbed by one draw from a normal centred on its sample value, and the adjacency search plus orientation rules were rerun with a shrunken decision threshold. Each run's output was screened — graphs that fail to be proper equivalence classes (cycles, bi-directed edges) are discarded as far from the population CPDAG — and only the survivors were kept. Step two read off, for every DAG inside every surviving class, the back-door estimate and its standard error, built one interval per run as the union over the class, and took the final interval as the union across runs. Partial background knowledge was used in the form of tiers: tests are skipped when the conditioning set contains a variable later than both endpoints, and edges can never point into an earlier tier. The threshold shrink factor was tuned by a simple heuristic — pick the value at which the fraction of runs producing a valid graph is smallest — and reported as calibrated by an explicit error term rather than derived. They compared against a naive interval (treat the selected graph as known), against an oracle interval (true graph known), and against a row-bootstrap variant of the same pipeline.

## What counts as an explanation here

Not an account of the mechanism — an inference object with a stated guarantee: a set-valued answer whose form is "union over the graphs I failed to rule out", plus a stated theorem about what that set covers. It is partial by construction and honest about it: what is claimed is coverage of the true effect at a nominal level in the limit, not identification, and not efficiency. The account predicts a quantity (the effect) under intervention with an attached error, which is exactly the output shape our white-box case needs, but the prediction is conditional on a hypothesis class the method did not itself choose.

## How they judged success

Coverage — the empirical rate at which the constructed interval contained the true effect, over 500 simulated DAGs per setting — against the nominal level, and interval length beside it, compared with the naive and oracle intervals across resample counts and sample sizes. Screens: the fraction of runs kept as valid equivalence classes, used both as a diagnostic and as the tuning objective. Additional controls: a sparse-graph replicate, a truncated resampling variant to check that tail draws were not driving the result, a nonparametric-bootstrap comparison with up to a thousand replicates, and a nonparametric independence-test arm. Theory: two limit theorems (resample-error to zero; coverage at least nominal) and one sufficient condition under which the interval collapses to the oracle's.

## What they found

The naive interval under-covered badly: it stayed below 65% nominal-95% coverage across every sample size tried, at either of the two nominal test levels. The resampled interval reached approximately correct coverage with an appropriate threshold and enough runs, and got closer to the oracle's length as the threshold was tuned. Coverage degraded as the threshold constant grew — a sparser graph can drop the edge that the confounding adjustment needs — while the number of graphs surviving the screen *rose*; the useful range was where the kept fraction was smallest (about 7% in their dense setting). More resamples improved coverage and allowed looser thresholds, at the cost of width and compute; a sparse ground truth made coverage far less sensitive to the threshold. The row bootstrap was strictly worse: a thousand replicates gave 87% coverage, where the proposed method reached 90% at fifty runs. On the protein data, six of a hundred runs survived and oriented into 22 candidate estimates, and the aggregated interval for the effect of interest excluded zero under two different adjustment-set conventions. Under a strong-separation condition the interval provably equals the oracle's; they admit this condition is very strong and call the result mainly of theoretical interest.

## Limits and open problems they name

The threshold constant is a user knob with only a heuristic for choosing it, and coverage as a function of it is monotone in the wrong direction for one of the two diagnostics, so the heuristic is a trade rather than an optimum. The method guarantees validity, not efficiency — the union can be wide, and they explicitly list a bound on inflation as open. Faithfulness and causal sufficiency are assumed; with unmeasured confounding (a PAG-producing procedure) the effect may be unidentified inside some members, and their own suggestion for honest handling is to return the whole real line, which they call trivially valid and probably undesirable. They also note that aggregation over "identified" and "not identified" answers is an unsolved mixing problem, and that extension to score-based algorithms is unclear.

## What we take from it

Three portable pieces. (1) **Resample the test, not the data**: perturbing the statistic rather than the rows gives a valid account ensemble for a constraint-based procedure, avoids the bootstrap's inflated error rates, and needs one order of magnitude fewer replicates. (2) **Screen invalid accounts rather than repair them**: discarding the graphs that fail a structural validity check is what makes the kept fraction a usable diagnostic, and the fraction is a quantity we should print beside every ensemble. (3) **A union is the honest interval for an ensemble, but it buys coverage with width** — so the headline claim must be a pair, coverage and length, each compared to the oracle. Warning for us: their own fairness point — a confidence interval for a functional of the selected model is not a confidence interval for the target, and it is easy to publish the former while saying the latter — applies verbatim to any behavioural claim of ours computed with the account we ourselves selected.

## Tags

unit: circuit | feature | whole-model
form: causal-abstraction | probe | replacement-model
method: decomposition | probing | other
eval: prediction/simulation | faithfulness | benchmark | ablation
goal: understanding | evaluation-of-methods | safety/auditing
