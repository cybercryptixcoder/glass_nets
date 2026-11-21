# Conditional Independence Tests for Constraint-Based Causal Discovery: A Survey

- **Key:** arxiv-2608.11156
- **Authors / organisation:** Pavel Averin, Theodoros Moysiadis, Ioannis Katakis; Department of Computer Science, School of Sciences and Engineering, University of Nicosia, Nicosia, Cyprus
- **Date:** 2026-08-11 (v1, single version; published in TMLR 07/2026)
- **Link:** https://arxiv.org/abs/2608.11156
- **Kind:** survey/agenda
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Every constraint-based discovery procedure — PC, FCI, their stable and parallel variants — decides
edge deletions and edge orientations by calling a conditional-independence test, so the quality of the
graph is bounded by the quality of those binary decisions. The survey's complaint is that CI testing is
usually treated as one plumbing detail among many rather than as the organising object of study: reviews
exist for algorithms, datasets and software, but not for the tests themselves, their assumptions, their
calibration, and the way a test-level error turns into a graph-level error. The authors' larger goal is to
let a practitioner pick a test deliberately — from data type, expected conditioning depth, sample budget and
compute — and to let a reader of a published graph audit which decisions stand behind each edge. The
placement (a biomedical framing, published in TMLR) makes it a bridge document rather than a method paper:
nothing new is trained or proved, and the contribution is the cross-family comparison plus the propagation
argument from test error to skeleton and v-structure error.

## What they studied

Six families of CI tests, on observational tabular data: partial-correlation tests (Fisher's Z and rank
variants), contingency-table tests (Pearson chi-square, the G-squared likelihood-ratio statistic), regression-
based tests (including the Generalised Covariance Measure and double-machine-learning cross-fitting
variants), nearest-neighbour conditional-mutual-information estimators, kernel tests (KCI and its randomised
random-Fourier-feature approximations RCIT/RCoT), and machine-learning-based tests up to diffusion-model
conditional samplers. Alongside these, a layer of "robustness wrappers" — shrinkage, permutation nulls, ensemble
partition-and-aggregate p-value combination, heteroskedastic corrections, discretisation-aware tests — which
modify a base test rather than forming a sixth-seventh family. The unit of analysis is the **test call**, and
the survey's stated chain is: a test's assumptions and calibration determine its per-call error rates, its power
decays with the size of the conditioning set, and the per-call errors propagate through skeleton pruning and
orientation rules into spurious edges, missing edges and wrong CPDAGs. No model weights, no learned
representation; the graph-level object is the CPDAG/PAG.

## How they did it

The method is a structured comparison, not an experiment. First, foundations are laid out and named: DAGs and
Bayesian networks, CPDAGs and Markov equivalence, Markov blankets, and the four assumptions a constraint-based
algorithm leans on (Causal Markov, Faithfulness, Causal Sufficiency, and — the load-bearing one here — *correct
statistical decisions from CI tests*). Second, the algorithms are described as callers of tests: the PC
skeleton phase removes an edge on the first conditioning set from the adjacency sets that renders the pair
independent and stores that set as a separation set, orientation turns unshielded triples into v-structures
when the middle node is absent from the stored separation set, and Meek's four rules then propagate as far as
possible; FCI is described with its PAG output and latent-confounder handling; Grow-Shrink is included for
coverage. Third, each CI test family gets the same treatment — the statistic, the null it is calibrated against,
the data types it accepts, the computational complexity of a call, and the failure mode. Fourth, the practical
section makes the propagation argument explicit in four moves: power decays as the conditioning set grows
(partial-correlation statistics lose degrees of freedom as the conditioning size is subtracted from the sample
size, contingency tables thin out and expected counts destabilise), type I and type II errors are asymmetric
in consequence (a false rejection adds a spurious edge; a missed dependence deletes a true edge *and* removes
the separation set that the orientation phase needs), finite-sample behaviour and model misspecification move
both rates, and multiple-testing correction as usually applied is a poor fit because the calls are neither
independent nor known to be positively dependent. Fifth, they tabulate which of seven libraries (bnlearn,
pcalg, MXM, gCastle, pgmpy, causal-learn, Tigramite) implement which family and which PC features (stability,
parallelism, conditioning-set cap, CI-call logging, custom tests), and finish with scalability strategies
(parallel-PC, GPU batching, recursive divide-and-conquer skeleton merging, per-test subsample-and-aggregate
ensembling) and an interpretability section arguing that a parametric test yields a human-readable justification
per edge whereas a kernel or nearest-neighbour decision is hard to audit.

## What counts as an explanation here

A taxonomy plus three decision diagrams and a set of reporting prescriptions. The "explanation" of a learned
edge is an audit trail: the named test, its significance level, the maximum conditioning-set size used, and the
separation sets that justified each deletion and each orientation. The survey is explicit that this is a
*partial* account — it explains which decisions were made and under what assumptions, not why the data had the
structure it did, and it cannot predict a graph from a test choice. `arxiv-2610.13520`'s derivation-graph idea
is the closest thing in the folder to a machine-checkable version of this same object; this survey asks for the
ledger in prose form.

## How they judged success

No metric, no benchmark, no ablation — the survey reports no new measurements and its comparative claims are
assembled from cited studies. The one quantitative-looking artefact is a trade-off table whose cells are
ordinal judgements (strong / moderate / weak distributional assumptions; small / moderate / large sample needs;
weak / moderate / good scalability under large conditioning sets; very fast / moderate / slow runtime), and a
support matrix of ticks for library coverage. The authors are honest that this is where the discipline stops:
they name the absence of a systematic empirical comparison across families — plugging each family into the same
constraint-based algorithm and scoring graph-level outcomes against per-test calibration and power — as an open
problem rather than something they did.

## What they found

The cross-family summary is a bias–variance statement with prices attached. Parametric tests win on
power-per-cost when their assumptions hold, and Fisher's Z is the cheapest call in common use (analytic
p-values, cubic cost in conditioning-set size given a precomputed covariance); contingency-table tests degrade
fast with conditioning dimensionality through table sparsity and an explosion in degrees of freedom; kernel and
nearest-neighbour tests detect much broader dependence classes but need larger samples, cost substantially more
per call, and get variance-inflated exactly where the survey says the trouble is; regression-based and
machine-learning tests are only asymptotically calibrated and, in the GCM's case, the power is inherited from
the quality of the nuisance regressions. Two conclusions carry beyond the survey's own scope. The first is a
hard impossibility imported from Shah and Peters: conditional independence is not testable without restrictions,
because a test with valid type-I control against *every* absolutely continuous null has no power against any
alternative — so every CI test in the table is a bet on a null class, and "nonparametric CI test" means "a bet
that is cheap to state", not "a bet that is free". The second is about multiple testing: the per-test
significance level in PC is a tuning knob, not an error guarantee, and because the calls share variables the
usual corrections do not transfer — the route the survey endorses is false-discovery-rate control (in the
Benjamini–Yekutieli form) or edge-specific p-values, and it notes that most implementations apply neither.

## Limits and open problems they name

Named explicitly: mixed-type data without ad hoc discretisation (the discretisation section concedes the null
being tested is no longer conditional independence in the original space, so discretising changes the question);
small-sample regimes where dozens of observations face thousands of variables; diagnostics that quantify how
badly the Markov/Faithfulness/sufficiency assumptions are violated; estimating **expected power at the level of
the graph** rather than the call, which they say does not exist because the decisions are interdependent and
reused for orientation, leaving only asymptotic guarantees; and the empirical validation of their own decision
diagrams. They also flag LLM-proposed orientations as a soft prior requiring provenance and robustness checks
against hallucinated constraints. A limit they do *not* name is the one that matters most for us: the survey
covers only settings where the experimenter waits for data and then tests it, so the entire question of a *test
whose cost the design must budget for* — the quantity we would want to put in a probe-cost ledger — is outside
its frame.

## What we take from it

A ledger discipline and a warning. Discipline: adopt the survey's audit-trail prescription verbatim as a
minimum standard for every account we publish — for each accepted and each rejected site, name the test, the
calibration used, the conditioning-set size, and the separating set that justified the verdict, because that
tuple is what makes a decision re-checkable after the fact. Warning, and the more important one for us: the
Shah–Peters impossibility is the sentence that our probing batteries have to carry in the abstract, since every
"we showed these two sites are independent" line rests on a restricted null whether we say so or not. Second
warning: type II errors are not symmetric with type I errors in any downstream account-building pipeline, because
a missed dependence does not merely delete one edge — it removes the separating set a later orientation step
depends on, so a battery that tunes its threshold to avoid spurious findings is paying a downstream price that a
single per-call power number will not show. Third, and the one we can act on today: the graph-level power gap
they name is exactly the "expected answer quality per probe" quantity we have been rebuilding by hand in every
entry of the design line, and their diagnosis of why it is hard — decisions reused and interdependent — is the
reason our per-probe utility scores have never summed to a battery score.

## Tags

unit: feature
form: description
method: probing
eval: faithfulness
goal: evaluation-of-methods
