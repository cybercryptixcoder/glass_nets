# Large-Sample Theory for the Bergsma-Dassios Sign Covariance

- **Key:** arxiv-1602.04387
- **Authors / organisation:** Preetam Nandy (Seminar for Statistics, ETH Zürich); Luca Weihs and Mathias Drton (Department of Statistics, University of Washington, Seattle); Nandy and Weihs marked as equal contributors
- **Date:** 2016-02-13 (v1, single version). Published as *Large-sample theory for the Bergsma-Dassios sign covariance*, Electron. J. Statist. 10(2):2287-2311, 2016, DOI 10.1214/16-EJS1166 (venue confirmed from the journal record, not from the arXiv page, which prints no journal reference)
- **Link:** https://arxiv.org/abs/1602.04387
- **Kind:** paper
- **Read from:** full text (v1 HTML, ~81,000 cleaned characters: abstract, §§1-6, Appendix A's proofs for §3, Appendix B's proofs for §4, and the reference list). The appendices' later proofs were read for their statements and the structure of the argument rather than line by line; **Figures 1-6 were not inspected as images**, so no number in this entry is digitised from an axis
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The folder has now read the sign covariance three times from the outside: its definition, its generalised family, and its fast evaluator. What none of those papers supplies is the thing a test actually needs — the distribution of the estimated coefficient when the two variables are independent. Computing a t* on a probe read-out gives a number; deciding whether it means anything needs a null. The obvious null is a permutation loop, which costs a full re-evaluation of an order-n^4 statistic per resample. The authors want to replace that loop with arithmetic. Their larger goal is to make the only rank measure in the field that vanishes exactly under independence as cheap to *test* with as tau is, so that ordinal screens become routine rather than expensive, and their stated motivation is precisely the probing workflow: many comparisons at moderate n, each of which currently needs a Monte Carlo p-value.

## What they studied

No model and no learned representation: pairs (X, Y) in R^2 drawn i.i.d. from a continuous, discrete-with-finite-support, or mixed bivariate law. The unit of analysis is the distribution of the sample sign covariance under the null of independence. The objects manipulated are U-statistics and their kernel projections — the statistic rewritten as an average over 4-tuples of a permutation-symmetrised kernel, the kernel's lower-order conditional expectations, and the integral operator those expectations define. Everything is a statement about the law of a statistic at finite n and in the limit; there are no features, directions or activations anywhere in the paper.

## How they did it

They define t* in U-statistic form (the unbiased average over distinct 4-tuples), note that the symmetrised kernel takes exactly three values — two thirds on a concordant quadruple, minus one third on a discordant one, zero when the quadruple is inseparable — and then do the standard U-statistic bookkeeping: write the kernel's one-argument and two-argument projections and their variances. Lemma 3.2 shows the one-argument projection is identically zero under independence, so the usual root-n normal limit degenerates and is useless as a null. Theorem 3.4 then proves the converse for continuous laws with a rectangular support: any dependence at all makes that projection genuinely variable, so degeneracy is not an artefact of the estimator but a signature of independence. Section 4 computes the degenerate limit by finding the spectrum of the two-argument operator. The key structural fact is Lemma 4.1: under independence the projected kernel factorises as a product of a pure-X function and a pure-Y function, so the operator is a tensor product and its eigenvalues are pairwise products of two one-dimensional spectra. For continuous marginals, ranks let them push both coordinates to uniform on the unit square, where the one-dimensional kernel turns out to be the Cramér-von Mises kernel with known eigenvalues. For discrete marginals they build a finite matrix from the marginal CDF and PMF and take its eigenvalues. Simulations at level 0.05 compare the resulting tests against Pearson's t-test, the chi-square test of independence, a two-sample t-test, and a permutation test on distance covariance, and a final section turns the non-degenerate limit into a sample-size formula given a bound on the variance.

## What counts as an explanation here

A closed-form limiting distribution: the null law of n times t* as an infinite (or finite, in the discrete case) weighted sum of centred one-degree-of-freedom chi-squares, with the weights given explicitly in terms of known eigenfunctions or a small matrix. It is a complete explanation of the statistic's null behaviour and of nothing else — no claim about which alternatives are detectable, no characterisation of the alternative distribution, and the paper is explicit that the local-alternative theory is the piece it did not do. Prediction here means "what value would this statistic take on many independent samples", which is exactly the form of prediction a test needs.

## How they judged success

Two criteria, applied separately. For the approximation: kernel-density estimates of the simulated distribution of n t* against the asymptotic density, at sample sizes from 10 to 80 in ten-point steps, over three settings (both marginals normal; both discrete, one uniform on ten values and one geometric; one normal and one discrete on five values), with the asymptotic density obtained by numerical inversion of the characteristic function. For the test: power at nominal level 0.05 against six alternative families, each with its own natural competitor — bivariate normal against Pearson, three discrete mixtures against chi-square, and two discrete-covariate-normal-response settings against a two-sample t-test and against a permutation test on distance covariance. The controls are the classical tests themselves, not ablations; the paper runs no size study of the asymptotic p-value on real data because it has no real data.

## What they found

The continuous null limit is a constant times the Hoeffding-D null limit, and the constant is stated exactly: the weights are the reciprocal products of squared indices scaled by 36 over pi to the fourth. The discrete null is a weighted sum over the product of the two marginal supports, with weights that are eigenvalues of an explicit matrix built from marginals; a Bernoulli pair collapses to a single chi-square. The variance table under independence is exact and worth remembering: zero, one two-hundred-and-twenty-fifth, eight such units, and fifty such units for the four projections. The asymptotic distribution is said to be close already at n of 80 in all three settings. On power, the sign covariance test is competitive with Pearson on bivariate normal and with the two-sample t-test on the discrete-covariate case, loses badly to chi-square on the deliberately non-ordinal discrete alternative, wins substantially on the ordinal discrete one, and beats the distance-covariance permutation test in the six-level covariate case. Sample-size planning is honest about its own weakness: the universally valid upper bound on the variance is very conservative, while a bivariate-normal-specific bound reproduces the simulated requirement closely, and the authors advise over-estimating rather than under-estimating the variance.

## Limits and open problems they name

No local-alternative asymptotics, for a stated engineering reason (weighted non-central chi-squares are awkward to invert numerically, and the authors need densities for power computation). The distributional theory covers continuous, discrete-finite and mixed laws and stops there; whether the "zero only under independence" property itself survives beyond those three families is named as unknown, so the test's consistency claim is inherited rather than extended. Every discrete result depends on marginal probabilities that must be estimated in practice, and the paper says so without quantifying the cost of that substitution. One structural limit worth flagging on our own account: the theory is for a *bivariate* pair, so any screen over a bundle of directions must be applied one direction at a time and corrected for multiplicity outside the paper.

## What we take from it

Three things. First, a drop-in null: with one read-out direction and a label, the sign-covariance test needs no permutation loop, and the null is available in two flavours (a scaled Hoeffding-D limit for continuous scores, an explicit marginal-matrix spectrum for discretised scores). That directly kills the main cost objection to the four-point measures recorded in earlier entries. Second, a diagnostic worth reusing — **degeneracy of the first-order projection is a testable signature of exactly-independence, and the paper proves it cuts both ways in the continuous case**, which is the cleanest justification in this folder for treating a degenerate empirical projection (not the point estimate) as evidence of independence rather than as estimator failure. Third, a reporting template for sample-size claims: a universal variance bound, a distribution-specific bound, and a plot of both against simulation, with the advice to pick the conservative one. The warning to carry: power depends on whether the dependence is ordinal, and the sign covariance loses outright to a contingency-table test when it is not.

## Tags

unit: whole-model
form: probe
method: other
eval: benchmark
goal: evaluation-of-methods
