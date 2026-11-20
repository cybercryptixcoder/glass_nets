# Testing independence in high dimensions with sums of rank correlations

- **Key:** arxiv-1501.01732
- **Authors / organisation:** Dennis Leung; Mathias Drton — Department of Statistics, University of Washington, Seattle (addresses and e-mail addresses printed in the v3 HTML; the abs page lists no affiliation)
- **Date:** 2016-12-02 (v3; first submitted 2015-01-08)
- **Link:** https://arxiv.org/abs/1501.01732
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Classical independence tests assume the number of variables is fixed while the sample grows. In settings where the two are comparable or the variables outnumber the rows — a gene panel, a small experiment with hundreds of measured coordinates — the sample covariance is singular and the usual likelihood-ratio route breaks down. The authors want a test of whether *all* m coordinates of a continuous vector are mutually independent, nonparametrically, without assuming normality or even finite moments. Their specific target is the regime where dependence is spread thin over many pairs, so that no single pair stands out; that is what distinguishes their statistics from the well-developed literature on the largest pairwise correlation. The larger goal is a single global verdict on a high-dimensional block, priced as one number, with a way of knowing how much total dependence that number implies.

## What they studied

The unit of analysis is the **pairwise rank correlation** between coordinates (p, q) of an m-dimensional vector, treated as a U-statistic of some degree k, and then a **sum across all m-choose-2 pairs**. Four concrete measures are carried through: Kendall's tau (k = 2), the leading U-statistic term of Spearman's rho (k = 3), Hoeffding's D (k = 5) and the Bergsma–Dassios sign covariance t* (k = 4). The latter two are degree-two degenerate under independence, which turns out to be the whole structural difficulty. Nothing is trained; the data are simulated (noncentral t, and multivariate normal / t / power-exponential elliptical families), and the asymptotic regime is both m and n going to infinity with m/n approaching a positive constant.

## How they did it

First they build a general template: any symmetric bounded kernel on pairs of observations that depends on the data only through ranks yields a per-pair statistic, and the population signal of interest is the sum over pairs of the squared expectations (or of the expectations themselves, for the one-sided version). Three global statistics are then formed from the pairwise numbers: a recentred sum of squares, an *unbiased* estimator of the sum of squared population values built by taking a degree-2k kernel that averages products of two independent k-tuples, and a plain sum. The null distribution is obtained by ordering the variables and conditioning sequentially, which makes each statistic a martingale in the variable index; a martingale central limit theorem (Hall and Heyde) then gives normality, with the only data-dependent quantities being the covariance of two kernel evaluations sharing c indices and, in the degenerate case, a fourth-moment quantity over four tuples arranged in a ring. Those constants are tabulated for the four measures, giving explicit rescaling factors. Power is then studied in a minimax frame: a lower bound on the signal norm needed for any level-α test, a rate-optimality statement under Gaussian equicorrelation for the unbiased statistic, and a negative result showing the max-of-pairwise statistic cannot reach constant power against constant-size equicorrelated signal. Simulations cover size (7 statistics × 4 sample sizes × 8 dimensions, 5,000 draws each) and power (three distributions × three signal strengths × 500 draws, plus a banded-covariance variant and a 5%-contamination variant).

## What counts as an explanation here

Not an explanation of a mechanism — an **asymptotic null distribution**. The deliverable is a closed-form Gaussian limit for each of three global statistics, valid across a family of rank-based pairwise measures, with the scaling constants the practitioner needs. It is partial by design: it says nothing about the alternative distribution, only that the null is centred and normal after the right rescaling, and detection claims are stated as rates rather than as power functions. Nothing here is meant to predict anything about a system; it is an instrument with a calibrated null.

## How they judged success

Size and power, both against explicit targets. Size: Monte Carlo rejection rates at nominal 0.05, 5,000 draws per cell, on noncentral-t data with three degrees of freedom — a heavy-tailed law where Pearson correlations are known to misbehave. Power: 500 draws per cell against elliptical alternatives with total squared Kendall signal fixed at 0.1, 0.3 and 0.7, compared against Schott's Pearson sum-of-squares, the Han–Liu max-rank statistic, the likelihood-ratio test where implementable (m < n), and the Cai–Ma statistic. The theory is judged by whether the normal limit activates at small n, and by whether the rate bounds are attained under the equicorrelation family. Controls are the parametric competitors in the Gaussian cells and the contamination setting, where 5% of the data matrix is replaced by signed draws from a shifted normal.

## What they found

All three statistics are asymptotically standard normal under independence after rescaling, and the rescaling differs in order between the non-degenerate and degenerate cases — a factor of order n/m for the former, n²/m for the latter — which is the paper's central practical point. On size, the rank-based tests converge to 0.05 as n grows (the authors single out 128 as the point where the theory has clearly kicked in), whereas Schott's Pearson sum-of-squares stays visibly anti-conservative throughout the non-Gaussian cells, e.g. 0.115 at n = 128, m = 512. The recentred sum-of-squares using Spearman's rho has good size even at n = 16. On power, the max-of-pairwise statistic collapses against equicorrelated alternatives — as low as 0.020–0.058 in the m = 256, n = 64 cells — while the sum-based statistics reach 0.99+ at higher signal, and the unbiased estimator gives power that is nearly identical across normal, t and power-exponential data, which is the empirical shadow of the rate-optimality theorem. Under 5% contamination the Pearson-based statistic loses the most and the Kendall statistics the least.

## Limits and open problems they name

The power theory covers only Kendall's tau, only the two sum-of-squares statistics, and the general lower bound in Theorem 5.1 is described by the authors as rough. Rate-optimality of the unbiased statistic is proved under Gaussian equicorrelation and *conjectured* for a broader elliptical class; the conjecture is supported by simulation, not by proof. They cannot give a simple relationship between the expectation of t* or D and the scatter matrix of an elliptical law, so signal strength is not interpretable for those two measures the way it is for tau. They flag a Berry–Esseen or Edgeworth refinement as future work, and note that S_r might be minimax optimal for independence under Gaussianity but that this is unproved.

## What we take from it

- **A framing:** a global independence verdict over a block of coordinates is a *sum of many weak pairwise signals*, not a max, and the two are not interchangeable — they are optimal against different norms of the signal vector. Take the norm explicitly: the sum-of-squares targets the Euclidean norm (dependence spread over many pairs), the max targets the sup norm (one strong pair). Report which one you tested.
- **A measure:** for a probe battery over many parallel directions or heads, "is this whole bundle independent of the label" can be answered by a sum of squared rank correlations with a closed-form normal null, no permutation loop, and no distributional assumption. The Kendall version is O(n log n) per pair.
- **A warning about the parametric shortcut:** the Pearson-based version of the same statistic is invalid on heavy-tailed data by a wide margin. Anything we report that sums squared *linear* correlations across a large block inherits that.
- **A method detail worth copying:** the recentring constant for the sum of squares is known in closed form per measure (it decays like n^{-d}, with d the kernel's degeneracy order), so the recentring is arithmetic, not a bootstrap. The paper tabulates it.
- **A shortcut for the unbiased version:** when the estimand is a sum of *squared* expectations, an unbiased estimator of each squared term is the average product over pairs of disjoint index sets, which costs one extra combinatoric factor but removes the square's bias — the same trick that makes the unbiased statistic's power stable across distribution families.

## Tags

unit: feature | direction
form: probe
method: probing | other
eval: benchmark | prediction/simulation
goal: evaluation-of-methods
