# Computing the Bergsma Dassios sign-covariance

- **Key:** arxiv-1605.08732
- **Authors / organisation:** Yair Heller (independent; gmail address printed, no affiliation line) and Ruth Heller — Department of Statistics and Operations Research, Tel-Aviv University, Israel
- **Date:** 2016-05-27
- **Link:** https://arxiv.org/abs/1605.08732
- **Kind:** paper (short technical note)
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The Bergsma–Dassios sign covariance is an independence measure with the property that it vanishes exactly when two variables are independent, which the classical rank measures do not have. Its obstacle was cost: written as a sum over quadruples of sample points it is a fourth-order statistic, and the naive evaluation enumerates all quadruples. Weihs, Drton and Leung had already reduced this to two factors of n times a logarithmic factor per pair using balanced search trees. This note asks whether the logarithmic factor is necessary at all, and answers no: a single two-dimensional cumulative-count array, built once, lets each pair be scored in constant time. The goal is pure engineering — the same statistic, a cheaper evaluator — with no new theory of dependence offered or needed.

## What they studied

Pairs of ordinal observations (x_i, y_i), i = 1..n. The unit of analysis is a *quadruple* of sample points: the statistic averages a sign kernel over all distinct ordered 4-tuples, where the kernel compares two within-pair gaps against two across-pair gaps. Everything is computed on ranks, so the method is invariant to any strictly increasing change of either coordinate. The object studied is the algorithm, not a model; no networks, no data sets, no empirical application appear.

## How they did it

Three steps. (1) Convert both coordinates to ranks. (2) Build the empirical bivariate cumulative distribution A(r, s) counting observations with both ranks at most (r, s), by a single s-major sweep over an (n+1)×(n+1) grid using the standard inclusion–exclusion recurrence; this costs quadratic time *and* quadratic space. (3) Rewrite the count of concordant quadruples as a sum over pairs (k, l) of a "choose two" term on each of two quadrant counts, and note both quadrant counts are now a single array lookup — the count of points strictly below-left of (min x, min y), and points left of min x but above max y. Each pair therefore costs constant time after the sweep, giving overall quadratic time. The tie-free case reduces the statistic to an affine function of the concordant count alone. The tie case needs the discordant count too, because a quadruple can be neither concordant nor discordant; they order the sample by x, decompose the discordant count into five quadrant-and-equality counts (above, between, below, equal-to-min, equal-to-max) whose product formula they take from Weihs et al., and observe each is again a lookup. The one remaining term — a sum of within-y-value pair counts over the open y-interval — needs a second sweep accumulating running pair counts per row, after which it too is a lookup.

## What counts as an explanation here

Not an explanation of dependence at all — a complexity claim. The form of understanding is an algorithm plus a proof of its cost, i.e. an argument that a quantity previously thought to need logarithmic per-query overhead is actually answerable in constant time given the right precomputation. It is a *replacement* for the previously known evaluator, not for the statistic. Nothing here predicts anything about a model; the predictive content is entirely about run time and memory.

## How they judged success

Success is asymptotic complexity in the standard sense, stated per component and totalled: ranking costs n log n, the cumulative array costs quadratic time and quadratic space, each pair query costs constant time, so the whole statistic is quadratic. The comparison set is explicit — the naive quadruple enumeration (n⁴) and the Weihs et al. red-black-tree algorithm (n² log n), both named in the introduction. There is no timing experiment, no implementation, no simulation, no code link, and no benchmark; the tie-handling is justified by correctness of the counts rather than by measurement.

## What they found

The logarithmic factor is removable. Both the tie-free and tie cases are computable in quadratic time after a quadratic-time, quadratic-space preprocessing step, because every quantity the pair decomposition needs is a range count over the rank plane and range counts are constant-time once the cumulative array exists. In the tie case, only the y-interval within-value term requires a second, differently weighted cumulative array (sums of within-value pair counts rather than point counts). The paper closes by naming what it did not do: the *power* of the measure relative to finer-partition methods is left open, and they suggest it as a comparison worth making.

## Limits and open problems they name

The stated open question is power, not cost — they do not establish that the sign covariance is competitive with methods that examine all partitions of the plane, only that it is cheap. Structural limits the reader can see: the method is quadratic in memory as well as time, so the cumulative array becomes the binding constraint well before the pair loop does at the sample sizes our own batteries run at; the tie machinery is considerably more intricate than the tie-free case and rests on a formula quoted from a prior paper rather than rederived; the note is short and provides no pseudocode, no constants, and no indication of how the grid behaves when one coordinate has very few unique values.

## What we take from it

A **method**, and a cheap one. Three transferable items. First, the technique: any statistic that decomposes into a sum over pairs of *range counts* in the rank plane is quadratic, and the range counts come free from one cumsum sweep along one axis plus a second sweep for the weighted variant — which is exactly the shape of the inexpensive dependence screens this folder has been assembling, and it means a full sign-covariance screen over a battery of read-out directions costs one sort per direction plus a pair loop, no trees and no hashing. Second, a **warning about the memory axis**: a quadratic space term is what actually caps n here, so for our sample sizes the array, not the asymptotics, is the engineering decision; on grid sizes where n approaches a few thousand in the unit axis the array is the dominant allocation. Third, a **framing** worth carrying into how we write our own methods section: "the same estimator, a cheaper evaluator" is a legitimate and cheap contribution, but it licenses nothing about whether the estimator is useful, and this paper is scrupulous about saying so in its last paragraph — the analogue in our work is that a faster probe sweep is not evidence that the probe's statistic detects what we think it detects, and the two claims need separate support. Coreference `arxiv-1708.05653` (the range-tree generalisation whose per-query log factor this note removes in the two-dimensional case, and which states its own crossover timings that this note does not provide) and `arxiv-1501.01732` (the same statistic family at high dimension, where the null constants rather than the kernel are the expensive part).

## Tags

unit: feature
form: probe
method: probing | other
eval: benchmark
goal: understanding | evaluation-of-methods
