# A Globally Convergent LP and SOCP-based algorithm for Semidefinite Programming

- **Key:** arxiv-2202.12374
- **Authors / organisation:** Biel Roig-Solvas, Mario Sznaier; Northeastern University, Boston
- **Date:** 2022-02-24
- **Link:** https://arxiv.org/abs/2202.12374
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Semidefinite programs are the standard way to certify a gain bound about a network - our whole
constraint thread rests on per-layer matrix inequalities - and their cost is the reason those
guarantees are reported for small nets only. The ticket price of a generic SDP is a cubic in the
matrix dimension per Newton step, which is why the field keeps asking whether the semidefinite
constraint can be swapped for something linear or conic. An earlier line of work replaced the cone
with diagonally-dominant or scaled-diagonally-dominant matrices, which gives an LP or an SOCP, and
repeated the replacement between Cholesky changes of basis; that iteration is fast but was known to
stall on a suboptimal value, and whether a general SDP could be solved through such cones at all was
left open. This paper answers that question affirmatively and then makes the counterintuitive
admission that the resulting algorithm is still slower than the mature interior-point solvers it
replaces, which is the most interesting thing in it for us.

## What they studied

The object is the standard-form SDP: minimise a linear functional over an N-by-N symmetric positive
semidefinite variable subject to M trace equalities. There is no model or dataset in the machine
learning sense; the empirical subjects are the classical SDP library benchmark instances (Lovasz theta
number problems and the max-cut relaxations of graphs with 100 to 250 vertices) plus four randomly
generated problems. The unit of analysis is the whole conic program - its variable dimension N, its
constraint count M, its optimal value and its solve time - not any internal object.

## How they did it

The algorithm alternates two phases. In the decrease phase it fixes a Cholesky factor from the
previous iterate, applies that change of variable to the data matrices, and solves a cone program in
which the semidefinite constraint is replaced by a scaled-diagonally-dominant one; the previous
iterate is always feasible for that problem, so the objective cannot rise. In the centering phase, at
the same objective value, it solves a sequence of barrier problems on the *relaxed* set whose
boundary function is defined blockwise over the pairwise two-by-two submatrices, an analytic centre
computation written as further SOCPs, until the iterate is close to the central path of the original
SDP. The iterate now sits well inside the cone rather than near its boundary, so the next decrease
phase makes real progress instead of stalling on ill-conditioning. The loop repeats until a duality
gap computed from a bracketed central-path parameter falls below target. The convergence proof shows
the combined step multiplies that parameter by a fixed factor greater than one per cycle, so the
number of cycles is logarithmic in the target gap; two further results extend the argument to a
centering phase that stops short of the path exactly and to a computable termination test that only
uses quantities the algorithm already holds. Experiments use one implementation on a laptop-grade
machine, with the decrease step handed to a commercial solver and the centering step hand-coded.

## What counts as an explanation here

Not an explanation of a model in our sense at all - it is an *algorithm with a complexity guarantee*.
What it explains is why a cheaper cone sweep was observed to stall: the iterates accumulate near the
boundary, and no amount of additional decrease steps repairs the conditioning. The deliverable is a
loop structure (decrease-and-center) plus a proof that the loop cannot be trapped, together with the
small list of properties a replacement cone must have for the same proof to go through. It is a
complete argument about its own procedure and says nothing about prediction.

## How they judged success

Three things: a polynomial iteration bound whose constants depend only on problem data (a per-cycle
multiplicative improvement of the central-path parameter, and an allowed centering tolerance derived
in closed form); empirical optimality gap on the benchmark set, reported to a fixed threshold; and a
head-to-head table against the predecessor algorithm on the same instances, both objective value and
wall-clock. The predecessor's known failure mode - returning a strictly better-than-feasible-looking
but suboptimal point - is the falsifiable claim being tested, and it fails on nearly every row.

## What they found

The proposed algorithm reaches the declared optimality gap on every benchmark instance, while its
predecessor misses by visibly wider margins on all the mid-size problems, and by a large margin on
two of the random instances where the predecessor's value is off in sign and size rather than by a
few percent. No number of decrease steps rescues the predecessor, confirming that the deficiency is
structural rather than a tuning matter. But on time the paper is candid: the new algorithm's worst-case
complexity is the same order as an interior-point method, because every basis change costs M matrix
products of the full dimension, and empirically it is far slower than mature solvers on the larger
instances - tens of thousands of seconds against a few thousand for the predecessor on the largest
max-cut problem, and the ordering is not monotone in problem size. The authors also note the opposite
of what the relaxation literature advertises: the cheap-cone programs are fast on their own, yet the
outer loop around them is not, and they suggest their own loop as a blueprint for cheaper bounded-band
cones rather than as a solver.

## Limits and open problems they name

The global convergence needs assumptions (orthogonalised data matrices, strict feasibility, finite
optimum) that must be arranged in advance; the implementation is a proof-of-concept, not competitive
on wall-clock; and the speedup the authors expect is deferred to future work using band-structured
intersections of the cones, which they state have not yet been implemented. The gap between "cheap cone
program" and "cheap solve of the original problem" is left entirely as future work.

## What we take from it

A warning and a measure. The warning: a certificate's cost is not the cost of one conic program but
the cost of the *outer loop* that makes successive relaxations agree with the exact one, and the two
differ by orders of magnitude here - the relaxation is a fast subroutine and the convergence machinery
eats the gain. For us this is the exact accounting we owe whenever we propose replacing an exact
per-layer semidefinite constraint with a cheaper one: the report is (cone passed, exactness of the
final certificate, total solve time of the loop), never the cheap cone's own time. The measure: the
paper's convergence instrument is a duality-gap bracket computed from quantities the algorithm already
carries, which is the kind of certificate-adjacent number our own account can print beside any
guarantee - a distance to the true optimum that coexists with the guarantee rather than replacing it.
Also a shortcut worth remembering: making an optimality bound computable at a *suboptimal* iterate is
what makes early stopping safe, the same move our per-boundary slack numbers make one level up.

## Tags

unit: whole-model
form: program
method: other
eval: benchmark
goal: evaluation-of-methods
