# Construction Methods for Gaussoids

- **Key:** arxiv-1902.11260
- **Authors / organisation:** Tobias Boege, Thomas Kahle both at OvGU Magdeburg, Germany (per-author addresses at the end of the text, with emails)
- **Date:** 2019-12-11 (v2; v1 2019-02-28)
- **Link:** https://arxiv.org/abs/1902.11260
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why
Gaussoids are the combinatorial shadow of conditional independence among jointly Gaussian variables: sets of CI
statements, in the sense of a symbol (ij|K) meaning "i and j are independent given K", closed under the four
Lněnička–Matúš axioms. The authors' question is a counting and construction question. How many gaussoids are
there on n variables, can every gaussoid be assembled from its small pieces, and what happens if only some kinds
of small piece are allowed. The larger goal is the same one that motivates matroid theory: to know the space of
possible independence structures well enough to say whether a given structure is realizable, and to have a
supply of structures for which it is not. A secondary motivation is explicitly negative: a finite forbidden-minor
characterization of *realizability* cannot exist, so the useful objects become the axiomatic classes and their
growth rates.

## What they studied
Combinatorial objects, not models: n-gaussoids for small and general n, viewed as subsets of the 2-faces (squares)
of the n-cube. The unit of analysis is the triple (i,j|K) read as a square of the cube — i and j are the two axes
the face extends along, K marks which other axes are pinned to the "conditioned" side. Also studied: 3-gaussoids,
of which there are exactly eleven, falling into five symmetry classes under the permutation group on three letters,
named E (empty), L and U (the two kinds of singleton), B (belt) and F (full); and 26 "special" classes generated
by allowing only some of these five as allowable 3-face pieces.

## How they did it
Three moves, in order. First, a rewriting of the axioms as a constraint on every 3-face of the n-cube: if a
gaussoid contains a "knee" (two squares of a 3-cube sharing an edge) it must contain the unique belt through it,
and if it contains two opposing squares it must contain one of the two belts through them. Second, a locality
lemma: a set of squares is an n-gaussoid exactly when all of its 3-minors are 3-gaussoids, so building a gaussoid
is a jigsaw puzzle whose pieces are 3-gaussoids and whose conflicts come from pieces sharing a square. Third,
they encode those conflicts in a family of transitive graphs Q(n,k,p,q) whose vertices are the k-faces of the cube
and whose edges join faces that both intersect some p-face in at least q dimensions; the degree is computed in
closed form and the graphs are shown to be complete exactly at a stated inequality between the parameters. An
independent set in such a graph is a collection of faces far enough apart to be puzzled freely, so Brooks' theorem
on the chromatic number of a regular graph yields a large independent set, which yields lower bounds by lifting
arbitrary 3-gaussoid choices into disjoint faces, and an injective perturbation argument yields matching upper
bounds. The special classes are then obtained by taking each of the 32 subsets of the five symmetry classes as an
allowed basis, writing the resulting axioms on the 3-cube, and enumerating or counting the solutions with SAT
solvers.

## What counts as an explanation here
A classification theorem, not a model of anything. The explanation of a gaussoid is its decomposition into 3-face
pieces plus the statement that no other constraint exists (the "puzzle property"), and the explanation of a
special class is a bijection to a familiar combinatorial object — undirected graphs, graphs whose complement has no
triangle, forests of paths, partitions, involutions, subsets. These bijections are exact and complete within their
class; nothing is probabilistic and nothing is fitted. The paper also delivers an explicit count of the eleven
3-gaussoids, which is the complete list of local constraints, so an n-gaussoid's anatomy is fully described by
n+3 pieces per face and nothing else.

## How they judged success
Proofs plus exhaustive enumeration. The lower bound on the number of n-gaussoids is a counting inequality
(Proposition 3.9), the upper bound a perturbative injection into non-gaussoids (Proposition 3.10), both fed by
Brooks' theorem with the exact degree from Corollary 3.7. For the special classes the ground truth is machine
computation: SAT solvers (sharpSAT for counting, an all-solution solver for enumeration) over the axioms, with
Sage used to build the input files; nine of the resulting integer sequences were matched against OEIS, which is the
external check that the axioms and the solver agree with something independently known. An independence-number
argument shows the n·2^n order of the lower bound is the best this construction can give.

## What they found
The number of n-gaussoids is doubly exponential, and the bounds are tight enough to locate it: for n at least 5 the
log of the count lies between a constant times n·2^n and a constant times n²·2^n. The 3-gaussoids number exactly
eleven, in five symmetry classes, which is a complete and very small local basis for all the closure structure.
Not every class grows fast: classes whose basis omits the empty and the two singleton types are incompatible —
their members vanish or collapse to the empty or full gaussoid above a small dimension (the class LUB has counts
9, 111, then 0, 0; LU vanishes at n=5). The graphical classes land on familiar objects with known counts: ascending
gaussoids are in bijection with undirected simple graphs (2^{n choose 2} of them), EUB with graphs whose complement
is triangle-free, UBF with forests of paths, EUF with partitions, dense BF with involutions, and EB is determined by
a single binary vector, giving 2^{n-1}. One class resists all of this — LUBF, the "exceptional" one, with counts
10, 142, 1,166, 12,796, 183,772, 3,221,660 for n = 3 to 8, every one divisible by 42 and from n=5 by 210, with a
conjecture of single-exponential growth and a stated open challenge to find a forbidden-minor characterization for
the non-orientable members. Two results are carried over rather than proved: graphical gaussoids are all
realizable, and realizable gaussoids are only single-exponential in number because a positive-definite covariance
turns the gaussoid into a vector matroid on a 2n-element ground set (Nelson's result that almost all matroids are
non-representable does the work). Remark 3.11 states exactly which two properties the doubly-exponential
construction needs — a puzzle property and a basis containing the empty set and all singletons — and notes that
realizable gaussoids lack the first and graphical ones lack the second.

## Limits and open problems they name
Realizability is characterized only for four variables; the five-variable case is open and no general forbidden-minor
characterization of realizability can exist (citing Šimeček and Sullivant). The upper-bound construction is shown to
be optimal for the graph they use, so the n²·2^n gap between the two constants is not closed by this technique.
The exceptional class LUBF has no interpretation, no OEIS match after trying arithmetic transformations of the
counts, and an unknown growth rate (Challenge 4.10, Conjecture 4.11). Beyond combinatorics they flag that
orientability of the graphical classes is understood but the properties of the tamer classes are not, and that
gaussoids are known to coincide with singleton-transitive compositional graphoids, a fact they use for positioning
rather than develop.

## What we take from it
Three things. (1) A guaranteed local basis, which is the cheapest possible test of whether a probe's independence
relation is even gaussoid-shaped: check the eleven 3-face patterns and nothing else — if every 3-variable
restriction of a measured CI set is one of the eleven, the whole set satisfies every axiom by Lemma 3.3, with no
need to test axioms over large conditioning sets. That is a finite, exhaustive, model-free validity check for the
kind of CI structure entries arxiv-2206.05829 and arxiv-2310.17611 assume. (2) A counting warning with numbers: the
axioms admit a doubly exponential family while the realizable ones are merely single exponential, and the special
(graphical) classes are smaller still and land on partitions, involutions and forests — so a set of CI claims that
happens to be graphical is a very strong extra assumption, not a neutral simplification, and a dictionary-derived
independence relation should be reported with which special class, if any, it falls into. (3) A method warning
about building structure from local pieces: the 4-cycle example shows that the class of a gaussoid cannot be read
off the induced subgraphs of its graph — all observable 3-minors are U while the true class is EU — so assembly
from overlapping pieces needs the conflict structure (the graph Q) and not just consistent pieces.

## Tags
unit: feature
form: causal-abstraction
method: decomposition
eval: benchmark
goal: understanding
