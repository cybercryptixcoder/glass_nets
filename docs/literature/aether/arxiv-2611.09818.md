# Generative Latent Priors for Interpreting and Steering Protein Language Models

- **Key:** arxiv-2611.09818
- **Authors / organisation:** Daniel Hinjos (Barcelona Supercomputing Center; Universitat Politècnica de Catalunya), Lucía Urcelay (Centre for Genomic Regulation); equal contribution; Interp4Discovery workshop at NeurIPS 2026
- **Date:** 2026-11-06
- **Link:** https://arxiv.org/abs/2611.09818
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

The mechanistic-interpretability toolkit for protein language models is dominated by sparse dictionary learning, which buys its interpretable units by assuming the activation space decomposes linearly and sparsely. The authors ask what the same activations look like when nothing is assumed: they port a Generative Latent Prior — a diffusion (flow-matching) model trained on a frozen host's activations, learning their distribution directly — from language models to ESM-2, and evaluate it in the two roles an interpretability tool has to play: reading the representation, and editing it. The larger goal is to find out whether an unconstrained generative model of activations gives a better lens than a dictionary, and whether "on-manifold" editing can turn a crude steering direction into a usable control instrument for protein design. They are also explicit that both roles need to be judged on outcomes the field usually does not report — folding the sequences that come out.

## What they studied

Two frozen hosts, ESM-2-8M (d = 320, layer 4) and ESM-2-650M (d = 1280, layer 18) — exactly the layers InterPLM released SAEs for, so the two representations can be compared on the same tensor. Training activations come from UniRef50 sequences (5M random proteins; 100M activations at both hosts, 500M at 8M). Units of analysis: the diffusion model's own gated SwiGLU blocks read at a fixed diffusion time ("meta-neurons", one scalar per unit), the SAE dictionary's 10,240 features, ESM's feedforward neurons, and raw residual-stream dimensions. Concepts: 110 discrete residue-level Swiss-Prot annotations and 10 continuous whole-sequence physicochemical properties, both frozen before any probing.

## How they did it

A flow-matching network is trained to predict the velocity from a partly noised activation back to the real one, so integrating it from noise generates an activation and integrating it from a partly corrupted one projects that activation back onto the learned manifold. Generation quality is measured as Fréchet distance to held-out real activations at 1,000 sampler steps, divided by the real-versus-real floor at the same sample size so the number is comparable across hosts. For the lens: one unit at a time is probed, 512 candidates shortlisted by the positive-minus-negative activation gap on four of eight protein shards, each fit as a one-dimensional logistic probe (or OLS for continuous targets) with inner 5-fold selection, then scored on the other four shards — with every evaluation accession removed from the GLP's training pool because Swiss-Prot overlaps UniRef50 by about 98%. Two falsification tests follow on each concept GLP wins: regress the winning meta-neuron on the full SAE dictionary, and find the smallest joint probe over the SAE's top-k features that matches its AUC. For the edit: DiffMean directions for hydrophobicity, charge and instability are normalised to the mean activation norm and added at the masked positions of a masked-Gibbs decoding loop, with the activation either left edited or re-noised and denoised back onto the manifold before the head sees it; 40 starting proteins per direction, seven strengths, 1,120 generations, then ESMFold on the outputs. A second edit type clamps one SAE latent toward four structural concepts. Effect is reported as a divergence — up minus down — which cancels whatever the intervention does to both arms; the corrected arm's own drift is measured in a zero-strength run.

## What counts as an explanation here

A learned, unconstrained generative model of the activation distribution — a manifold — plus single units of that model as candidate features. Interpretation is read in two forms: a unit that isolates a concept better than any unit of a competing representation, and a navigable manifold that lets an edit be applied and then repaired. It is deliberately partial and non-causal: the manifold says what activations are plausible, not what the host computes, and the paper's own falsification tests are designed to show that a single unit's advantage need not mean new information.

## How they judged success

For the manifold: diffusion loss against the trivial-predictor floor, Fréchet distance at fixed step count, its ratio to the real-versus-real floor, and the collapse of eight depth/data runs onto one compute curve. For the lens: mean held-out test AUC over 110 concepts and mean Spearman ρ over 10 properties, each GLP-versus-baseline margin tested as a matched pair with a 10,000-resample paired bootstrap and a two-sided paired Wilcoxon, plus per-concept win counts. For the edit: the up-minus-down divergence against each property's UniRef50 spread (unpaired Mann–Whitney, n = 40 per direction), a zero-strength control isolating the operator's own bias, GLP diffusion loss on the steered activation before and after correction, ESMFold pTM/pLDDT on the finished sequences, DSSP/pepcoil/PROSITE detection for structural motifs, and two alternative decoders as a budget-controlled comparison.

## What they found

The manifold itself works: loss falls below the trivial floor, all eight 8M runs collapse onto one curve in 6ND, and Fréchet distance improves monotonically with depth (FD ratio 4.47× at 3 blocks down to 2.42× at 24 blocks, 500M scale), reproducing on the 650M host.

As a lens, single meta-neurons beat every baseline on discrete concepts at both hosts: mean AUC 0.903 [0.887, 0.918] against SAE 0.845 and residual stream 0.833 at 8M, and 0.932 [0.917, 0.946] against SAE 0.872 and residual stream 0.872 at 650M, winning 104 of 110 concepts against the SAE and 105 of 110 at the larger host. On the 10 continuous properties the picture reverses at scale: 0.669 at 8M (best in column) but 0.555 [0.498, 0.613] at 650M against the SAE's 0.569, intervals overlapping.

The falsification tests then attribute that lead to concentration rather than content. Each winning meta-neuron is largely a linear read-out of the SAE dictionary (median held-out R² 0.756 at 8M, 0.725 at 650M), and a joint probe over at most 20 SAE features reaches its AUC for 94 of 104 concepts at 8M and 101 of 105 at 650M; where it does not, the shortfall is a near miss (median 0.0035 and 0.0041) and the largest gaps fall on different concepts at the two hosts, so the authors do not read them as concepts the dictionary cannot represent.

As an editing prior, the correction works in isolation: steering raises the GLP loss on the activation from 0.869 to 1.743 at the top of the grid and one SDEdit pass returns it to 0.944, recovering over 90% of the excess — but the property gains are what it buys. Hydrophobicity divergence is +1.400 naive at α = 0.642 and +3.382 corrected; at mid-grid α = 0.128 the naive arm is insignificant (+0.112, p = 0.23) while the corrected arm is +0.969.

Two failures follow. The correction is itself directional: at zero steering strength it raises GRAVY from both the hydrophilic and hydrophobic starting sets (+1.145 and +0.536) — a fixed bias of +1.77 SD — driven by leucine and valine up, charged residues down. And the decoding loop imposes a foldability fixed point: with no steering at all, six starting sets spanning pTM 0.369-0.650 all land between 0.227 and 0.315 (loses 27-55%). Rewriting coverage, not steering strength or the correction, is the cause: at matched expected coverage, few rounds of many positions and many rounds of few give the same result, foldability holds up to roughly 15% coverage and declines beyond it. A stochastic beam-search decoder trades the other way — pTM 0.409 at 50 edits against 0.293 for masked-Gibbs, with divergence +0.51 against +3.99 — so both decoders sit on one efficacy/naturalness curve parameterised by edit count. Structural clamping is concept-dependent: only Helix rises consistently (DSSP fraction 0.54 -> 0.77 from the correction alone at c = 0), Beta strand barely moves and falls under correction, a Coiled-coil "gain" turns out to be single helices scored as heptad repeats, and Zinc finger never forms a complete C2H2 motif — which the authors explain structurally: a per-position bias can set local propensity but cannot pair residues that are far apart in sequence.

## Limits and open problems they name

One hook layer per host; correlational probing on a domain-heavy annotation pool; steering at one layer and only at masked positions. They note the AUC is not metric-invariant with InterPLM's domain-adjusted F1, so the ordering could differ under another metric. The corrected arm's candidate pool is larger than the others because GLP units are read at three diffusion times. Naturalness is blamed on decoding and leaves open better decoders, finer-grained guided coverage, a host whose output stays continuous (so no re-embedding), or clamping inside the sampler so the result is on-manifold by construction. Automated interpretation of what a meta-neuron encodes beyond a curated annotation is called future work.

## What we take from it

Three things, in order of usefulness for a predictive white-box model.

(1) **The falsification pair as the way to report a single-unit advantage.** Asking (i) whether the winner is linearly recoverable from the competing dictionary and (ii) how many of its units are needed to match — and then reporting "concentrated, not new information" when both say yes — is the cleanest published shape of the check our entries keep recommending. It is a *re-representation* test: the strongest available alternative basis must fail to reproduce the win.

(2) **A learned generative model of activations as the manifold term the folder has been missing.** Every steering result in this folder assumes "off-distribution" is bad without measuring it; here the diffusion loss on the steered activation is a printed number before (0.869) and after (1.743) the edit, the correction's recovery is quantified (0.944, over 90% of the excess), and the paper then shows the recovery evaporates on re-embedding the decoded sequence. That pairing — repair works on the tensor, fails on the output — is the general warning: a manifold correction applied to an intermediate state is only as good as the map from state back to the thing being evaluated.

(3) **The intervention's own bias as a mandatory zero-strength arm.** The α = 0 run is what turns "our correction improved the property" into "our correction adds hydrophobicity everywhere it starts". Cheap, and it is exactly the arm run 918's displaced-share argument wanted next.

Also worth copying: the up-minus-down divergence (cancels artifacts that push both arms the same way), the coverage-parameterised efficacy/naturalness curve (mask fraction and round count act only through their product — a warning that a two-knob grid can look like a two-factor effect), and the refusal to credit a detector because it is the best controller (the strongest Zinc-finger F1, 0.907, never forms the motif).

## Tags

unit: feature | weights
form: dictionary | replacement-model | steering
method: SAE | probing | steering | weight-analysis | ablation
eval: ablation | prediction/simulation | downstream-task | faithfulness
goal: understanding | control/editing | evaluation-of-methods
