# APP: Accelerated Path Patching with Task-Specific Pruning

- **Key:** arxiv-2511.05442
- **Authors / organisation:** Frauke Andersen, William Rudman, Ruochen Zhang, Carsten Eickhoff (University of Tübingen; The University of Texas at Austin; Brown University)
- **Date:** 2025-11-07
- **Link:** https://arxiv.org/abs/2511.05442
- **Kind:** paper
- **Read from:** full text (arXiv HTML v1; abstract, S1-S7, references, Appendices A-F, Tables 1-4, Figures 1-16; Tables 2, 3 and 5 read cell by cell)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Circuit discovery is expensive because path patching has to run a forward pass per candidate path, so a full sweep of the search space stops being affordable as models grow. Pruning is the cheap alternative - a few forward passes of weight-and-activation statistics - and the authors ask the obvious question directly: if you prune a model hard enough that the survivors keep its task performance, have you found its circuit? Their answer is no, and the paper is built around why. The larger goal is a way to make the expensive causal method affordable on models where a hand-tuned circuit has never been computed, by using the cheap method only as a search-space filter.

## What they studied

Two units are compared on the same footing: the **attention heads included in a circuit** (MLPs excluded by design, and by them specifically here), and the **per-head importance score** a structured pruning method assigns. Models: GPT-2 small, GPT-2 large, Qwen2.5-0.5B and Qwen2.5-7B. Tasks: IOI, Greater-Than, Gendered Pronouns, Induction and Docstring, each as a clean/corrupted minimal pair with a logit-difference metric. The pruning method is FLAP, which scores a head from weight magnitude times activation magnitude; the authors run it in three variants - vanilla (clean data only), "Contrastive-FLAP" (scores computed on the clean-minus-corrupted activation difference), and a merge of the two - and compare all of them against automatic path patching.

## How they did it

Ground-truth circuits come from automatic path patching: start at the final logits as receiver, patch each earlier head in from the corrupted run, keep heads whose influence clears two thresholds (an importance threshold in units of the empirical standard deviation of the influence scores, tightened for senders further from the receiver, and a minimum-magnitude cut), then recurse with the selected heads as new receivers, up to 100 random samples per task; the smallest circuit reaching 75 percent performance is kept, else the best performance-per-size point. Pruning parallel-runs FLAP at sparsities from 0 to 1 in 0.1 steps on 200 samples and keeps the smallest pruned circuit that matches the patched circuit's performance, with the operating sparsity picked automatically from "cliff points" - the first sharp drop, the biggest drop, or a fixed ceiling - above a floor of 0.6. Contrastive-FLAP differs only in substituting the clean-minus-corrupted activation norm into the FLAP score, which is meant to keep heads whose activation is *task-dependent* rather than merely large. APP then unions the vanilla and contrastive pruned sets and runs the same automatic path patching **only inside that union**. Accuracy of the resulting circuit is measured by patching its heads back into a corrupted run and computing the fraction of the clean logit difference restored; agreement with the ground-truth circuit is measured head-wise as recall (true-positive rate) and precision, with GFLOPs and wall clock as cost.

## What counts as an explanation here

A circuit: a set of attention heads, with no edges and no account of composition. It is deliberately partial and the authors say the paper is about discovery, not analysis. The pruning scores are not offered as an explanation at all - the paper is explicit that the union of pruned heads is a *search space*, not a mechanism. What can predict here is the circuit's behaviour under patching: restore its heads to a corrupted run and the logit difference comes back to a stated fraction of the clean value.

## How they judged success

Three quantities, all head-set based: restoration performance (percent of clean logit difference recovered by patching the circuit's heads back in), circuit size and sparsity, and true-positive rate plus precision against the patched circuit treated as ground truth. Efficiency is compared in GFLOPs and compute time. There is no held-out-task control, no random-head control of matched size, and no statistical test anywhere; every number is a point estimate over 100 (patching) or 200 (pruning) sampled prompts.

## What they found

Pruning alone does not recover circuits: when performance is matched, patched circuits are at least 47.62 percent smaller than vanilla-FLAP circuits and at least 4.17 percent smaller than Contrastive-FLAP ones. Vanilla FLAP keeps on average 86.61 percent of the ground-truth heads at an average sparsity of 0.45, Contrastive-FLAP keeps 86.17 percent at 0.68 - nearly identical recall bought at much higher sparsity, which is the paper's case for the contrastive scoring. The union set is what APP patches: it cuts the candidate pool by an average 56 percent, and the resulting speed-up is 4.11-14.87x fewer GFLOPs on the two large models and 2.45-8.33x on the two small ones, with a similar profile in wall clock. The accuracy cost is real but modest on average: APP circuits restore 70.42 percent against 76.00 for plain patching, at sparsity 0.9 against 0.87, i.e. APP tends to return smaller-but-weaker circuits. Per model, the best average recall is 77.53 percent (GPT-2 small) and the best average precision 86.22 percent (GPT-2 large). The per-cell table is far less clean than those averages: on GPT-2 small IOI, APP restores 82.47 percent against 96.95 for patching with 85.71 recall and 94.73 precision; on GPT-2 large IOI it restores 68.33 against 92.60 with 47.66 recall; on Qwen2.5-7B Greater-Than APP reports 79.43 against 79.33 at 62 heads versus 34, and several Qwen2.5-7B patching cells are below 60 percent restored (IOI 61.06, Docstring 58.77). The authors attribute the GPT-2 large shortfall to hyperparameters tuned for the wrong regime. Two results are worth more than the efficiency headline: on Gendered Pronouns, vanilla FLAP produces a 87-head circuit where patching finds 5, Contrastive-FLAP 38 heads at higher restored performance (74.78 against 72.98), and on Docstring a pruned circuit *outperforms* the patched one while having lower recall - which the authors read, correctly, as evidence that recall against one circuit is not a quality metric. Automatic patching itself is not the hand-built circuit either: on GPT-2 small it differs from the manual circuits on 7 of 21 IOI heads, 2 of 8 Greater-Than, 1 of 5 Gendered Pronouns, 9 of 22 Induction and 14 of 36 Docstring heads, and restores less in four of five tasks (e.g. IOI 75.58 against 96.95).

## Limits and open problems they name

Neither automatic patching nor APP reliably lands on the best circuit, and hyperparameter search was itself rationed by patching's cost on the 7B model. Choosing the size-versus-performance trade-off is left as the researcher's judgement even after Pareto analysis. APP's recall is capped by the union set: if pruning drops a needed head, no amount of patching recovers it. Only attention heads are in scope; MLP-bearing tasks such as factual association are out. The pruning heuristic is presented as usable in front of any head-level discovery method (they name ACDC) but only automatic path patching is actually tested. Whether GPT-2 small circuits transfer to the larger models cannot be checked, since no prior patching circuits exist for them.

## What we take from it

Method and warning, mainly. (1) *Search-space restriction is a hypothesis, and its ceiling is measurable*: the union set's recall bounds what APP can find, so the honest way to report a pruned-then-attributed pipeline is to print the filter's own recall beside the end-to-end result - which this paper does, and which our model should do for any candidate-shortlisting step. (2) *Necessity-style scores computed from clean-data activations are not task relevance*: the contrastive version differs from the vanilla one only in subtracting the corrupted activation, and it moves the retained-head quality far more than any threshold sweep in the paper - a direct argument for computing importance on the contrast that defines the behaviour. (3) *Recall and precision against a single reference circuit are weak instruments*: the Docstring case beats its own ground truth and the Gendered Pronouns case shows two very different sizes reaching similar performance, so a per-head F1 against one discovery run measures agreement with that run, not with the mechanism. (4) The automatic-patching thresholds are a graded, position-dependent significance rule (senders further from the receiver must clear a higher bar), which is exactly the kind of pooling choice the folder has repeatedly found decisive; the paper varies K and the magnitude cut on a grid and senses the fragility itself, since one hyperparameter setting turns a 96.95 IOI cell into 75.58.

## Tags

unit: head | circuit
form: graph
method: patching | ablation | decomposition
eval: patching | benchmark
goal: understanding | evaluation-of-methods
