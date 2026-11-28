# Catastrophic Failure of LLM Unlearning via Quantization

- **Key:** arxiv-2410.16454
- **Authors / organisation:** Zhiwei Zhang, Fali Wang, Xiaomin Li, Zongyu Wu, Xianfeng Tang, Hui Liu, Qi He, Wenpeng Yin, Suhang Wang; The Pennsylvania State University, Harvard University, Amazon
- **Date:** 2024-10-21 (v1 only)
- **Link:** https://arxiv.org/abs/2410.16454
- **Kind:** paper
- **Read from:** full text (abstract, sections 1-7, full reference list, Appendices A-H with Tables 1-10)
- **Relevance to us:** 3 (central)

## What they wanted, and why

Unlearning a harmful or private passage from a language model is supposed to make it unrecoverable, but the benchmarks in use score the full-precision checkpoint's outputs and stop there. The authors noticed that deployed models are almost always compressed before release, and asked whether the compression can undo the deletion. The failure they find is not a corner case of one method; they argue it is a structural consequence of what "good" unlearning looks like today — a small, utility-preserving edit — and therefore affects the whole family. The paper is aimed at benchmarking practice as much as at methods: they propose a third objective that unlearning should satisfy, and a way to measure it.

## What they studied

Six unlearning variants built from two families (gradient ascent and negative preference optimization) crossed with two utility regularisers (gradient descent on the retain set, KL minimisation to the target model on the retain set). The unit of analysis is the **whole model's weight matrices under a rounding map** — no neurons, features or circuits. Data is the MUSE benchmark's NEWS (BBC articles) and BOOKS (Harry Potter novels as forget, FanWiki as retain) splits, plus RWKU for a replication. Model sizes are not named in the text read; the setting follows MUSE's protocol (AdamW, learning rate 1e-5, 10 epochs on NEWS and 5 on BOOKS, regularisation weight grid-searched from {2, 5, 10, 100, 300}).

## How they did it

Each method is trained at full precision, then quantised post-hoc with round-to-nearest at 8-bit and 4-bit, and both checkpoints are scored with MUSE's four metrics: verbatim memorisation (ROUGE between the model's continuation and the true next tokens), knowledge memorisation on the forget set (ROUGE against GPT-4-generated QA pairs), privacy leakage (Min-K% membership-inference AUC relative to a retrained model, so zero is the target), and utility on the retain set (the same knowledge metric). 2-bit is excluded because it breaks utility outright. They then repeat the 4-bit point with two calibration-based quantisers, GPTQ and AWQ, to ask whether the choice of rounding scheme matters — the calibration corpus is generic rather than drawn from the forget domain. The explanation step is analytic rather than experimental: with step size Δ = max|w| / (2^(N-1)) a weight in the bin around zero maps to exactly zero, and any update smaller than the bin width vanishes, so the quantised unlearned model and the quantised target collapse together; their worked example with max|w| = 200 gives a 25-wide int-4 bin against an int-8 bin about sixty-four times narrower. The method they propose, SURE, responds by raising the learning rate to 1e-4 but masking which modules may move: a module-level saliency score (Frobenius norm of the forgetting-loss gradient per attention or feed-forward module) is thresholded at a percentile γ, and only modules above it are updated.

## What counts as an explanation here

A **mechanistic argument about the quantiser plus a fix that follows from it**. The claim is not "unlearning is fake" but "unlearning edits are smaller than the resolution of the format they will be shipped in", which is why 8-bit is mostly safe and 4-bit is not. The proposed repair is a hybrid: a weight-saliency mask borrowed from the localisation literature, applied at module rather than weight granularity, so large-step updates land only where the forgetting gradient says they must. It is a partial account — nothing about where the knowledge lives or what the edited weights compute — but it does make a falsifiable prediction (masked large-step edits should survive 4-bit rounding, full large-step edits should lose utility) which the appendix tests.

## How they judged success

MUSE's four metrics, each computed on both the full-precision and the quantised checkpoint; the recovery reported as the retained fraction of intended-forgotten knowledge rather than as a raw score. Utility is checked not only on the retain set but on MMLU 5-shot, TruthfulQA MC1 6-shot, TriviaQA 6-shot F1 and an AlpacaEval-based n-gram entropy fluency score, so the trade-off shows up outside the benchmark's own domain. Ablations: precision (8-bit vs 4-bit), quantiser (RTN vs GPTQ vs AWQ), the saliency threshold γ over the 50th, 90th, 95th and 99th percentiles, and removing the mask entirely (SURE/S) at a large learning rate. RWKU adds fill-in-the-blank, QA and adversarial-query probes.

## What they found

At 4-bit, the restore is large and consistent. GA_KLR on BOOKS goes from 13% forbidden-knowledge retention at full precision to roughly 89% after quantisation; the abstract's aggregate is 21% average retention at full precision rising to 83% at 4-bit for the utility-constrained methods; the untethered GA and NPO, which destroy utility, show no recovery — which the authors read as the recovery being tied to the very regularisers that make unlearning deployable. In Table 1, several constrained arms go from single digits or near-zero knowledge memorisation to scores near the target model's: NPO_GDR on NEWS moves from 46.1 to 51.4 against a target of 63.9 and a retrain of 33.1, and its privacy-leakage score moves from 107.2 to -99.8. 8-bit leaves the metrics essentially unchanged across the board. GPTQ and AWQ land where RTN does, so the failure is a property of the bit budget, not of the rounding heuristic. SURE recovers most of the loss: GA_GDR + SURE + 4-bit on BOOKS reads 4.8 knowledge memorisation and -6.3 leakage against 33.7 and -35.2 without it, with utility close to the full-precision method. The price is visible in Table 3, where GA_GDR + SURE drops truthfulness to 0.2 and factuality to 0.0, and RWKU shows the same pattern with GA full-precision at 27.6 QA against 49.3 after quantisation.

## Limits and open problems they name

The theory is a bin-width argument, not a measurement — no experiment reports how many weights actually changed bins between the target and unlearned model, so the mechanism is inferred from the pattern of results rather than counted. SURE's mask threshold is percentile-based and the percentile is tuned per dataset (90 to 99 across methods), which is a free parameter dressed as a convention. The new objective they propose is checked on two benchmarks plus a replication; no model size, seed, or repetition is printed in the text read, so the cells are single runs. SURE's utility losses are concentrated in truthfulness and factuality, and the authors attribute them to the large step and acknowledge the trade-off is unresolved. Lower precision than 4-bit and quantisation-aware training are untouched.

## What we take from it

A **measurement warning with a mechanism we can copy**: an internal state that survives a readout at one resolution can be destroyed by the same readout at a coarser one, so a white-box claim about what a model "contains" is only valid at a stated resolution, and the resolution is part of the claim. For our own predictive model this rules out treating float weights as the ground object without saying what happens when they are rounded, and it supplies a cheap audit — re-run every probe after quantising and report both numbers in one row. Second, the **module-saliency mask** is a useful instrument in its own right: a per-module gradient norm under the target loss, thresholded at a percentile, is a coarser and more robust version of the per-neuron relevance maps the folder keeps finding fragile, and it is exactly the kind of object a white-box model can predict and then have checked. Third, the appendix's observation that GDR and KLR push the privacy-leakage metric in opposite directions (GDR drives it strongly positive, KLR strongly negative) is a reminder that a single "closer to zero is better" score cannot distinguish under- from over-unlearning.

## Tags

unit: weights
form: replacement-model
method: weight-analysis
eval: ablation
goal: safety/auditing
