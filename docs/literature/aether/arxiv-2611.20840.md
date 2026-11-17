# Interpretable deep concepts of organoid morphology

- **Key:** arxiv-2611.20840
- **Authors / organisation:** Lion J. Gleiter, Tingying Peng (Helmholtz Munich / Helmholtz AI; Technical University of Munich, School of CIT)
- **Date:** 2026-11-16
- **Link:** https://arxiv.org/abs/2611.20840
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Deep networks classify brightfield microscopy images of organoids well enough to predict drug response, viability or differentiation, but the cues they use are hidden, so a researcher cannot tell whether a classifier is reading the organoid or a batch effect. Hand-crafted morphometric features are readable but weaker. The authors want the readable part of the deep space back: a dictionary of named visual concepts extracted from frozen foundation models, so that a downstream decision can be described in the same terms a biologist would use. They also want to test whether such concepts are merely decorative or can actually be *used* — to improve a classifier or to catch one that is cheating. The larger goal is trustable image-based assays rather than new predictive power.

## What they studied

Two frozen foundation models: DINOv3-large (general-purpose vision) and ViTAMIn-O (an organoid-specific generalist representation model). The unit of analysis is a learned concept — a dictionary atom over patch and class tokens of the last layer — and, for each concept, a set of properties computed over five pooled organoid datasets. Downstream, five morphology tasks (anterior placode, otic, abnormal brain organoids, budding brain organoids, ductal/PDLO) are run as logistic regressions on those concepts' per-image activations.

## How they did it

They train a sparse autoencoder on the frozen representations: patches resized to 224×224 crops from five public datasets, roughly 60,652 crops per epoch, giving about 3M tokens per epoch for ViTAMIn-O's 7×7 grid and 12M for DINOv3's 14×14 grid plus class token. The SAE is a BatchTopK variant with a dictionary of 4,000 atoms over a 32,000-point convex hull of the representation space, allowing 32 active codes per token, trained for 100 (ViTAMIn-O) or 25 (DINOv3) epochs. Two modifications are added and ablated: an auxiliary top-1 reconstruction loss on samples with no active code, and replacing the free decoder with a relaxed archetypal dictionary whose rows are constrained by a row-wise softmax over convex combinations of hull points plus a norm-bounded free term. Each concept is then annotated on a balanced pool of 1,000 crops per source across five datasets: the token types it fires on (background / border / organoid / global), its prototype images, its activation frequency, its dataset specificity (entropy of the dataset label over the images where it is active, normalised for activation frequency), and its task relevance (adjusted mutual information between the thresholded per-image activation and the task label, maximised over ten thresholds). Finally they test two uses: classifying token type from concept-weighted affinities to get segmentation, and filtering concepts out by their spatial annotation before re-training the downstream logistic regression.

## What counts as an explanation here

A feature dictionary with per-feature metadata — the SAE atoms, each carrying a spatial role, a set of exemplar images, a specificity score and a task-relevance score. It is partial by design and it does not predict. Its claims are of two kinds: a descriptive claim (this atom means tissue at the organoid border), tested by whether the label transfers to unseen datasets, and an instrumental claim (this atom is a confound, or this atom is noise), tested by whether removing it changes downstream accuracy.

## How they judged success

Downstream AUROC of logistic regression on SAE features versus on the raw foundation-model features versus on CellProfiler hand-crafted features, with error bars from two SAEs trained under different seeds. Ablations of the SAE losses (top-1 auxiliary, softmax decoder, archetypal relaxation δ) against reconstruction R², percentage of dead codes, and concept stability — cosine similarity of Hungarian-matched atoms across seed runs. Transfer of spatial labels is measured as balanced accuracy of token-type classification on the five test datasets. Also: fraction of the dictionary needed for a given AUROC, and qualitative inspection of prototype images.

## What they found

The SAE features stay close to the base models (slightly lower, or better for DINOv3 on the anterior placode and otic tasks) and remain at least comparable to CellProfiler features, so the readable dictionary is close to free in accuracy. Fewer than 32 of the most task-relevant concepts already reproduce the full-dictionary AUROC on every task except otic. Spatial labels transfer: DINOv3 concepts reach 81.7% versus 66.4% balanced accuracy for ViTAMIn-O on token-type classification, which the authors attribute to the coarser 7×7 token grid of ViTAMIn-O. Filtering DINOv3 concepts by border/background/foreground affinity improves the budding-brain, PDLO and otic tasks at particular quantile thresholds; the same filtering does nothing for ViTAMIn-O. The most useful negative result: the top task-relevant concept for abnormal (non-budding) brain organoids was reading padding introduced by the authors' own cropping step, and those artifacts correlate with the label — so the dictionary found a preprocessing leak that had been reporting as biology. Config-wise, the top-1 auxiliary loss cuts dead codes for ViTAMIn-O from 55.7% to 5.3% (at R² 0.62→0.55), the softmax decoder lifts DINOv3 stability from 0.59 to 0.84 (dead codes 32.9%→11.4%), and raising δ to 1.0 pushes dead codes lower still.

## Limits and open problems they name

Dataset scale and coverage are acknowledged as a limit; one task's month-to-month differences are not interpreted for that reason. ViTAMIn-O's spatial attribution is too coarse for the filtering trick to work, so the one *useful* result is model-dependent. Only the last layer is studied. Weights and code are promised only upon publication, which means nothing here is reproducible from the preprint. They name as future work other layers, better concept attribution with patch-level intensity/texture features, and aligning concepts with biomedical language towards mechanistic reasoning.

## What we take from it

Three things, in decreasing order of directness. (1) **A reporting format for a dictionary atom.** Four scalars and one exemplar set — activation frequency, dataset specificity as a normalised entropy, task relevance as an MI thresholded and maximised, and the top-5 activating inputs — cost almost nothing to compute and turn "this feature is meaningful" into four separately falsifiable statements. Our per-component readouts should carry the specificity number in particular, because it is what separates a general mechanism from a dataset fingerprint, and it is exactly the axis our batteries never score. (2) **The confound-finding use, which is the real justification for a dictionary.** The strongest result in the paper is that the most task-relevant atom was reading an artifact of preprocessing: relevance-ranked prototypes are a cheap confound detector, and the check is whether the label is present in the artifact. For us the analogue is an input-preprocessing leak reported as a mechanism, which is the failure mode a predictive model would happily encode and never flag. (3) **A warning about saturation in the metric they use to say a feature matters.** Task relevance is the maximum of AMI over ten thresholds — a maximisation over a tuning parameter with no held-out split for that selection, so a concept that is relevant at exactly one lucky threshold scores as well as one that is relevant everywhere; the top-n-curves that follow inherit the optimism. Any ranked-by-relevance filtering we publish should fix the threshold on held-out data or report the whole curve, not its maximum.

## Tags

unit: feature
form: dictionary
method: SAE | probing | other
eval: ablation | downstream-task | benchmark
goal: understanding | debugging
