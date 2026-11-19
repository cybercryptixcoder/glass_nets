# Sanity Checks for Saliency Maps

- **Key:** arxiv-1810.03292
- **Authors / organisation:** Julius Adebayo (MIT, work done during the Google AI Residency); Justin Gilmer, Michael Muelly, Ian Goodfellow, Been Kim — Google Brain; Moritz Hardt — Google Brain and UC Berkeley
- **Date:** 2018-10 (v1 2018-10-08; v3 2020-11-06, the version read)
- **Link:** https://arxiv.org/abs/1810.03292
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 3 (central)

## What they wanted, and why

Saliency maps had become the default way to show what an image classifier "looked at", and the field was choosing between them by how convincing the pictures were. The authors' worry is methodological rather than technical: if a map is judged by its appearance, a method can be selected for producing plausible-looking masks and never be checked against the model it claims to describe. They therefore set out to build a test that asks a deliberately blunt question — does this explanation change at all when the thing it supposedly explains is destroyed? — and to apply it across several methods at once. Their larger goal is the one this folder shares: a way to reject an account as inadequate *before* it is deployed, using a test that costs one retraining or one re-initialisation and needs no human raters.

## What they studied

Image classifiers and pixel-level explanation maps. Models: Inception v3 pretrained on ImageNet (93.9% top-5 on the ImageNet test set, 17 blocks for per-block randomisation), a small CNN on MNIST and Fashion MNIST (two 5x5 conv+pool stages, a 1024-unit fully connected layer, 99.2% test accuracy), a 2500-1500-500-10 MLP on the same two datasets (98.7%), and one Inception v4 trained on pediatric skeletal radiograms. Methods examined: raw gradient, gradient-times-input, integrated gradients, SmoothGrad, VarGrad, GradCAM, Guided Backpropagation, and Guided GradCAM (the element-wise product of the last two). The unit of analysis is the per-input explanation map `E: R^d → R^d` produced by an attribution method, i.e. the map itself, not the weights — though the tests are the first in this folder to interrogate the *weights* through the map.

## How they did it

Two tests, both framed as invariances. **Model parameter randomisation**: take the trained network and re-initialise its weights, then compute explanations with the same method on the same inputs and measure how similar the two maps are. Two variants — *cascading* (re-initialise from the top block down, one block at a time, so the map is tracked as learned weights are destroyed layer by layer) and *independent* (randomise exactly one block or layer while all others keep their trained values, which isolates which depth of weights a method is actually reading). **Data randomisation**: permute the training labels, train a fresh model of the same architecture to above 95% training accuracy (so it must memorise the shuffled labels rather than learn any input-label relationship), then compare its explanations with those of the honestly trained model on a held-out set of inputs. Similarity is quantified four ways — Spearman rank correlation with and without absolute value, SSIM, and the Pearson correlation of histograms of oriented gradients — with an explicit calibration exercise: each metric's value for a map against a random mask, and for two random maps, is measured so that "high" can be given a scale. Finally the authors argue the mechanism in closed form: for a linear model the gradient is the weight vector; for a one-layer 3x3 convolution with ReLU and sum-pooling the input gradient at a pixel is a sum of the filter taps gated by the local ReLU pattern, so the activation pattern — and therefore an "edge" in the image — determines the map, independent of what the filter learned.

## What counts as an explanation here

A per-input relevance map, treated as an object whose *sensitivity* is the property that matters. The paper does not produce an account of the network; it produces a **rejection test** for accounts. The claim is deliberately partial: passing means the method is not constant in the model parameters and the labels, which is a necessary and not a sufficient condition for being an explanation. Implicitly it defines the form understanding must take for these methods — an attribution must be a function of the trained weights and of the learned input-label relationship — and it names the failure mode where a method is really an unsupervised image-processing routine (edge detection, segmentation, denoising) that happens to look like an explanation.

## How they judged success

The four similarity metrics above, reported as curves across randomisation depth rather than as scalars, with SSIM and HOG computed only on ImageNet because those metrics assume natural-image statistics. Controls: the metric calibration against random masks (gradient vs Gaussian mask returns essentially zero on all four, e.g. −0.00049 / 0.00032 / −0.0016 / 0.00027, so a high score is not an artefact of the metric), truncated-normal versus uniform re-initialisation (identical results reported), cascading versus independent randomisation (used as a mutual check), and the one-layer convolution plus linear-model analysis as an analytic baseline. Guided Backprop is additionally checked against a known theoretical result (that it partly recovers the input). No human evaluation is used anywhere — the whole point is a mechanical criterion.

## What they found

Gradient, gradient-times-input, integrated gradients, SmoothGrad and GradCAM change substantially when weights are re-initialised; Guided Backprop and Guided GradCAM do not — their maps stay close to the trained model's until the *lowest* blocks (the ones nearest the input) are randomised, so they are effectively blind to the higher-layer parameters that carry what the network learned. Under label randomisation the same split appears: gradient-based maps change, GradCAM's patches fragment, while Guided Backprop still marks a large, plausible-looking region of the digit. On the "does it look right" axis every method *fails* the test in the same way: after several blocks of randomisation the bird in the ImageNet example is still recognisable in the gradient and integrated-gradient maps, and SSIM and absolute rank correlation stay high, while the signed rank correlation collapses to near zero as soon as the top blocks are randomised — visual inspection and sign-sensitive ranking disagree completely. The input-gradient product experiment explains why: multiplying a fixed input by two independent random gradient fields leaves maps similar to each other and to the input, so methods that approximate "input times gradient" are largely returning the input. The one-layer convolution analysis gives the same mechanism analytically: the local ReLU activation pattern determines each pixel's gradient, edges have distinctive patterns, uniform regions do not, so the map draws edges regardless of the filter's learned values.

## Limits and open problems they name

The tests are necessary-not-sufficient: passing does not establish that a map is a good explanation, only that it is not invariant to the model and the data. The paper covers image classification with pixel-level attribution on two small datasets plus one large model, and states that its analysis is instantiated there while claiming generality for the methodology. Architecture remains a live confound — they concede that an explanation insensitive to parameters may still be informative about the *architectural prior*, and say such a method should only be used where architectural knowledge alone suffices. The label-randomisation test requires training one model per model-dataset pair, so it is not free. The authors flag further transformations (other invariances) as unexamined and present the work as a stepping stone rather than a verdict. A version note in the paper records that an earlier draft over-claimed: Guided Backprop is not wholly invariant, only invariant to the higher layers — corrected in the version read here.

## What we take from it

**A method, and it is the cheapest control in the folder.** Re-initialise the weights, recompute the account, and measure how much it moved — a read-out that does not move is not reading the trained parameters, whatever it looks like. **A second method**: retrain on shuffled labels and check that the account changes; for us the analogue is refitting the decoder on a model whose training signal was destroyed. **A measure**: report similarity with signed and unsigned correlation *and* a perceptual metric, because this paper's central empirical fact is that these disagree — the unsigned/SSIM view stays high while the signed view collapses, and only the signed one tracks whether the relevance was ever real. **A calibration habit**: print the metric's value for an account compared with a random object, so a reader can see what "0.8 similarity" means on that scale. **A warning that lands directly on us**: the "input-times-gradient returns the input" argument is the picture-domain version of a decoder that reads its own input, and the edge-detector comparison names the outcome — a parameter-free image statistic dressed as an account. **A framing**: state invariances explicitly. Instead of asking "is this explanation good", ask "what transformations leave it unchanged", and reject it if one of those transformations is something the task requires it to be sensitive to.

## Tags

unit: weights
form: description
method: weight-analysis
eval: faithfulness
goal: evaluation-of-methods
