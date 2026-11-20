# A Tale of a Probe and a Parser

- **Key:** arxiv-2005.01641
- **Authors / organisation:** Rowan Hall Maudslay, Josef Valvoda, Tiago Pimentel, Adina Williams, Ryan Cotterell — University of Cambridge (the first three and Cotterell's group at Cambridge, by the byline emails), Facebook AI Research (Williams), ETH Zürich (Cotterell)
- **Date:** 2020-05-12
- **Link:** https://arxiv.org/abs/2005.01641
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Probing had drifted away from borrowing methods that already exist for the task being probed. The authors object specifically to the *structural probe* (Hewitt & Manning, 2019), which recovers dependency trees from contextual embeddings by regressing pairwise syntactic distances through a learned bilinear form — a design with no counterpart in the parsing literature. Their worry is methodological, not empirical: if the probe's architecture and loss are invented for the occasion, then a reading of the resulting number cannot distinguish a fact about the representation from an artefact of the tool. So they ask a narrow, answerable question — does a conventional parser with *the same* lightweight parameterisation do better? — and, having got a divided answer, argue that the real neglected problem in probing is not probe design but **metric justification**. The larger goal is a probing practice where every claim about a model carries an argument for why the chosen yardstick measures what is claimed.

## What they studied

One frozen model: multilingual BERT (12 layers, 768 hidden, 110M parameters) final hidden layer, word-level vectors obtained by averaging WordPiece vectors. Nine typologically diverse Universal Dependencies 2.4 treebanks — Arabic, Basque, Czech, English, Finnish, Japanese, Korean, Tamil, Turkish — each at the largest available treebank, pruned to 12,000 sentences in an 8:1:1 split, with one-word sentences and sentences over 50 words dropped. Unit of analysis: a sentence's predicted tree, and through it the pairwise-distance matrix over words. Two models share one parameterisation: a rank-*r* matrix *B* defining a Mahalanobis-style distance between contextual vectors.

## How they did it

Both models learn the same distance function — the square root of a quadratic form in the difference of two word vectors, with a learned *B* of rank at most *r* — and differ only in the loss applied to it. The **structural probe** minimises the summed absolute difference between predicted distances and gold tree distances over all word pairs, with each sentence's loss divided by the square of its length so long sentences do not dominate. The **parser** is a structured perceptron: it scores a candidate tree by the sum of predicted distances along its edges and takes the loss to be the gold tree's score minus the score of the minimum-spanning-tree candidate found by Prim's algorithm, which is zero when the prediction is exactly right. Their argument is that swapping loss functions is not a complexity increase, so this is a fair test of "is the bespoke design doing anything". At test time both are decoded the same way: compute the *n*×*n* distance matrix, run Prim's to get the minimum spanning tree, score it against the gold tree. Three hyperparameters (rank *r*, learning rate, dropout) are searched randomly per language against development loss; batch 64, early stopping after five steps without improvement, Adam. They then report three numbers: **UUAS** (percentage of correctly recovered undirected edges, the standard parsing metric), **DSpr** (Spearman correlation of predicted distances with gold distances over all word pairs, the metric Hewitt & Manning introduced), and **DSpr P+FW** (DSpr recomputed after the predicted matrix is passed through Prim's and then Floyd–Warshall, so both models are scored on the rank-order implied by their decoded tree).

## What counts as an explanation here

Not a representation claim and not a white-box model — an argument about the *instrument*. What they deliver is a controlled comparison of two measurement devices that share every parameter, plus a normative claim about which reports should be trusted. It is meant to be partial and adversarial: they do not claim to say what syntax BERT encodes, only that one widely used way of asking the question understates the answer relative to a standard alternative, and that the second metric in that probe's own paper is uninterpretable without justification.

## How they judged success

Three metrics on the same held-out sentences, read against each other: UUAS (the parsing community's number), DSpr (the probe paper's number), and DSpr P+FW (a hybrid that removes the encoding-strategy confound). Nine languages rather than one, with the disagreement examined per language. Hyperparameters chosen per language on development loss, so no model is handicapped by a shared setting. No statistical tests, no error bars, no seeds reported.

## What they found

The two metrics give **opposite** rankings of the same two models. By UUAS the parser wins in seven of nine languages, and in English by 11.1 points. By DSpr the structural probe wins in all nine, with the parser faring worst on Japanese and Arabic — which they attribute speculatively to longer average sentence length in those treebanks, noting the irony that these languages had among the largest BERT pretraining corpora. The third metric dissolves most of the disagreement: after the Prim-then-Floyd–Warshall post-processing the mean DSpr gap across the nine languages is 0.0006 in the parser's favour, and the Arabic and Japanese deficits close almost entirely. Their reading is mechanical rather than mysterious — each model is trained to optimise what it is scored on, and the parser only commits to a near/far ordering of pairs because that is all Prim's needs, so a direct distance-correlation metric punishes it for a compression that costs nothing once the tree is decoded. Their conclusion leans to UUAS: it is standard, well understood, and its trend is clear, whereas the hybrid shows no difference, which suggests recovering pairwise distances from a good tree is easier than the reverse.

## Limits and open problems they name

One model (multilingual BERT), one layer (the final one), one probe family; the numbers are explicitly declared incomparable with Hewitt & Manning's because they use UD rather than the Penn Treebank and a 12,000-sentence subset. Word vectors are naive averages of WordPiece outputs, which they say likely weakens both models. Only the distance half of the structural probe is tested — the depth/branching half that supplies tree ordering is set aside. They report no variance across seeds or random-search draws, so "seven of nine" is reported without a stability estimate. A reviewer's objection is recorded and half-conceded: injecting the tree constraint through Prim's means the parser cannot show trees emerging organically, though they reply that the structural probe is trained on the same trees merely re-expressed as distances. The metric question they raise is left as an open programme rather than resolved.

## What we take from it

A framing and a warning, both directly transferable. The **framing**: a read-out's score is jointly a property of the representation *and* of the loss the read-out was trained under, and when two read-outs share capacity but differ in loss, the loss picks which metric they win — so any probe score we report is a claim about the pair (representation, probe objective), never about the representation alone. This is the folder's recurring "decodable versus used" distinction arriving from a different direction, and it sharpens the practical rule: report the probe objective beside the score, and prefer objectives that are conventional for the target task, because familiarity substitutes for validation. The **warning** is about invented metrics: a new yardstick introduced by a paper is a hypothesis, and the burden is on the proposal, not on the standard. For our own batteries the concrete move is to add a **decoding-equivalent** version of every read-out (score the object the downstream consumer would actually receive, not the raw regression output) and to report the standard metric of the field alongside the bespoke one. The verb "dig" in their definition of a probe — a probe should expose what is present rather than search for it — is also a usable one-sentence statement of the capacity constraint this folder's investiture entries have been circling.

## Tags

unit: direction | feature
form: probe
method: probing | other
eval: benchmark
goal: evaluation-of-methods | understanding
