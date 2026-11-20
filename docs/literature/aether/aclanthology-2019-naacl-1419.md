# A Structural Probe for Finding Syntax in Word Representations

- **Key:** aclanthology-2019-naacl-1419
- **Authors / organisation:** John Hewitt, Christopher D. Manning — Stanford University
- **Date:** 2019-06 (NAACL-HLT 2019, Minneapolis, pp. 4129-4138)
- **Link:** https://aclanthology.org/N19-1419/
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

By 2019 the standard way to ask whether a representation contains linguistic knowledge was to train a probe and read off its accuracy. The authors argue that this family of methods answers a weaker question than it appears to: a probe that reports part-of-speech tags, or even a probe that reconstructs a tree from whole-sentence vectors, does not test whether the *entire* syntax tree — every word's distance to every other word — is present in the geometry of the representation. Their goal is to state that stronger claim as a concrete, falsifiable hypothesis about a linear transformation of the representation space, and then test it on two pretrained models and on baselines that should not pass. The framing they want to inherit is the vector-offset claim for analogies: a structural property of a space, not a task performance.

## What they studied

ELMo (5.5B-word pretrained weights) and BERT (base cased, large cased), probed layer by layer, with ELMo layers of dimension 1024, BERT-large layers 1024 and BERT-base 768. Gold structure is the Stanford Dependencies conversion of the Penn Treebank phrase-structure parses (the standard parsing train/dev/test splits), with no preprocessing; subword vectors are averaged into PTB tokens, which the authors note makes the BERT numbers a lower bound. The unit of analysis is a word's contextual vector at a given layer, and the object claimed is a linear map on the whole representation space — a global metric, not a per-word label.

## How they did it

Two probes share one parameterisation. The *distance probe* learns a matrix B (equivalently a positive-semidefinite form hᵀAh with A = BᵀB) and trains it so that the squared distance between transformed vectors of two words reproduces the number of edges between them in the gold tree; the loss is the absolute error over all word pairs, normalised per sentence by the squared sentence length. The *depth probe* keeps the same form but matches a word's squared norm against its depth, i.e. its edge count to the root. Optimisation is Adam at learning rate 0.001, batch 20, up to 40 epochs with a dev-loss-triggered learning-rate reset. Trees are not read back out of heads: the predicted pairwise distances are turned into a tree by taking the minimum spanning tree of the distance graph, which is what the accuracy metric scores. Because the map is a global metric, the same matrix is fitted across all sentences, so a probe that "learns to parse" per sentence cannot hide. Four baselines are designed to supply useful features without being able to parse: a left-to-right chain, character-level embeddings with no context, an exponentially distance-weighted average of those embeddings, and — crucially — the same embeddings passed through a randomly initialised BiLSTM of ELMo's width, which they call PROJ0.

## What counts as an explanation here

A linear transformation, i.e. a direction-defining object with a clear operational meaning: an inner product on the original space under which squared distance *is* tree distance and squared norm *is* depth. It is a partial explanation by design — one property of the space, stated as an existence claim and fitted by supervision — but it is a predictive one in a narrow sense: from the fitted map you can reconstruct a tree for a new sentence and compare it with a parser's output. The authors add a second, quantitative layer to the explanation: the *rank* of the map required to do this, which turns "syntax is encoded" into "syntax is encoded in a low-dimensional subspace".

## How they judged success

Undirected unlabeled attachment score (UUAS) of the minimum spanning tree against the gold tree, with punctuation ignored; distance Spearman (DSpr.), the per-word correlation of true against predicted distances, averaged over sentences of a fixed length and then macro-averaged over lengths 5-50; and for the depth probe, norm Spearman with the same averaging, plus root% — whether the least-deep word is identified as the sentence root. Baselines are the four designs above, plus the layer sweep as its own control: the same probe architecture applied to every layer, so the layer effect is measured within a fixed instrument. The rank analysis sweeps the truncation dimension k of B and re-measures reconstruction accuracy.

## What they found

Table 1 on the PTB WSJ test set gives UUAS / DSpr. / root% / NSpr. as LINEAR 48.9 / 0.58 / 2.9 / 0.27, ELMO0 26.8 / 0.44 / 54.3 / 0.56, DECAY0 51.7 / 0.61 / 54.3 / 0.56, PROJ0 59.8 / 0.73 / 64.4 / 0.75, ELMO1 77.0 / 0.83 / 86.5 / 0.87, BERTBASE7 79.8 / 0.85 / 88.0 / 0.87, BERTLARGE15 82.5 / 0.86 / 89.4 / 0.88 and BERTLARGE16 81.7 / 0.87 / 90.1 / 0.89. So the ordering is BERT-large above BERT-base above ELMo, all far above every baseline; note that the two context-free and decay baselines sit at or barely above the left-to-right chain, while the randomly initialised BiLSTM is a strong baseline that the trained models still beat by a wide margin. The layer sweep shows syntax peaking in the upper-middle layers of both architectures rather than monotonically rising. The rank sweep is the paper's most striking number: accuracy on the transformed distances saturates around k of 64 to 128, with no further gain from a larger transformation, i.e. the syntactic structure the probe recovers occupies a small subspace of a 768- or 1024-dimensional layer. The authors note all three models seem to need maps of roughly the same rank and leave that unexplained.

## Limits and open problems they name

They say plainly that the probe tests one very strict notion — that all pairs of words carry their syntactic distance in the geometry — and not syntactic knowledge in general, and that a deep parser-like probe would almost certainly score higher on tree reconstruction, so their numbers are not a parser comparison. Their reviewers pushed on whether they should have probed headedness directly rather than distance, which they answer by arguing distance avoids arbitrary decisions about PP attachment, determiner heads and auxiliaries. Why squared distance works better than distance itself is left open, along with what function of the distance would be optimal. The shared transform rank across models is flagged as unexplained. They also note the hypothesis was developed on LSTMs and then applied unchanged to a self-attention model.

## What we take from it

Three things transfer directly. First, the **framing**: state the interpretability claim as a hypothesis about a *structural property of the whole space* (an existence claim about a metric), not as a classification accuracy, because that form admits falsification by a baseline that cannot possibly have the property — and here two of the four baselines are essentially at the floor, which is what makes the top of the table mean anything. Second, the **randomly initialised same-architecture baseline** is the load-bearing control and is the single cheapest device in the paper: it holds width and nonlinearity fixed and removes only training, so the gap between it and the trained model is the only part attributable to learning. Third, **the rank sweep is a measurement of the read-out's footprint**: fitting a map of every size and reading where accuracy saturates converts "a linear explanation exists" into "a k-dimensional linear explanation exists", which is exactly the form our own white-box claims should take — a directional claim should come with the smallest dimension in which it holds. Warning to carry: the probe is *supervised* on the same trees it is scored on, so the score measures geometric recoverability, not use — a point the folder's `arxiv-2005.01641` entry makes from the parsing side and `2020.acl-main.698` from the behavioural side.

## Tags

unit: direction | layer
form: probe | direction
method: probing | other
eval: benchmark | prediction/simulation
goal: understanding | evaluation-of-methods
