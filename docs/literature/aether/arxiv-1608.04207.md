# Fine-grained Analysis of Sentence Embeddings Using Auxiliary Prediction Tasks

- **Key:** arxiv-1608.04207
- **Authors / organisation:** Yossi Adi, Yoav Goldberg — Bar-Ilan University, Ramat-Gan; Einat Kermany, Ofer Lavi — IBM Haifa Research Lab; Yonatan Belinkov — MIT CSAIL
- **Date:** 2017-02-09 (v3; v1 2016-08-15)
- **Link:** https://arxiv.org/abs/1608.04207
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Sentence embeddings were being judged almost entirely by how well they performed when plugged into a downstream classifier, and the authors argue that this is too coarse a verdict to say anything about *what* a sentence vector contains. Two encoders can reach the same sentiment score while one retains the words and the other has dissolved them; the downstream number cannot separate the two. They want a measurement that is independent of any consumer task, so that representations can be compared on their contents rather than on their luck with a particular head. It is a step toward opening the sentence vector as such, and it is the paper that established the sentence-level probing paradigm — the three tasks it invents here (length, word content, word order) are the ancestors of the ten-task battery the folder already has at source.

## What they studied

Unit of analysis is the **whole-sentence vector**, plus word vectors as side inputs to two of the three probes. Three representation families: (1) CBOW — the element-wise average of word2vec skip-gram vectors, deliberately order-blind; (2) an LSTM auto-encoder (encoder-decoder trained to reproduce the input sentence), the "general purpose encoder" model class; (3) skip-thought vectors, taken as a published checkpoint rather than retrained. Embedding size *k* is swept over {100, 300, 500, 750, 1000} for the first two, and skip-thought is used at its recommended 4800 dimensions. Corpus: 1M sentences from a 2012 Wikipedia dump for training the encoders, sentence lengths constrained to 5–70 words, 50k vocabulary; the probe datasets are drawn from **200,000 Wikipedia sentences** (150k train / 25k dev / 25k test) which are a subset of the encoders' own training data, deliberately, so the encoders are measured on their home territory.

## How they did it

Three prediction tasks, each isolating one surface property of a sequence. **Length**: eight binned length classes (5-8, 9-12, …, 34-70), multiclass; the majority class is 5-8 words and predicting it alone gives 20.1%, which is their floor. **Word content**: given the sentence vector concatenated with a word vector, predict whether that word occurs in the sentence — binary, balanced, 50% floor. **Word order**: given the sentence vector and two word vectors that do occur in the sentence, predict which came first — binary, balanced, 50% floor, with the negative example made by flipping the order of the same pair. Probe architecture is a one-hidden-layer ReLU MLP, hidden size equal to the input size, softmax head, dropout 0.8 before the softmax, AdaGrad at 0.01, early stopping after five non-improving epochs — tuned once and held fixed across all experiments. The paper's real methodological move comes after the first results: two **control datasets**. In one, every word is replaced by a random word from the dictionary (so word identity and long-range lexical cues are destroyed but sentence length is kept); in the other, call it permuted, each sentence's word order is randomly shuffled while the same trained encoder is used. Additionally, for the order task they retrain the probe with the sentence vector removed entirely, leaving only the two word vectors, which measures how much of the "order" score is just general statistics of English word pairs.

## What counts as an explanation here

A **task-relative capability profile**: a triple of accuracies (length, content, order) plus a curve of each against dimensionality, plus the deltas under the two control datasets. It is explicitly partial and explicitly behavioural — the authors say at the outset that these tasks measure preservation of surface structure and deliberately do not measure syntactic or semantic generalization, and they name that as a limitation rather than a claim. It predicts something modest but real: it predicts how the representation will behave at low-level tasks, and it predicts the failure mode of a downstream use (a content-blind encoder will not support retrieval). It does not localize anything inside the network, and there is no claim about mechanism.

## How they judged success

Accuracy on each probe, read against its stated floor (20.1% for length, 50% for the two binary tasks), reported as curves over embedding size and compared against the encoder-decoder's reconstruction **BLEU** as the incumbent proxy for representation quality. The two synthetic datasets (random-word substitution, permutation) are the ablation axis, and the sentence-vector-removal variant of the order task is the third control. Significance is asserted with paired t-tests, in three appendix tables giving p-values per dimension pair per task; they also report the pairs that came out *not* significant (ED 500 vs 750 on order, p=0.11; CBOW 750 vs 1000, p=0.3), which is a more honest table than most of this era.

## What they found

LSTM encoder-decoders encode length at **82–87%** with enough dimensions, and order at up to **91%** at 1000 dimensions; both saturate rather than growing smoothly with size. Word content peaks at 750 dimensions and then **drops** at 1000, while the decoder's BLEU keeps rising over the same range — the paper's interpretation is that the decoder gets strong enough to carry the sentence itself, letting the encoder become less informative. CBOW is the surprise: **45–65%** on length against a 20.1% majority baseline, and **70%** on word order against a 50% floor, from a representation that by construction has erased order. The controls explain both honestly. Replacing every word with a random word leaves the CBOW length score nearly unchanged, so it is not vocabulary-driven, and the sentence vector's **norm decreases as the sentence gets longer** — the authors give the central-limit argument (averaging more roughly zero-centred vectors pulls the mean toward the origin), and the effect is strongest where sentences are short, as the argument predicts. For order, removing the sentence vector from the CBOW probe costs almost nothing while the LSTM-ED loses a great deal, so CBOW's 70% is general English word-pair statistics plus about 3 points from the sentence itself. On Permuted input the LSTM encoder is essentially unaffected on all three tasks (the decoder's BLEU, by contrast, falls from 32.5 to 8.2), so the auto-encoder is a general sequence encoder that does not lean on natural word order; **skip-thought is not** — its length accuracy falls from 82.1% to 68.2% and content from 79.7% to 76.4% under permutation, so it is specialized to natural text. Their closing claim is a warning: **BLEU over the decoder is sub-optimal for choosing an encoder**, because content accuracy and BLEU peak at different sizes and can move in opposite directions.

## Limits and open problems they name

The tasks measure surface preservation only, not syntax or semantics, and the authors state that closing this needs a complementary set of auxiliary tasks they leave open. The conclusions are empirical and may not generalize past the datasets used, though they defend Wikipedia as representative for these three properties. Unstated but visible: the probe dataset is drawn from the encoders' own training distribution, there are no seeds or variance estimates (only t-tests over probe runs), the skip-thought comparison is across corpora and dimensions and they say so, and the probe architecture is fixed at one MLP configuration so probe capacity is never varied — precisely the gap the control-task paper fills.

## What we take from it

Three things, and one is a warning. **(a) A matched negative control dataset is worth more than a bigger model.** Substituting random words and permuting word order are two edits to the *input* that leave the measurement pipeline untouched, and they turn two impressive numbers (CBOW at 70% on order and 45–65% on length) from findings about representation into findings about natural-language statistics. Any probe battery we run should define its own two such edits per task before it reports a score. **(b) A norm confound is the cheapest possible alternative explanation for a length-correlated read-out, and it should be checked by default** — here a sentence vector's magnitude is a function of how many vectors went into it, so an accuracy a probe extracts from a magnitude is not a property the encoder learned. **(c) The downstream proxy and the probe can disagree in sign** (content accuracy down at 1000 dims while BLEU up), and the paper reads this as evidence that the proxy measures the *decoder* rather than the representation — the same "score the object the consumer actually receives" rule the probe-versus-parser entry reached from the other direction. Warning: the probing dataset here is a subset of the encoders' training data, which invites the memorization the later control-task work had to price — do not copy that choice.

## Tags

unit: whole-model
form: probe
method: probing
eval: benchmark
goal: science
