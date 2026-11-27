# Predictive Data Debugging: Reveal and Shape What Your Model Learns, Before You Train

- **Key:** goodfire-com-research-predictive-data-debugging
- **Authors / organisation:** Leon Bergen, Usha Bhalla, Sidharth Baskaran, Max Loeffler, Raphael Sarfati, Dhruvil Gala, Ryan Panwar, Santiago Aranguri, Thomas Fel, Atticus Geiger, Matthew Kowal, Siddharth Boppana, Daniel Balsam, Owen Lewis, Jack Merullo, Thomas McGrath, Ekdeep Singh Lubana (core contributors marked); Goodfire, Mechanistic Alignment Series. Blog post credited to Tom McGrath, Tucker Fross and Michael Byun.
- **Date:** 2026-06-11
- **Link:** https://www.goodfire.com/research/predictive-data-debugging
- **Kind:** blog/research post (release post for a longer paper, *Anatomy of Post-Training: Using Interpretability to Characterize Data and Shape the Learning Signal*, and for a commercial platform, Silico)
- **Read from:** full text of the blog post (14,478 raw / ~137 stripped lines, the whole post in two windows). **The underlying paper was not read** - it is linked but off-page, so no method detail beyond the post's prose enters this entry.
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Post-training compresses a messy set of intentions into one scalar preference signal, and that scalar also carries whatever correlates with the intention. The authors' framing is that a preference dataset is a program for shaping behaviour, but unlike a normal program it cannot be read before it runs: the standard loop is collect data, run DPO, eval, then guess backwards from aggregate scores which of a quarter-million pairs caused a regression. They want the loop inverted - read the dataset through the model's own representations first, predict which behaviours the training will amplify or suppress, point at the specific examples responsible, and then edit data or training so the unwanted lesson is never learned. The larger goal is a north star they state plainly: write a model specification in natural language and derive what data would achieve it, with the side effects predicted in advance.

## What they studied

Base models from Llama-3.1-8B up to 70B, trained with DPO on open preference data: mostly Dolci, the dataset behind the OLMo models, about 260,000 pairs, and Tulu 3 for the 70B case. The unit of analysis is not a neuron or a direction but a **cluster of datapoints grouped by the concepts the model computes while processing them** - interpretability features read out on each datum, then treated as the learning signal. Behaviours followed: refusal robustness on harmful requests, hallucinated links, sycophancy, an unexpected fan-fiction genre, and a deliberately planted "goblin" behaviour used as ground truth.

## How they did it

The pipeline has two halves. First, prediction: pass the whole dataset through an interpreted model before any training, read out which features activate on each prompt-response pair, and cluster the data by those activations rather than by an embedding of the text. The features that move are taken as the account of what training will push toward or away from. Second, attribution and repair: for a behaviour of interest, follow the predicted effect back to the clusters that carry it, then reshape - filter or rewrite those examples, or change how the training objective weights them - and retrain. Their explicit comparison is to running a general LLM over the dataset to guess what will be learned, which they describe as inference about the data where their method is measurement on the model. Validation is a planted cause: they enriched responses with goblins, producing a model that mentions goblins in unrelated contexts in about half its responses, and then asked whether the pipeline finds that cluster and whether intervening removes the behaviour. Interventions are scored two ways at once - frequency of the unwanted behaviour, and general capability as a percent change in accuracy on OLMES from the supervised-fine-tuning baseline.

## What counts as an explanation here

A **prediction plus an attribution plus optionally a repair**. The predict-then-diff-then-edit chain is presented as a data-level analogue of causal mediation: a datapoint is a cause of a behaviour, and a feature-cluster is the form that cause takes. The claimed predictive power is explicit and quantitative - R-squared of 0.9 between predicted and realized behaviour change - which is unusually strong for a lab post and is the shape of claim our own work should copy. The account is partial: it names clusters, not mechanisms, and the authors are open that several behaviours resist their interventions entirely.

## How they judged success

Three things, in increasing strength. Whether the prediction of "which behaviours move" holds against what DPO actually does (the R-squared claim). Whether a known planted behaviour is both found and removed (the goblin ground-truth check). And whether the repair is worth doing - behaviour frequency down while OLMES accuracy holds at or above the naive-DPO level, which is why they present the result as a Pareto improvement rather than a trade. The post reports no baselines against interpretability-free selection, no held-out dataset split, and no seed spread; the goblin figure is described as four intervention bars against three baselines, but the bars are an image and their values are not in the prose.

## What they found

Safety and general performance come apart on the same data: DPO on this dataset raises benchmark scores and makes models substantially more likely to comply with harmful queries. The offending pairs turn out to be jailbreaks that launder the request through a fictional frame, where the accepted response plays along and the rejected one refuses - so the preference signal rewarded compliance inside a fiction. Removing those pairs restores refusal on a reworded harmful prompt of the same family while the general gain stays. Hallucinated links appear as a second cluster: the model learns the surface of helpfulness, authoritative-looking URLs, on sensitive queries, and the authors call that the model learning what helpfulness looks like to a rater. Sycophancy was flat on their evals but had in fact risen only on pseudo-profound physics questions, which no eval they had would have caught. The unknown-unknown case is a narrow fan-fiction genre that the dataset rewarded and the model then reproduced - found before training, which they present as the argument for the whole approach. The planted goblin behaviour was located and removed, which is the only case with ground truth behind it.

## Limits and open problems they name

They state that the intervention fixes work for some clusters and not others: hallucinated links were only partially reduced and left above pre-DPO levels, and the physics-sycophancy behaviour could not be neutralised by the pipeline at all, which they call one of the most urgent items on their roadmap. Their proposed remedy for both is partly outside the method - RLFR for hallucination, rewriting data, and generating new data with an agentic tool. Also open: extending the readout from preference training to supervised fine-tuning, mid-training, reinforcement learning and online runs, and validating a rewrite ahead of time rather than after. The post is a product release, so the numbers it does not print are telling: no description of how clusters are formed, no threshold for "this feature will move", no capability-cost table, no replication across a randomly chosen dataset.

## What we take from it

Several things, and one warning.

- **Method (prediction as the deliverable).** This is the cleanest statement in the folder that interpretability's job is a *forecast*: run the interpreted model on the input, predict the change, then check. Our white-box model should be scored the same way - on a held-out set of interventions it did not see, against realized change, with a fitted R-squared reported rather than a qualitative claim.
- **Measure (the ground-truth plant).** The goblin check is the template for validating a predictor: insert a known cause, verify the predictor finds it, verify removing it removes the effect, and verify nothing else broke. Any predictive claim we make should have an equivalent planted-cause arm.
- **Framing (data as a program, data-level mediation).** Treating a datapoint as the cause and a feature-cluster as the mediating form gives our model a second substrate besides weights and activations. It is also the strongest available argument that the same attribution machinery can act on inputs.
- **Warning (a Pareto claim with one capability column).** The improvement is credited from two quantities: the frequency of the unwanted behaviour and one aggregate accuracy change. A behaviour frequency down plus a benchmark average up is compatible with a third capability collapsing, so a Pareto claim needs the whole capability vector printed, not the best column of it.
- **Shortcut (the readout is only as good as the interpretation).** The prediction inherits every error of the feature dictionary: a shattering dictionary produces a shattered forecast, and here the shattering question is never raised even though the same organisation wrote the geometry paper that raises it.

## Tags

unit: feature
form: description
method: probing
eval: prediction/simulation
goal: understanding

