# Why Units Die in the Layer After a Frozen Random Encoder, and What the Number of Live Units Measures

- **Key:** arxiv-2611.29752
- **Authors / organisation:** Scott M. Norton (Rochester address listed; affiliation given as independent researcher)
- **Date:** 2026-11-22
- **Link:** https://arxiv.org/abs/2611.29752
- **Kind:** paper
- **Read from:** full text (arXiv HTML v1: abstract, sections 1-9 including the discussion, all five tables as cells, and every figure caption; figures 1-7 were not inspected as images, so plotted values are taken only where prose or captions state them)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The author studies the common situation where the first part of a network is never trained - a random convolutional encoder, a reservoir, a frozen pretrained encoder feeding a small head, or the input side of a sparse autoencoder - and asks what that fixed front end does to the first layer trained on top of it. The concrete phenomenon is dead ReLU units: units whose input sum is negative on every example, so they output zero, receive no gradient, and cannot recover. The motivation is partly practical (few live units are usually read as a network in trouble) and partly self-correcting: an earlier paper by the same author concluded that the number of surviving units measures how complex a game is, and this paper tests that conclusion against an alternative explanation and finds it unsupported. The larger goal is to replace a convenient statistic of a network with a measured property of its inputs that can be read before training starts.

## What they studied

PPO agents on three Atari games (Pong, Breakout, Space Invaders) whose encoder - three convolutional layers producing 3,136 numbers per four-frame screen - is frozen at random initialisation, with two 64-unit dense ReLU layers after it. The unit of analysis is the individual ReLU in the first trained dense layer (FC1), characterised by whether it is alive at the end of training. Three non-game systems test generality: small CIFAR-10 classifiers with a frozen random encoder and a 256-unit hidden layer, a reservoir computer on the Lorenz 96 system with a trained 64-unit layer, and sparse autoencoders trained on GPT-2 small layer 8 activations.

## How they did it

They define an *offset* for the inputs of the trained layer - the length of the mean input vector divided by the square root of the total variance across examples - which is large when the examples form a tight cluster far from the origin, and small when they scatter around zero. Because a random encoder multiplies pixel patches by random signs, it preserves the energy of the average screen and of the deviations from it alike, so it passes the offset through roughly unchanged. The author raises the offset by adding a constant to the last convolutional bias, and lowers it by subtracting the average screen (centring) before the encoder, or by adding a fixed vector after the encoder to change the offset while leaving the screens untouched. The same encoder is then trained in an otherwise identical PPO setup and the count of live FC1 units is compared across offset levels. Counts use two rules (a 1 percent switch-on rule as primary, an average-output-above-0.01 rule where pass levels were set in advance), and many thresholds were written down before the run. A separate line of work varies the ratio of learning rate to fan-in to test whether one Adam step's reach, which grows with the number of weights feeding a unit, accounts for the layer with more inputs losing more units. The offset of a random encoder's output is also predicted analytically from the inputs and the layer shapes alone, without drawing weights.

## What counts as an explanation here

A two-quantity causal account with an analytic forward prediction: the number of surviving units is claimed to be a function of (i) how far the layer's inputs sit from zero relative to their spread and (ii) how far one update can move a unit's input sum, both measurable before training. The explanation is explicitly partial and is paired with a refutation - the same author's earlier reading of the count as a complexity measure is withdrawn. Nothing here is a dictionary, a circuit, or a per-prompt predictor of behaviour; it is a property of the layer's input distribution that forecasts a structural outcome of training.

## How they judged success

Pre-registered pass/fail rules per experiment (a rank correlation of -0.8 or below between offset and live units; a median of at least 24 live units after centring; a ratio of 1.5 or less in at least 5 of 6 encoders for "about as fast"), counts under 15 different counting rules for sensitivity, median differences between repeated runs of the same encoder used as a noise reference (up to 7 live units and 16.45 points in Breakout, up to a factor of 1.796 in the step to threshold in Pong), a silencing check that zeroes units counted alive versus units counted dead and asks whether the score collapses, and a 25 percent band for the analytic offset formula across 44 conditions.

## What they found

Raising the offset cut the number of live Breakout units from a median of 16 of 64 to between 0 and 2 across 24 raised-offset agents, with every one of the 144 paired comparisons going the same way, and average training score falling from 21.4 to about 2. Centring Pong raised live units from 3-6 to 59-63 and, crucially, took the agents to a training score of 18 within about 5 million steps where none of the frozen-encoder agents on unaltered screens reached it in 50 million - roughly matching a trained encoder on unaltered screens. Breakout did not close the gap: centring raised live units from 9-26 to 30-37 and let agents reach a score of 10 three to five times sooner, but average score stayed 199-232 points behind the trained encoder. The same sign appeared in classifiers, the reservoir (rank correlation -0.908), and sparse autoencoders on GPT-2, where at eight times the unaltered offset a median 0.991 of units ended dead and most of those had been alive at the start, so they died during training. The offset formula came within 25 percent in all 44 conditions (median error 1.4 percent). The fan-in story is weaker: a learning rate divided by the fan-in held live units roughly constant (classifier exponent 1.158, 0.923 excluding the largest fan-in; sparse autoencoder 0.971 after a fix to an experiment whose pre-set test had failed at 0.670). An encoder that is allowed to train lowers its own offset early - in all twelve runs it dropped below 2.0 within the first 1.5 million agent steps.

## Limits and open problems they name

Small networks, three games, mostly one run per encoder and setting. Changing the offset also changed the overall magnitude of the layer's inputs, and the two were not separated outside the sparse autoencoders. Several pre-registered tests failed or partly failed, including the silencing check in three experiments and a variety of step-size scaling rules. The formula predicts only a median over random encoders, not any particular encoder, and does not apply once weights are trained. Whether offset and step size compound was not tested. The author does not know why Breakout behaves differently from Pong, cannot say how much of the unit loss happens before versus during training in the agents, and does not know whether one causes the fall in the trained encoder's offset. The setting is also explicitly listed as "AI-assisted": most code and drafting was done by Claude under the author's review.

## What we take from it

Three reusable pieces. A **warning with a mechanism**: a statistic of a dictionary or a layer - how many units are alive, how many features fire - is a function of the *input distribution's geometry*, not only of the model or the task, so two models differing in that statistic can differ only in the mean-to-spread ratio of whatever feeds the layer; before publishing any count of live, dead or dormant units, publish the offset or an equivalent normalised-geometry number computed the same way on the same data. A **cheap pre-training diagnostic**: for any layer we train on fixed features (probe readouts, SAE inputs, activations from a frozen model), one pass computing mean magnitude over spread tells us whether a large fraction of ours will be born dead - and the prescribed remedy, subtracting the data mean, is one line. A **template for retracting a metric**: the author withdraws the earlier "unit count measures task complexity" reading by holding the task fixed and moving one input property, which is exactly the experiment our head keeps asking for when a metric is promoted into an explanation - pair the metric with an input-side variable you can move without changing the task. Cross-reference run 696 (a per-unit worth is a property of the ablated set) and run 699 (latents positioned by chat formatting rather than meaning): all three say that a unit's reading is co-determined by its inputs, not its concept.

## Tags

unit: neuron | feature | layer
form: description | probe
method: probing | ablation | other
eval: ablation | prediction/simulation | benchmark
goal: understanding | evaluation-of-methods
