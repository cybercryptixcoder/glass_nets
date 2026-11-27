# Recovered or Memorized? Investigating Dynamical Consistency and Faithfulness in World Model Latents

- **Key:** arxiv-2611.18807
- **Authors / organisation:** Jerome Francisco (Southern Alberta Institute of Technology), Kyle Deng (Carleton University), George Flint (University of California, Berkeley, listed as senior author), Kevin Zhu (Algoverse AI Research); v1 2026-11-13, cs.LG, CC BY 4.0, code released.
- **Date:** 2026-11-13
- **Link:** https://arxiv.org/abs/2611.18807
- **Kind:** paper
- **Read from:** full text (HTML v1: body, Tables 1-6, discussion, limitations, Appendix A tables located and sampled)
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Recent work argues that world models carry linearly decodable, causally relevant representations of their environments. The authors ask the next question: is a world model merely reproducing recorded trajectories, or has it actually acquired the dynamics that generate them? They want a test that separates one from the other, and they want it to work on the latent state directly, without a decoder. The larger goal is a certification that a learned internal simulator can be trusted as a substitute for the environment it is supposed to stand in for - which is the same shape as our problem of deciding when a white-box object may be used to predict what the network will do under an intervention.

## What they studied

One model family: TD-MPC2, a decoder-free world model, trained on the DMControl pendulum swing-up task. The observation is passed through a frozen random MLP lift (3 to 64 to 32, tanh) so that a memorising baseline has structure to exploit. Thirty-five models are trained across seven hyperparameter conditions (rollout horizon, latent width, data-augmentation scale, each swept low/mid/high around a median baseline) and five seeds per condition. The unit of analysis is the model's latent trajectory over k = 5, 10 and 25 steps, plus the latent subspace that a state nudge is written into.

## How they did it

The scoring instrument is a discrete Fréchet distance between a predicted trajectory and the simulator's true trajectory, normalised into a "Fréchet Skill Score" whose denominator is the aggregate true displacement over the scoring window. The same instrument is applied to three kinds of objects so that a candidate's score has a scale: a ceiling (the true simulator, which scores 1 by construction), a floor (the strongest of four nulls), and a perturbation ladder (87 perturbation draws of the pendulum's gravity, damping and control coefficients at four magnitudes). The four nulls are a nearest-neighbour memoriser in lifted observation space, a nearest-neighbour memoriser in autoencoder latent space, an untrained copy of the architecture, and a persistence baseline that repeats the initial observation. Null selection happens on a 25% alignment slice of the replay buffer; the candidate and the chosen null are then scored on the disjoint 75% probe slice, and a paired one-sided t-test with Holm-Bonferroni correction across all 21 constraint-horizon cells decides sufficiency. A separate arm tests faithfulness: latents are mapped into the lifted observation space by an affine gauge fitted by least squares on deliberately off-support states, a PCA basis is computed from encoder Jacobians and the smallest set of components covering 99% of the variance is kept, the latent's projection onto that subspace is patched before a rollout, and the model's trajectory response is compared to the simulator's under a relative error whose do-nothing value is 1.

## What counts as an explanation here

A certification rather than a description: a verdict per constraint and horizon that the model's latent dynamics are distinguishable from memorisation and from no-motion, plus a quantitative statement of how faithfully the latent responds to a push. It is explicitly partial and comparative - the honest content is a margin over the best null, not an assertion that the governing equations were recovered. The authors say this in as many words: Fréchet distance allows temporal reparameterisation, so the result is evidence of trajectory consistency rather than of operator recovery, and similar ladder scores do not license reading a score as a recovered fraction of the dynamics.

## How they judged success

Fréchet Skill Score against a seeded floor and against a perturbation ladder; paired one-sided t-tests with Holm-Bonferroni correction over 21 tests; relative error of the steered response against the simulator response on off-support probes at three horizons; per-seed tables in the appendix. No held-out environment, no second architecture, no downstream planning metric.

## What they found

Sufficiency is horizon-dependent and was only solid at one horizon. At k = 10 all seven constraints exceeded their floors on average and six of seven remained significant after correction; the largest margins are 0.372 for the narrow-latent condition and 0.300 for the baseline, with 95% intervals that exclude zero. At k = 5 only the baseline (0.135) and the narrow-latent condition (0.267) survive correction. At k = 25 no constraint is significant, and two constraints score below the floor on the raw means, so sufficiency is unsupported there - the authors attribute the failure to the models not learning the task at that horizon, and they note the "no data augmentation" condition is the one cell at k = 10 that does not reach significance. Narrowness of the latent helps at both short horizons (0.744 at k = 5 against a 0.477 floor), and the two augmentation conditions are the weakest cells at every horizon, with the high-augmentation model falling to -0.564 at k = 25 - below all three estimated nulls.
The steering arm is the paper's own counterweight to that positive result: average relative error exceeds 1 at k = 10 and k = 25 for every one of the seven constraints, meaning the model's predicted response to a nudge is worse than predicting no response at all, and only some k = 5 cells sit below 1 (0.84 to 1.05 across constraints). Best case cells are the no-augmentation model at k = 10 (1.29) and the narrow-latent model at k = 10 (1.38); the baseline reads 1.00, 1.61 and 2.09 at the three horizons.
The ladder gives the scale for the positive half: a ground-truth simulator with its coefficients perturbed by the smallest dose (epsilon = 0.05) still scores 0.942 to 0.945, by 0.1 it scores 0.891 to 0.897, by 0.2 0.802 to 0.812, and only at 0.5 does it fall to 0.581 to 0.595 - so the trained models' 0.40 to 0.64 sits below a simulator whose dynamics were already visibly wrong.

## Limits and open problems they name

One architecture; the authors flag that a decoder-based model such as DreamerV3, trained with a reconstruction loss, might behave differently, and that the comparison would strengthen the work. The affine gauge is an accepted limitation: if TD-MPC2 encoded the pendulum nonlinearly, the linear gauge would understate it. The steering result is conditional on the gauge, the PCA subspace and the patching protocol, and it does not exclude dynamics encoded outside the selected subspace. Intervention magnitude is set by the randomly sampled off-support probe rather than fixed globally, so a null steering result and a badly scaled steering probe are not separated. Two structural facts reduce what the ARC comparison can mean, and the authors state them rather than hiding them: because the score's denominator is the distance between the true trajectory and a trajectory pinned at the initial state, the persistence null scores exactly 0 by construction, and at k = 25 the floor is that fixed zero rather than a measurement, since all three estimated nulls fell below it. The appendix tables show those nulls scoring around -1 to -3, so the k = 25 floor is the weakest possible competitor. There is also no planning or control metric anywhere, so "trajectory consistency" is not connected to whether the model is useful for action.

## What we take from it

Three things. First, a reusable scoring pattern: a candidate score is only meaningful against a ceiling, a seedable floor and a ladder of degraded-but-real references, all computed with the same instrument. The ladder is the part the folder has never had - it converts "our model scored 0.6" into "our model scored below a ground-truth object that was deliberately damaged", which is exactly the calibration a predictive white-box model needs, since a good-looking number from a proxy metric can be worse than a trivially wrong reference. Second, a warning with a sharp edge: this paper's two arms agree on nothing. A model can beat every memoriser on trajectory consistency and still respond to an intervention worse than not responding at all, and the discrepancy is not a technicality - consistency under a time-flexible distance metric is a weaker property than the one we care about. So the folder's rule is strengthened: any claim that a learned latent "carries the dynamics" is printed as a pair - consistency against nulls and response error against the true system - and the pair is what gets quoted. Third, a construction detail worth copying: null selection on one slice, scoring on a disjoint slice, with the comparison-tests corrected for the full scan. It costs nothing and it is the only thing standing between a floor and an artefact of choosing the floor after seeing the candidate.

## Tags
unit: layer | circuit | whole-model
form: probe | causal-abstraction
method: probing | patching | ablation | other
eval: prediction/simulation | benchmark | ablation
goal: evaluation-of-methods | science | understanding
