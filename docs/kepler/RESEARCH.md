# Kepler: research behind the design (Nov 24, 2026)

What exists, what each piece contributes, and where Kepler goes further. Searched 09:35–09:50 EST; sources at the
end of each section.

## 1. Knowledge inside chess networks

- **McGrath et al. 2022 (PNAS), "Acquisition of chess knowledge in AlphaZero".** Probes whether human chess
  concepts (taken from a conventional engine's evaluation terms) are linearly decodable from each layer, over the
  course of training: "what, when and where". Many human concepts can be regressed from the trained network even
  though AlphaZero never saw a human game. This is Kepler's level 1 (the concept atlas) and the source of its
  classical concept library.
- **Schut et al. 2025 (PNAS), "Bridging the human–AI knowledge gap through concept discovery and transfer in
  AlphaZero".** The closest prior work to Kepler's chess goal.
  - Concepts are vectors found by convex optimization (minimum L1 norm), with constraints contrasting MCTS rollouts
    AlphaZero chose against rollouts it rejected, at layers 19 and 23.
  - Filters:
    - **teachability:** a student network (an earlier checkpoint) must learn the concept's prototypes; 97.6% of
      concepts were filtered out here;
    - **novelty:** a concept must be reconstructed better by an SVD basis of AlphaZero-game activations than by one
      of human-game activations.
  - Four grandmasters (former or current world champions) improved on concept puzzles after studying the prototypes.
  - Stated limits: the concepts are linear; **the concept descriptions were written by the authors**; only
    AlphaZero/chess was studied.
- **Jenner et al. 2024 (NeurIPS), "Evidence of learned look-ahead".** Activation patching per square in Leela (a
  transformer treating squares as tokens): Leela represents future optimal moves internally, and those
  representations matter causally. Follow-ups: Cruz 2025 (look-ahead is context-dependent) and an extended logit
  lens study (2025: the policy forms non-smoothly across layers).
- **Sparse dictionaries on chess models.**
  - Karvonen et al. 2024 measure SAE quality on chess and Othello GPTs with board-based metrics.
  - Poupart 2024 uses contrastive SAEs on pairs of Leela trajectories.
  - "Tracing the Thought of a Grandmaster-level Chess-Playing Transformer" (2026) decomposes Leela's MLP and
    attention with sparse replacement layers plus causal interventions.
  - Chessformer (2026) trains transcoders on Maia-3 and annotates top-activating positions by hand.
- **Human-like engines.** Li et al. (CogSci 2026) probe concepts in networks that imitate human play.

Sources: [McGrath 2022](https://arxiv.org/abs/2111.09259), [Schut 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12002201),
[Jenner 2024](https://arxiv.org/abs/2406.00877), [Cruz 2025](https://arxiv.org/abs/2505.21552),
[Karvonen 2024](https://arxiv.org/abs/2408.00113), [Poupart 2024](https://arxiv.org/abs/2406.04028),
[Tracing LC0 2026](https://arxiv.org/abs/2604.10158), [Chessformer 2026](https://arxiv.org/abs/2605.19091),
[Iterative inference 2025](https://arxiv.org/abs/2508.21380).

## 2. Laws read out of trained networks

- **Cranmer et al. 2020 (NeurIPS).** Distill symbolic expressions from the parts of a trained graph network:
  - it recovers known force laws;
  - it found a new formula for dark-matter concentration;
  - the symbolic expressions generalized out of distribution better than the network itself.
  SymTorch (Tan, Soubki, Cranmer, 2026) packages this for PyTorch.
- **MIPS (Michaud et al., 2024).** Turns small RNNs into Python programs (integer autoencoder to a finite-state
  machine, then symbolic regression); it solved 32 of 62 tasks.
- **PDE discovery from data** (SINDy, PDE-FIND, Rudy et al. 2017). Sparse regression of the time derivative on a
  library of candidate terms recovers Navier–Stokes from snapshots. Kepler uses the same machinery on the
  **model's** implied dynamics, not on the data.
- **Physics-encoded networks** (PeRCNN 2021–22). Built so that terms can be read off the weights. Kepler instead reads
  terms from an unconstrained network.

Sources: [Cranmer 2020](https://arxiv.org/abs/2006.11287), [SymTorch 2026](https://arxiv.org/abs/2602.21307),
[MIPS](https://arxiv.org/abs/2402.05110), [ML for PDEs review](https://arxiv.org/pdf/2303.17078).

## 3. Law or heuristic? World models

- **Vafa et al. 2025 (ICML), "What has a foundation model found?"** Inductive-bias probes:
  - a transformer that predicts orbits well does **not** apply Newtonian mechanics;
  - symbolic regression on its implied forces gives a nonsensical force law;
  - such models build task-specific heuristics.
  Liu, Sanborn, Ganguli and Tolias (2026, "From Kepler to Newton") show which inductive biases fix this, including
  temporal locality.
- **Kang et al. 2024.** Video generation models do not discover physical laws from video: they generalize case by
  case, by similarity to training examples.
- **Lesson for Kepler:**
  - every extracted law must be tested out of distribution;
  - a model that memorizes must be reported as memorizing (the A41 rule in this repo).

Sources: [Vafa 2025](https://arxiv.org/abs/2507.06952), [Liu et al. 2026](https://arxiv.org/html/2602.06923),
[Kang 2024](https://arxiv.org/abs/2411.02385).

## 4. Physics inside video models (2026)

- **"Interpreting Physics in Video World Models" (2026).** Physical variables in large video encoders are
  distributed, not factorized like a physics engine, and peak in middle layers.
- **"Does Physics Live in the Activations?" (2026).** Ridge probes on video diffusion transformers:
  - folds are split by video, and a permutation null leaves R² at chance;
  - probes injected back can change the generated motion.
  Kepler adopts this probing protocol: splits by trajectory or game, an untrained-network control, and a
  shuffled-target null.
- **"The Invisible Hand of Physics" (2026).** Physical plausibility is linearly decodable from video diffusion
  states; corrupting blocks shows which ones matter.
- **Steering and operators.** "Causal Physics Steering" (2026) steers physics judgments with concept vectors.
  Sparse-autoencoder neural operators (2025) bring SAEs to operator learning.
- **None of these extracts the governing equation a video model implements, or the hidden state it uses to do so.**
  That is Kepler's fluid contribution.

Sources: [Interpreting physics 2026](https://arxiv.org/abs/2602.07050),
[Physics in activations 2026](https://arxiv.org/pdf/2610.03154), [Invisible hand 2026](https://arxiv.org/abs/2606.05328),
[Physics steering 2026](https://arxiv.org/pdf/2605.24322), [SAE neural operators](https://arxiv.org/pdf/2509.03738).

## 5. Automated interpretability agents

- **MAIA (Rott Shaham, Schwettmann et al. 2024).** A vision-language agent that writes programs composing
  interpretability tools, forms hypotheses and tests them on vision models. It shows confirmation bias at times.
- **Marin-Llobet and Ferrando (2026).** Two coupled agent loops on language models: explanation refinement, and
  feature discovery.
- **Lesson for Kepler:** the agent proposes, but only frozen, pre-registered tests decide. The agent's hypotheses are
  executable code that the engine scores.

Sources: [MAIA](https://arxiv.org/abs/2404.14394), [Agents 2026](https://arxiv.org/html/2605.01555).

## 6. What Kepler adds (the claim to defend)

1. **One engine, one ladder, any model.** It works on a vision-like video model and a game network:
   - (1) concept atlas;
   - (2) the model's own units;
   - (3) mechanisms tested by intervention;
   - (4) laws and strategies, with the unexplained part measured.
2. **Laws read from the model, blind.** The true parameters are sealed, and pass lines are written before the model
   exists. Each term is localized to a site and checked by intervention. For the smoke model, the hidden variable
   (velocity) is read out of the model and used to write the law.
3. **Concepts as executable rules.** Schut et al. found concepts as vectors and described them by hand, which they
   list as open. Kepler states a candidate concept as a program over the board, scores it against the network on
   held-out positions, steers the network along it, and measures novelty as the share of the network's evaluation
   that the classical library cannot explain.
4. **Teachability as an operational test.** Add the rule to a classical evaluator and measure how much closer that
   evaluator's choices and values move to the network's.
5. **An automated scientist whose hypotheses are code.** An LLM loop proposes, and Kepler's frozen tests judge.
