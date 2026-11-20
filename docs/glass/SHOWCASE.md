# Glass Networks: blind, automatic recovery of a network's algorithm (one-page showcase)

*Nov 20, 2026. Every number below is from a test whose pass rule was written down before the network it judges existed (ledger: `docs/glass/CHANGES.md`, entries A37, A41, A42, A44).*

## The claim

Give the pipeline a **freshly trained 227k-parameter transformer** doing modular arithmetic (a + b or a · b mod 113), one it has never seen. With no human in the loop, it produces a catalogue that states:

- **the number system the network computes in:** the ordinary number line for addition, discrete-logarithm order for multiplication. It chooses between them itself.
- **the frequencies it uses,** mechanism by mechanism.
- **what each output-side mechanism does:** "reads frequency k from the inputs → writes a frequency-k wave over the answers".
- **which parts it can vouch for and which are memorized lists.**

Each answer is scored blind against an answer key computed separately from the network's embedding weights.

## Results on networks never used for design

| fresh network | answer key (from the weights) | rules on the key frequencies | write back the frequency they read |
|---|---|---|---|
| multiplication, seed 1 | 19, 14, 20, 56, 38 (discrete-log order) | **95%** (77/81); same frequencies on the number line: 7% | **60%** (21/35) |
| addition, seed 3 | 14, 13, 51, 19, 11 | **92%** (55/60) | **71%** (15/21) |
| addition, seed 4 | 37, 8, 43, 52, 34 | **92%** (44/48) | **71%** (17/24) |

- **Pass rules:** at least 80% with at least 20 rules (chance about 5%), and more than half writing what they read (chance about 20%). All three networks pass both.
- **The memorization labels matter:** counting every trusted rule, including ones the system labels *memorized*, the scores fall to 76–79%.
- **Honesty about memorization:** a network with the same architecture trained on *random labels* gets no invented algorithm. In its MLP and unembedding, 187 of 193 mechanisms get no trusted rule, and 96% of its trusted mechanisms overall are labelled "lookup".

## How it works

1. **Decompose.** Each weight matrix is split exactly into rank-1 pieces using stochastic parameter decomposition (SPD; Bushnaq, Braun & Sharkey 2025), at two sparsity levels. The sparsest level that still reproduces the network on training data is kept.
2. **Explain.** For each piece, fit a depth-4 decision tree over task variables: a, b, the answer, and their positions on a circle at each frequency, in both bases. The tree is the rule for when the piece is needed.
3. **Label.** A rule counts as **rule** only if it also predicts the piece on *token values it never saw while fitting*. Otherwise it counts as **lookup** (memorized cases).
4. **Read the write side.** Take the Fourier spectrum of each output-side piece's effect on the answer logits.
5. **Certify.** Rebuild and gate the network from its pieces and count how many of the real network's decisions survive.

## What this is, and what it isn't (prior work)

- **The mechanisms themselves are published.**
  - Addition: the "clock" algorithm (Nanda et al., ICLR 2023).
  - Multiplication: the discrete-log version (Chughtai et al., 2023; Nguyen, "The Discrete-Log Clock", ICML 2026).
- **So this is a validation, not a discovery.** An automated pipeline recovers these known algorithms blind, on naturally trained networks: realistic weights, not hand-built or semi-synthetic ones as in InterpBench or Tracr. It is scored against keys fixed in advance, and it says when it can't explain something.
- **What the pipeline is told:** that frequencies might matter. It is offered 440 frequency features in two bases and must pick the right few itself. A network family with unknown structure would need a vocabulary of its task variables.

## Also shown

- **Small recurrent networks** (up to about 1.4k parameters, held out): 5 of 6 pass every certificate check.
- **A published challenge network** (the 2nd-largest-number network): one mechanism turns "largest" into "2nd largest". Remove it and the network returns the max.

## Limits and next step

- **No 227k network passes the full certificate yet.** SPD drops attention query/key weights that are spread over many pieces: its random partial ablations never switch them all off together.
  - Our three fixes repaired attention but disturbed planted known answers at some seeds.
  - The SPD authors' successor method, VPD (Oct 2026), replaces random ablations with adversarially chosen ones and counts pieces that are only removable together. That targets exactly this gap, so adopting its adversarial masks is the planned fix.
- **"What it computes" read-outs exist only for modular arithmetic.** For the recurrent networks, the rules mostly say *when* a piece acts.
- **The memorization test's pre-registered count was a single mechanism.** The substance above is solid; the formal count is thin.

## Run it yourself

```
bash scripts/glass/showcase_blind.sh mul 7     # or: add 7. Trains a fresh network, then decomposes, builds the atlas and scores it (GPU, about 1 h)
python3 lab/server.py                           # then open http://localhost:8765/catalog?t=modmul_p113_blind7
```

**Catalogue pages worth opening:**

| page | what it shows |
|---|---|
| `modmul_p113_s1` | discrete-log frequencies, found unprompted |
| `modadd_p113_s4` | number-line frequencies and their doubles |
| `modrand_p113` | the memorizer: lookups, no algorithm |
| `zoo_2nd_argmax_16_10` | the 2nd-largest mechanism in the challenge network |
