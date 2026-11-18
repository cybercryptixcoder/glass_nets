# Why the constructor has not gone past 32 parameters: a diagnosis from the traces

*Nov 18, 2026. Source: the saved transcripts (`run.json`: every step, the reader's notes, every submission and every
judge report) of all round-2 runs on networks above 32 parameters:
- M8,5 (112 parameters), M8,8 (136) and M16,10 (432): 4 runs each;
- lc_3_4, lc_4_5, lc_6_5 and lc_8_6 (108–656 parameters): 2 runs each.*

## Short answer

**We have tried, many times. The failure is mostly in what we asked for, not in what the reader can see.**
- The readers often understood a lot about the networks: some of it matches what ARC's own researchers found.
- But we asked for two things at once that bigger networks can't give together:
  - a program that reproduces the network almost exactly, stage by stage;
  - and one that is shorter than the network.
- The readers responded rationally. They either wrote exact programs (copies) or short programs (which failed the
  exactness bars). The judge then told them to cut numbers, and they spent their remaining steps doing that.

## The evidence

### 1. The readers find real structure

- **M16,10 (432 parameters), reader v2, run a.** Its description says units 1, 6, 7 and 8 behave like
  "relu(k·(running max − x_t))" and unit 4 like a comparison with the running second max.
  - ARC's Zihao Chen and a LessWrong commenter on ARC's AlgZoo post singled out **the same units, N1, N6, N7 and N8**,
    as the "running max / second running max" group, with N4 the positive-input unit.
  - So a cheap reader found, unaided, the same partial structure ARC's people found on the network ARC calls
    unsolved.
- **lc_6_5 (378 parameters), run b.** Its notes give a mechanistic story:
  - layer 0 is a soft "copy pointer" (position i attends to position x_i);
  - the readout's attention mixes position 0 with the orbit of node 4;
  - the network computes "the cycle length of node 4's orbit".
  - This is the reader's hypothesis and is unverified. But it is a story, not a copy.
- **lc_8_6 (656 parameters), run b.** It wrote a short "census" rule: a 2-cycle → class 1; fixed points →
  class 0; otherwise class 2.
  - The rule matches the network on **92.4%** of inputs.
  - The textbook task rule matches only about 85% (the network's own accuracy). The task-only (prior) control also
    got 85.7%.
  - So the rule is short and network-specific: it captures how this network differs from the task. The judge
    marked it "not verified", because it is under 95% and its variables don't sit at the embedding stage.
- **M8,8 (136 parameters), run v2a.** It measured that the network is "amplifying" (the recurrence's spectral radius
  is about 1.7). Changing even the least sensitive single weight to a round number drops agreement from 100% to
  5.6%.
  - It concluded that no file shorter than the network can track its exact decisions.
  - It declined to use the integer trick because it "would defeat the counter rather than explain the network".

### 2. The runs swing between "true but a copy" and "short but rejected"

Across the lc runs, submissions alternate between two kinds:
- 0 stages verified, failing behaviour or the embedding test: a real but approximate account;
- every stage verified, failing only "not an explanation (N constants)": an exact rewrite.

No submission sits in between, because the judge has no "in between". **lc_6_5 run b** made 14 submissions with
342–354 constants each, against a limit of 172. It spent about 100 steps and $0.85 shaving numbers off an exact
rewrite. Its notes show it understood the pointer-chasing mechanism early.

### 3. Our feedback steered the reader toward the wrong target

- Every judge report opens with a stage table and "FIX FIRST: <earliest unverified stage>".
- When all stages pass but the file is long, it says "Now compress (see the compression line)".
- So the instruction the reader actually receives is "make this exact, then cut numbers". It follows that
  instruction. Its knowledge (the pointer story, the census rule, the running-max units) stays in its notes and
  descriptions, which the judge never tests.

### 4. Investigation is cut into 6-step slices

- The prompt requires a first submission within 6 steps (8 for language models), then forces a submission after
  every 6 steps without one. This rule came from the first mod-add runs (E2/E3), where the reader never submitted.
- On bigger networks, the traces show a submission about every 7 steps from start to finish.
- Whether this slows understanding is **not established**: notes still accumulate across slices. It is a candidate
  cause, to be tested, not a finding.

## What this means

1. **Exact reproduction is the wrong target for networks with a messy remainder.** ARC says the same about M16,10:
   its approximation error is "not handled". Gross et al. call it compounding structureless noise. An explanation
   of a 400-parameter network will be approximate. The question is how good, measured.
2. **The reader's capability is not yet the binding limit.** It is not shown to be sufficient either. We can't
   measure capability with a target the reader can only meet by copying.
3. **The catalog is empty for this reason.** Only the two smallest networks admit an explanation that is exact and
   short at the same time.

The intent changes this leads to are in `docs/glass/INTENT.md`.
