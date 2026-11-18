# Constructor v2: the plan

Branch `constructor-v2`, cut from `constructor-attempt` at `d6b706ff` (Nov 17, 2026). That branch is left
untouched, so anything here can be dropped by going back to it.

## The job

Build **whole-model** constructions of real trained networks, each with an honest, measured certificate, and
grow a public library of them. A construction counts when it stands in for the **entire** network, not one
operation inside it. The headline number is **coverage**: how much of the network's behaviour, and of its inside
computation, the construction accounts for, with the leftover listed.

The first targets are ARC's AlgZoo (github.com/alignment-research-center/alg-zoo). These are tiny RNNs and
transformers trained on algorithmic tasks (8 to 1,408 parameters), and they are public. ARC reports the largest
model it fully understands has 10 parameters. A 432-parameter 2nd-argmax RNN is unsolved, and ARC has posted an
open challenge on it. These are whole models, small enough to finish, and climbing their size ladder is a
direct, checkable measure of progress.

## Rule for every item below

Each item names the job it serves and the specific failure it catches, and has a "done when" condition. Anything
that can't say which failure it catches is not built. Nothing that already exists is rewritten unless an item
needs it.

## Items

| # | Item | Why (the failure it catches or the job it serves) | Done when | Cost |
|---|---|---|---|---|
| 0 | **Baseline check**: re-judge the existing verified constructions on this branch | Catches regressions from the items below | Every existing verdict is reproduced exactly; re-run after each judge change | CPU, minutes |
| 1 | **AlgZoo targets**: a `rnn_relu` kind in `nets.py` (numpy forward, sites = hidden state after each step, one stage per step), a builder `lab/targets/build_algzoo.py`, inputs drawn as in AlgZoo's training, task labels saved for item 3 | The job: real whole-model targets with an open challenge. Without this the constructor has nothing new to run on | The numpy forward matches the PyTorch checkpoint to 1e-6 on 10k inputs; shuffled/untrained/permuted controls work; the reader and judge run on it unchanged | CPU |
| 2 | **Joint substitution** in the judge: replace every named site at once with the construction's predicted activations and run the network | A whole-model claim: today each site is tested alone, so sites that pass one at a time could still fail to work together | Reported as "joint substitution match", with a pass bar; existing complete constructions are re-scored (item 0) | CPU |
| 3 | **Worst case and the network's own mistakes**: (a) behaviour per output class and on the lowest-margin decile, with the worst slice reported; (b) when task labels exist, does the construction reproduce the network's errors?; (c) a short search for inputs where construction and network disagree most (continuous inputs only), with the found inputs saved | An average of 97% can hide a systematic hole. An explanation of how the network succeeds isn't one of how it fails | Report shows the worst slice, error agreement and the search's worst inputs, with a pass bar on worst slice and error agreement | CPU |
| 4 | **Coverage and the unexplained list** in the report: per stage verified or not, joint coverage, and a plain list of what isn't explained | The library entry must say what it doesn't cover. This is our "error nodes" | Every report ends with a coverage line and an unexplained list | CPU, small |
| 5 | **AlgZoo ladder runs**: blinded constructor runs on 2nd argmax, smallest first (the 10- and 32-parameter models ARC explained, then up to the 432-parameter one), each with the prior-only control and one shuffled-weights control | The test of the whole system. The small ones calibrate (we should solve what ARC solved); the 432-parameter one is the open frontier | Each run's verdict and coverage is in `docs/RESULTS_V2.md`; stop climbing at the first model the reader can't complete and record why | Luminal API (DeepSeek Flash). **Needs Shreyas's OK before starting** |

Order: 0 → 1 → 2 → 3 → 4, then 5. Items 2–4 are judge changes and are re-checked with item 0 after each one.

## Deliberately not built now

| Not built | Why not now | When it becomes needed |
|---|---|---|
| The public library page | Nothing to show until item 5 produces whole-model entries | After the first complete AlgZoo constructions |
| ARC's "mechanistic estimate beats sampling" test | Needs constructions written to be analysed without sampling; no evidence yet that the reader can produce whole models at all | After item 5 shows complete constructions |
| Required rival explanations | Matters for discovery claims (like the month line vs circle), not for the ladder | When a run makes a new mechanistic claim |
| Queryable constructions ("when does output X happen?") | Is a use of a finished construction; needs finished constructions first | After the ladder |
| Weight-space tools (parameter decomposition, SPD) in the reader | Only if item 5 shows the reader fails because activations alone aren't enough | If item 5's failure analysis points there |
| Edits back into weights (Goal 1 link) | A separate job | After the ladder |

## Records

- Results: `docs/RESULTS_V2.md` (created with the first result).
- Every change is committed and pushed to `constructor-v2` as it's made; docs are updated in the same step.
