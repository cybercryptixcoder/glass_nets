# Construction run `20261114-130627-38cc` on `gpt2_be_verb`

*Written by `lab/results.py` from the files in this folder. Network: gpt2_be_verb (124,439,808 weights). Known answer (never shown to a blinded reader): GPT-2 small prefers the form of 'to be' that agrees with the noun's number and the cue's tense: is / are / was / were..*

## The run

- Reader: deepseek-ai/DeepSeek-V4.1-Flash, reasoning effort high; step budget 80; used 4 steps, 1 submissions, 1458 s, $0.091.
- Weights: original. Reader was told the task, had NO weights and could not run the network (prior-only control), and had a single submission with no report back.
- Title: prior-only control 1. Status: done.

## Result

**1 of 5 stages verified**.

Run in place of the network, the construction makes the same decision on 100.0% of 360 held-out inputs (a constant output: 20.0%). Construction length: 2,750 characters.

## The mechanism, in the reader's words

The model uses the cue word to determine tense and the noun to determine number. At resid0 the cue identity is at position 1 and the noun identity at position 4. By layer 3 the cue information is available at the last position (4); the noun identity is also there. Layers 0-5 combine them into a 4-way answer code (present/past x singular/plural). The answer code is carried at the last position through layers 6-11 and read by the unembedding. We represent each stage's state with one-hot identities: cue (12), noun (60), and answer (4). The answer is recomputed from cue and noun at early stages, and the answer code is carried forward once formed. Later stages keep the answer code as the primary variable.

## Stages

| Stage | Status | Sites |
|---|---|---|
| embedding | named, failing | cut0 |
| layers 0-2 | named, failing | cut3 |
| layers 3-5 | named, failing | cut6 |
| layers 6-8 | named, failing | cut9 |
| layers 9-11 | verified | cut12 |

## Where each variable lives

| Variable | Dimensions | Sites |
|---|---|---|
| `cue_e_1` | 12 | resid0@cue |
| `noun_e_4` | 60 | resid0@noun |
| `cue_3_1` | 12 | resid3@cue |
| `cue_3_4` | 12 | resid3@noun |
| `noun_3_4` | 60 | resid3@noun |
| `ans_3_4` | 4 | resid3@noun |
| `cue_6_1` | 12 | resid6@cue |
| `cue_6_4` | 12 | resid6@noun |
| `noun_6_4` | 60 | resid6@noun |
| `ans_6_4` | 4 | resid6@noun |
| `cue_9_1` | 12 | resid9@cue |
| `cue_9_4` | 12 | resid9@noun |
| `noun_9_4` | 60 | resid9@noun |
| `ans_9_4` | 4 | resid9@noun |
| `ans_12_4` | 4 | resid12@noun |

## Site tests

Substitute: decisions kept when the site is replaced by the construction's prediction. Remove: decisions kept when the named subspace is removed. Mean-ablate: decisions kept when the whole site is replaced by its mean.

| Site | Named dims / PCs that suffice | Variance explained | Substitute | Remove | Mean-ablate | Pass |
|---|---|---|---|---|---|---|
| `cut0` | 40 / 36 | 0.39 | 51.7% | 50.0% | 20.0% | no |
| `cut3` | 52 / 34 | 0.41 | 95.0% | 40.8% | 20.0% | no |
| `cut6` | 52 / 29 | 0.46 | 83.3% | 62.2% | 36.7% | no |
| `cut9` | 52 / 28 | 0.50 | 83.3% | 70.3% | 36.7% | no |
| `cut12` | 3 / 12 | 0.09 | 100.0% | 42.8% | 20.0% | yes |

## Intervention tests

Share of cases where the predictor named the network's new decision, among cases where the decision changed ("no change" scores 0 on these by definition).

| Test | Variable | Site | Construction | Source output | Attribution patching | Pass |
|---|---|---|---|---|---|---|
| interchange | `all variables` | `cut0` | 96.3% | 91.7% | 0.9% | yes |
| partial | `cue_e_1` | `cut0` | 100.0% | 49.0% | n/a | yes |
| partial | `noun_e_4` | `cut0` | 100.0% | 43.7% | n/a | yes |
| interchange | `all variables` | `cut3` | 98.3% | 98.0% | 3.0% | yes |
| partial | `cue_3_1` | `cut3` | 100.0% | 47.0% | n/a | yes |
| partial | `cue_3_4` | `cut3` | 49.7% | 22.3% | n/a | no |
| partial | `noun_3_4` | `cut3` | 49.0% | 22.3% | n/a | no |
| partial | `ans_3_4` | `cut3` | 0.0% | 49.3% | n/a | no |
| interchange | `all variables` | `cut6` | 75.6% | 65.7% | 2.3% | no |
| partial | `cue_6_1` | `cut6` | 100.0% | 26.3% | n/a | yes |
| partial | `cue_6_4` | `cut6` | 49.7% | 22.3% | n/a | no |
| partial | `noun_6_4` | `cut6` | 49.0% | 22.3% | n/a | no |
| partial | `ans_6_4` | `cut6` | 54.0% | 49.3% | n/a | yes |
| interchange | `all variables` | `cut9` | 84.2% | 73.7% | 23.0% | no |
| partial | `cue_9_1` | `cut9` | 0.0% | 22.7% | n/a | yes |
| partial | `cue_9_4` | `cut9` | 0.0% | 23.0% | n/a | yes |
| partial | `noun_9_4` | `cut9` | 0.0% | 22.7% | n/a | yes |
| partial | `ans_9_4` | `cut9` | 58.2% | 52.0% | n/a | yes |
| interchange | `ans_12_4` | `cut12` | 99.6% | 99.7% | 99.6% | yes |

## What fails

- substitution at cut0: 51.7% (need 90%)
- partial of cue_3_4 at cut3: the network's output changed in only 0.0% of cases, but the construction predicted changes: at this site the network's output does not depend on this variable
- partial of noun_3_4 at cut3: the network's output changed in only 0.0% of cases, but the construction predicted changes: at this site the network's output does not depend on this variable
- partial of ans_3_4 at cut3: prediction right in 0.0% of the cases where the network's output changed (need 50%; guessing 'no change' gets 0% of these)
- substitution at cut6: 83.3% (need 90%)
- interchange of all variables at cut6: prediction right in 75.6% of the cases where the network's output changed (need 90%)
- partial of cue_6_4 at cut6: the network's output changed in only 0.0% of cases, but the construction predicted changes: at this site the network's output does not depend on this variable
- partial of noun_6_4 at cut6: the network's output changed in only 0.0% of cases, but the construction predicted changes: at this site the network's output does not depend on this variable
- substitution at cut9: 83.3% (need 90%)
- removal at cut9: removing the named subspace leaves 70.3% of decisions, while mean-ablating the whole site leaves 36.7%: what you named is not what the output depends on here
- interchange of all variables at cut9: prediction right in 84.2% of the cases where the network's output changed (need 90%)

## Files

- `construction.py`: the delivered construction (runnable: `model(x, iv)` and `ALIGN`).
- `report.txt`, `report.json`: the judge's report on it.
- `submission_NN.py`, `submission_NN.json`: every construction the reader submitted, with its report.
- `run.json`: the whole run: prompt, every step (reasoning, code, outputs), usage.
- `predict.json` (if present): the held-out intervention benchmark.

## Reproduce

```bash
python3 lab/judge.py gpt2_be_verb lab/constructions/gpt2_be_verb/20261114-130627-38cc/construction.py        # re-run every test on the delivered construction
python3 lab/predict.py gpt2_be_verb lab/constructions/gpt2_be_verb/20261114-130627-38cc/construction.py              # held-out interventions against the baselines
python3 lab/constructor.py gpt2_be_verb --told --no-weights --one-shot --steps 80   # a fresh run with the same settings (needs lab/.env)
```
