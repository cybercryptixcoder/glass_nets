# Construction run `20261117-053455-87be` on `gpt2_month_rollover`

*Written by `lab/results.py` from the files in this folder. Network: gpt2_month_rollover (124,439,808 weights). Known answer (never shown to a blinded reader): GPT-2 small mostly continues with the same year when month2 comes after month1 in the calendar, and with the next year when it comes before; the probability of the next year grows with the forward distance from month1 to month2..*

## The run

- Reader: deepseek-ai/DeepSeek-V4.1-Flash, reasoning effort high; step budget 80; used 3 steps, 1 submissions, 429 s, $0.053.
- Weights: original. Reader was told the task, had NO weights and could not run the network (prior-only control), and had a single submission with no report back.
- Title: prior-only control 1. Status: done.

## Result

**0 of 5 stages verified**.

Run in place of the network, the construction makes the same decision on 81.2% of 400 held-out inputs (a constant output: 64.0%). Construction length: 3,878 characters.

## The mechanism, in the reader's words

GPT-2 reads the date-range prompt as a calendar comparison. The month identities m1 and m2 and the input year are carried in the residual stream at their token positions through the early blocks (resid0, resid3). By block 6 the end position also carries the relation label: same year when m2 >= m1, next year when m2 < m1. Layers 6-8 combine that relation with the input year into an output-year identity at the final position (resid9); layers 9-11 pass that output year to the unembedding (resid12). The answer logit for the same-year token is highest when the output year equals the prompt year, for the next-year token when it is prompt year + 1, and for the previous-year token when it is prompt year - 1. Variables are categorical one-hot identities of the month, year, relation, and resulting output year; each value occurs many times in the prompt set.

## Stages

| Stage | Status | Sites |
|---|---|---|
| embedding | named, failing | cut0 |
| layers 0-2 | named, failing | cut3 |
| layers 3-5 | named, failing | cut6 |
| layers 6-8 | named, failing | cut9 |
| layers 9-11 | named, failing | cut12 |

## Where each variable lives

| Variable | Dimensions | Sites |
|---|---|---|
| `m1_0` | 12 | resid0@m1, resid0@m2 |
| `y_0` | 20 | resid0@m1, resid0@m2 |
| `m2_0` | 12 | resid0@m1, resid0@m2 |
| `m1_3` | 12 | resid3@m1, resid3@year, resid3@m2 |
| `y_3` | 20 | resid3@m1, resid3@year, resid3@m2 |
| `m2_3` | 12 | resid3@m1, resid3@year, resid3@m2 |
| `m1_6` | 12 | resid6@m1, resid6@year, resid6@m2 |
| `y_6` | 20 | resid6@m1, resid6@year, resid6@m2 |
| `m2_6` | 12 | resid6@m1, resid6@year, resid6@m2 |
| `rel_6` | 3 | resid6@m1, resid6@year, resid6@m2 |
| `ans_9` | 21 | resid9@m1, resid9@year, resid9@m2 |
| `m1_9` | 12 | resid9@m1, resid9@year, resid9@m2 |
| `y_9` | 20 | resid9@m1, resid9@year, resid9@m2 |
| `m2_9` | 12 | resid9@m1, resid9@year, resid9@m2 |
| `y_12` | 20 | resid12@m2 |
| `ans_12` | 21 | resid12@m2 |

## Site tests

Substitute: decisions kept when the site is replaced by the construction's prediction. Remove: decisions kept when the named subspace is removed. Mean-ablate: decisions kept when the whole site is replaced by its mean.

| Site | Named dims / PCs that suffice | Variance explained | Substitute | Remove | Mean-ablate | Pass |
|---|---|---|---|---|---|---|
| `cut0` | 22 / 20 | 1.00 | 31.5% | 64.0% | 64.0% | no |
| `cut3` | 31 / 887 | 0.69 | 31.2% | 57.2% | 36.8% | no |
| `cut6` | 32 / 887 | 0.71 | 28.5% | 57.5% | 36.5% | no |
| `cut9` | 41 / 887 | 0.38 | 54.0% | 58.5% | 33.2% | no |
| `cut12` | 19 / 646 | -0.09 | 66.0% | 66.8% | 36.8% | no |

## Intervention tests

Share of cases where the predictor named the network's new decision, among cases where the decision changed ("no change" scores 0 on these by definition).

| Test | Variable | Site | Construction | Source output | Attribution patching | Pass |
|---|---|---|---|---|---|---|
| interchange | `all variables` | `cut0` | 43.0% | 74.3% | 73.4% | no |
| partial | `m1_0` | `cut0` | 72.9% | 62.0% | n/a | yes |
| partial | `y_0` | `cut0` | 36.7% | 53.3% | n/a | no |
| partial | `m2_0` | `cut0` | 80.0% | 66.0% | n/a | yes |
| interchange | `all variables` | `cut3` | 28.1% | 71.0% | 68.5% | no |
| partial | `m1_3` | `cut3` | 75.7% | 62.7% | n/a | yes |
| partial | `y_3` | `cut3` | 12.9% | 56.3% | n/a | no |
| partial | `m2_3` | `cut3` | 87.8% | 61.7% | n/a | yes |
| interchange | `all variables` | `cut6` | 27.1% | 68.3% | 75.3% | no |
| partial | `m1_6` | `cut6` | 76.7% | 62.7% | n/a | yes |
| partial | `y_6` | `cut6` | 22.5% | 55.3% | n/a | no |
| partial | `m2_6` | `cut6` | 73.0% | 61.0% | n/a | yes |
| partial | `rel_6` | `cut6` | 81.8% | 58.7% | n/a | yes |
| interchange | `all variables` | `cut9` | 32.2% | 65.0% | 74.7% | no |
| partial | `ans_9` | `cut9` | 18.8% | 56.0% | n/a | no |
| partial | `m1_9` | `cut9` | 17.5% | 63.0% | n/a | no |
| partial | `y_9` | `cut9` | 45.5% | 53.7% | n/a | no |
| partial | `m2_9` | `cut9` | 17.9% | 62.3% | n/a | no |
| interchange | `all variables` | `cut12` | 50.0% | 42.3% | 93.0% | no |
| partial | `y_12` | `cut12` | 62.1% | 54.7% | n/a | yes |
| partial | `ans_12` | `cut12` | 54.3% | 53.3% | n/a | yes |

## What fails

- behaviour: the construction's output agrees with the network on 81.2% of inputs (need 95%)
- substitution at cut0: 31.5% (need 90%)
- interchange of all variables at cut0: prediction right in 43.0% of the cases where the network's output changed (need 90%)
- partial of y_0 at cut0: the network's output changed in only 0.0% of cases, but the construction predicted changes: at this site the network's output does not depend on this variable
- substitution at cut3: 31.2% (need 90%)
- interchange of all variables at cut3: prediction right in 28.1% of the cases where the network's output changed (need 90%)
- partial of y_3 at cut3: prediction right in 12.9% of the cases where the network's output changed (need 50%; guessing 'no change' gets 0% of these)
- substitution at cut6: 28.5% (need 90%)
- interchange of all variables at cut6: prediction right in 27.1% of the cases where the network's output changed (need 90%)
- partial of y_6 at cut6: prediction right in 22.5% of the cases where the network's output changed (need 50%; guessing 'no change' gets 0% of these)
- substitution at cut9: 54.0% (need 90%)
- interchange of all variables at cut9: prediction right in 32.2% of the cases where the network's output changed (need 90%)
- partial of ans_9 at cut9: prediction right in 18.8% of the cases where the network's output changed (need 50%; guessing 'no change' gets 0% of these)
- partial of m1_9 at cut9: prediction right in 17.5% of the cases where the network's output changed (need 50%; guessing 'no change' gets 0% of these)
- partial of y_9 at cut9: prediction right in 45.5% of the cases where the network's output changed (need 50%; guessing 'no change' gets 0% of these)
- partial of m2_9 at cut9: prediction right in 17.9% of the cases where the network's output changed (need 50%; guessing 'no change' gets 0% of these)
- substitution at cut12: 66.0% (need 90%)
- interchange of all variables at cut12: prediction right in 50.0% of the cases where the network's output changed (need 90%)

## Files

- `construction.py`: the delivered construction (runnable: `model(x, iv)` and `ALIGN`).
- `report.txt`, `report.json`: the judge's report on it.
- `submission_NN.py`, `submission_NN.json`: every construction the reader submitted, with its report.
- `run.json`: the whole run: prompt, every step (reasoning, code, outputs), usage.
- `predict.json` (if present): the held-out intervention benchmark.

## Reproduce

```bash
python3 lab/judge.py gpt2_month_rollover lab/constructions/gpt2_month_rollover/20261117-053455-87be/construction.py        # re-run every test on the delivered construction
python3 lab/predict.py gpt2_month_rollover lab/constructions/gpt2_month_rollover/20261117-053455-87be/construction.py              # held-out interventions against the baselines
python3 lab/constructor.py gpt2_month_rollover --told --no-weights --one-shot --steps 80   # a fresh run with the same settings (needs lab/.env)
```
