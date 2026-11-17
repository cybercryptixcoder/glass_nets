# Construction run `20261117-053452-314f` on `gpt2_month_rollover`

*Written by `lab/results.py` from the files in this folder. Network: gpt2_month_rollover (124,439,808 weights). Known answer (never shown to a blinded reader): GPT-2 small mostly continues with the same year when month2 comes after month1 in the calendar, and with the next year when it comes before; the probability of the next year grows with the forward distance from month1 to month2..*

## The run

- Reader: deepseek-ai/DeepSeek-V4.1-Flash, reasoning effort high; step budget 150; used 150 steps, 25 submissions, 3725 s, $0.784.
- Weights: original. Reader was blinded (not told what the network does).
- Title: blinded run A. Status: done.

## Result

**1 of 5 stages verified** (every site the delivered construction names passes).

Run in place of the network, the construction makes the same decision on 100.0% of 400 held-out inputs (a constant output: 80.3%). Construction length: 4,026 characters.

## The mechanism, in the reader's words

Task: predict the year token after '<|endoftext|> m1 YYYY - m2'. The network commits to an ANSWER YEAR ya = year + (T>0), T = logit(next)-logit(same) = C1[m1][m2] + C1b[year] (an additive month-order table: positive when m2 is not later than m1, so the range wraps into the next year; plus a small year offset); 'previous year' never wins (margin logit(prev)-logit(same) = C2[m1][m2]+C2b[year] ~ -1.5). The final residual at the end position holds the IDENTITY OF THE ANSWER YEAR, a one-hot over 1980..2000: late attention copies the year token from position 2 into the end position and shifts it by the margin. The three answer logits are a read-out of that identity against the three candidate year tokens of the prompt's year: replacing the held answer year by another one shifts the logit of candidate year y by H(|ya-y|) - H(|ya'-y|), where H falls off with the year distance; on top of that sits the prompt's own preference.

## Stages

| Stage | Status | Sites |
|---|---|---|
| embedding | not explained | resid0 (token + position embeddings) |
| layers 0-2 | not explained | resid1, resid2, resid3 |
| layers 3-5 | not explained | resid4, resid5, resid6 |
| layers 6-8 | not explained | resid7, resid8, resid9 |
| layers 9-11 | verified | cut12 |

## Where each variable lives

| Variable | Dimensions | Sites |
|---|---|---|
| `ans_year` | 21 | resid12@end |

## Site tests

Substitute: decisions kept when the site is replaced by the construction's prediction. Remove: decisions kept when the named subspace is removed. Mean-ablate: decisions kept when the whole site is replaced by its mean.

| Site | Named dims / PCs that suffice | Variance explained | Substitute | Remove | Mean-ablate | Pass |
|---|---|---|---|---|---|---|
| `cut12` | 20 / 646 | 0.48 | 100.0% | 49.4% | 51.7% | yes |

## Intervention tests

Share of cases where the predictor named the network's new decision, among cases where the decision changed ("no change" scores 0 on these by definition).

| Test | Variable | Site | Construction | Source output | Attribution patching | Pass |
|---|---|---|---|---|---|---|
| interchange | `ans_year` | `cut12` | 93.0% | 43.9% | 100.0% | yes |

## Files

- `construction.py`: the delivered construction (runnable: `model(x, iv)` and `ALIGN`).
- `report.txt`, `report.json`: the judge's report on it.
- `submission_NN.py`, `submission_NN.json`: every construction the reader submitted, with its report.
- `run.json`: the whole run: prompt, every step (reasoning, code, outputs), usage.
- `predict.json` (if present): the held-out intervention benchmark.

## Reproduce

```bash
python3 lab/judge.py gpt2_month_rollover lab/constructions/gpt2_month_rollover/20261117-053452-314f/construction.py        # re-run every test on the delivered construction
python3 lab/predict.py gpt2_month_rollover lab/constructions/gpt2_month_rollover/20261117-053452-314f/construction.py              # held-out interventions against the baselines
python3 lab/constructor.py gpt2_month_rollover --steps 150   # a fresh run with the same settings (needs lab/.env)
```
