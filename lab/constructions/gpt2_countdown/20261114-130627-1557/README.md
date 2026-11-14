# Construction run `20261114-130627-1557` on `gpt2_countdown`

*Written by `lab/results.py` from the files in this folder. Network: gpt2_countdown (124,439,808 weights). Known answer (never shown to a blinded reader): GPT-2 small continues the run: n4 + 1 after an increasing run, n4 - 1 after a decreasing run..*

## The run

- Reader: deepseek-ai/DeepSeek-V4.1-Flash, reasoning effort high; step budget 80; used 2 steps, 1 submissions, 1371 s, $0.042.
- Weights: original. Reader was told the task, had NO weights and could not run the network (prior-only control), and had a single submission with no report back.
- Title: prior-only control 1. Status: done.

## Result

**1 of 5 stages verified**.

Run in place of the network, the construction makes the same decision on 100.0% of 400 held-out inputs (a constant output: 1.5%). Construction length: 5,811 characters.

## The mechanism, in the reader's words

GPT-2 small continues a run of consecutive integers: after an increasing run it emits n4+1, after a decreasing run n4-1. The only two things the model needs are WHICH number comes last (n4) and WHICH WAY the run goes (sign of n4-n1, i.e. of the sum of the three consecutive differences). Both are number-token identities, so every variable below is a one-hot identity code (one-hot over the 362 number tokens ' 0'...' 361', or over the 2 directions, or over the 4 prefixes) -- the form the residual stream holds a token in at its own position. STAGE embedding (resid0): each of the four number positions carries its own token; the two prefix positions carry the prefix identity. STAGE layers 0-2 (cut resid3): the number identities stay on their own positions; the commas c1,c2,c3 carry the previous number (previous-token copying); comparing the numbers gives the run direction, which by layer 3 sits on the last position (end, the final comma). STAGE layers 3-5 (cut resid6): the last number is copied to the end position (attention from the final comma back to n4), so resid6@end holds (n4, direction). STAGE layers 6-8 (cut resid9): the successor/predecessor map is applied at the end position, so resid9@end holds n4+direction = the answer's number identity, alongside n4 and the direction. STAGE layers 9-11 (cut resid12): the end position holds only the answer identity; ln_f + W_U read out ' n4+direction'. The model below mirrors that chain: every later stage's variables are computed from the values iv returns for the previous stage, so an intervention on any variable (a patched n4, a patched direction, a patched answer code) propagates exactly the way the network propagates it, and the output logits are those of the answer identity the final stage holds.

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
| `emb_n1` | 362 | resid0@n1 |
| `emb_n2` | 362 | resid0@n2 |
| `emb_n3` | 362 | resid0@n3 |
| `emb_n4` | 362 | resid0@n4 |
| `emb_pfx1` | 4 | resid0@1 |
| `emb_pfx2` | 4 | (not aligned) |
| `L3_n1` | 362 | resid3@n1 |
| `L3_n2` | 362 | resid3@n2 |
| `L3_n3` | 362 | resid3@n3 |
| `L3_n4` | 362 | resid3@n4 |
| `L3_c1_n1` | 362 | resid3@c1 |
| `L3_c2_n2` | 362 | resid3@c2 |
| `L3_c3_n3` | 362 | resid3@c3 |
| `L3_end_dir` | 2 | resid3@end |
| `L6_n1` | 362 | resid6@n1 |
| `L6_n2` | 362 | resid6@n2 |
| `L6_n3` | 362 | resid6@n3 |
| `L6_n4` | 362 | resid6@n4 |
| `L6_c1_n1` | 362 | resid6@c1 |
| `L6_c2_n2` | 362 | resid6@c2 |
| `L6_c3_n3` | 362 | resid6@c3 |
| `L6_end_n4` | 362 | resid6@end |
| `L6_end_dir` | 2 | resid6@end |
| `L9_n1` | 362 | resid9@n1 |
| `L9_n2` | 362 | resid9@n2 |
| `L9_n3` | 362 | resid9@n3 |
| `L9_n4` | 362 | resid9@n4 |
| `L9_c1_n1` | 362 | resid9@c1 |
| `L9_c2_n2` | 362 | resid9@c2 |
| `L9_c3_n3` | 362 | resid9@c3 |
| `L9_end_n4` | 362 | resid9@end |
| `L9_end_dir` | 2 | resid9@end |
| `L9_end_ans` | 362 | resid9@end |
| `L12_end_ans` | 362 | resid12@end |

## Site tests

Substitute: decisions kept when the site is replaced by the construction's prediction. Remove: decisions kept when the named subspace is removed. Mean-ablate: decisions kept when the whole site is replaced by its mean.

| Site | Named dims / PCs that suffice | Variance explained | Substitute | Remove | Mean-ablate | Pass |
|---|---|---|---|---|---|---|
| `cut0` | 410 / 53 | 1.00 | 99.0% | 1.0% | 1.0% | no |
| `cut3` | 716 / 29 | 0.89 | 99.0% | 1.0% | 1.0% | no |
| `cut6` | 815 / 23 | 0.57 | 53.5% | 0.2% | 0.5% | no |
| `cut9` | 901 / 20 | 0.41 | 96.0% | 0.2% | 0.2% | no |
| `cut12` | 100 / 18 | 0.13 | 98.0% | 1.5% | 1.5% | yes |

## Intervention tests

Share of cases where the predictor named the network's new decision, among cases where the decision changed ("no change" scores 0 on these by definition).

| Test | Variable | Site | Construction | Source output | Attribution patching | Pass |
|---|---|---|---|---|---|---|
| interchange | `all variables` | `cut0` | 100.0% | 100.0% | 6.0% | yes |
| partial | `emb_n1` | `cut0` | 0.0% | 0.7% | n/a | no |
| partial | `emb_n2` | `cut0` | 0.0% | 0.7% | n/a | no |
| partial | `emb_n3` | `cut0` | 0.0% | 5.7% | n/a | no |
| partial | `emb_n4` | `cut0` | 40.7% | 40.3% | n/a | no |
| partial | `emb_pfx1` | `cut0` | 100.0% | 0.7% | n/a | yes |
| interchange | `all variables` | `cut3` | 100.0% | 100.0% | 18.8% | yes |
| partial | `L3_n1` | `cut3` | 0.0% | 0.7% | n/a | no |
| partial | `L3_n2` | `cut3` | 0.0% | 0.7% | n/a | no |
| partial | `L3_n3` | `cut3` | 0.0% | 1.0% | n/a | no |
| partial | `L3_n4` | `cut3` | 55.6% | 48.3% | n/a | yes |
| partial | `L3_c1_n1` | `cut3` | 100.0% | 0.7% | n/a | yes |
| partial | `L3_c2_n2` | `cut3` | 100.0% | 0.7% | n/a | yes |
| partial | `L3_c3_n3` | `cut3` | 100.0% | 0.7% | n/a | yes |
| partial | `L3_end_dir` | `cut3` | 50.7% | 0.7% | n/a | no |
| interchange | `all variables` | `cut6` | 52.0% | 52.0% | 21.1% | no |
| partial | `L6_n1` | `cut6` | 100.0% | 0.7% | n/a | yes |
| partial | `L6_n2` | `cut6` | 100.0% | 0.7% | n/a | yes |
| partial | `L6_n3` | `cut6` | 0.0% | 0.7% | n/a | yes |
| partial | `L6_n4` | `cut6` | 49.6% | 23.3% | n/a | no |
| partial | `L6_c1_n1` | `cut6` | 100.0% | 0.7% | n/a | yes |
| partial | `L6_c2_n2` | `cut6` | 100.0% | 0.7% | n/a | yes |
| partial | `L6_c3_n3` | `cut6` | 100.0% | 0.7% | n/a | yes |
| partial | `L6_end_n4` | `cut6` | 1.0% | 0.7% | n/a | no |
| partial | `L6_end_dir` | `cut6` | 50.7% | 0.7% | n/a | no |
| interchange | `all variables` | `cut9` | 100.0% | 100.0% | 17.1% | yes |
| partial | `L9_n1` | `cut9` | 100.0% | 0.7% | n/a | yes |
| partial | `L9_n2` | `cut9` | 100.0% | 0.7% | n/a | yes |
| partial | `L9_n3` | `cut9` | 100.0% | 0.7% | n/a | yes |
| partial | `L9_n4` | `cut9` | 50.0% | 1.0% | n/a | no |
| partial | `L9_c1_n1` | `cut9` | 100.0% | 0.7% | n/a | yes |
| partial | `L9_c2_n2` | `cut9` | 100.0% | 0.7% | n/a | yes |
| partial | `L9_c3_n3` | `cut9` | 100.0% | 0.7% | n/a | yes |
| partial | `L9_end_n4` | `cut9` | 19.2% | 7.7% | n/a | no |
| partial | `L9_end_dir` | `cut9` | 98.0% | 1.0% | n/a | yes |
| partial | `L9_end_ans` | `cut9` | 3.9% | 2.0% | n/a | no |
| interchange | `L12_end_ans` | `cut12` | 100.0% | 100.0% | 99.3% | yes |

## What fails

- partial of emb_n1 at cut0: prediction right in 0.0% of the cases where the network's output changed (need 50%; guessing 'no change' gets 0% of these)
- partial of emb_n2 at cut0: prediction right in 0.0% of the cases where the network's output changed (need 50%; guessing 'no change' gets 0% of these)
- partial of emb_n3 at cut0: prediction right in 0.0% of the cases where the network's output changed (need 50%; guessing 'no change' gets 0% of these)
- partial of emb_n4 at cut0: prediction right in 40.7% of the cases where the network's output changed (need 50%; guessing 'no change' gets 0% of these)
- partial of L3_n1 at cut3: the network's output changed in only 0.7% of cases, but the construction predicted changes: at this site the network's output does not depend on this variable
- partial of L3_end_dir at cut3: the network's output changed in only 0.0% of cases, but the construction predicted changes: at this site the network's output does not depend on this variable
- partial of L3_n2 at cut3: prediction right in 0.0% of the cases where the network's output changed (need 50%; guessing 'no change' gets 0% of these)
- partial of L3_n3 at cut3: prediction right in 0.0% of the cases where the network's output changed (need 50%; guessing 'no change' gets 0% of these)
- substitution at cut6: 53.5% (need 90%)
- interchange of all variables at cut6: prediction right in 52.0% of the cases where the network's output changed (need 90%)
- partial of L6_end_n4 at cut6: the network's output changed in only 0.0% of cases, but the construction predicted changes: at this site the network's output does not depend on this variable
- partial of L6_end_dir at cut6: the network's output changed in only 0.0% of cases, but the construction predicted changes: at this site the network's output does not depend on this variable
- partial of L6_n4 at cut6: prediction right in 49.6% of the cases where the network's output changed (need 50%; guessing 'no change' gets 0% of these)
- partial of L9_n4 at cut9: the network's output changed in only 0.7% of cases, but the construction predicted changes: at this site the network's output does not depend on this variable
- partial of L9_end_n4 at cut9: prediction right in 19.2% of the cases where the network's output changed (need 50%; guessing 'no change' gets 0% of these)
- partial of L9_end_ans at cut9: prediction right in 3.9% of the cases where the network's output changed (need 50%; guessing 'no change' gets 0% of these)

## Files

- `construction.py`: the delivered construction (runnable: `model(x, iv)` and `ALIGN`).
- `report.txt`, `report.json`: the judge's report on it.
- `submission_NN.py`, `submission_NN.json`: every construction the reader submitted, with its report.
- `run.json`: the whole run: prompt, every step (reasoning, code, outputs), usage.
- `predict.json` (if present): the held-out intervention benchmark.

## Reproduce

```bash
python3 lab/judge.py gpt2_countdown lab/constructions/gpt2_countdown/20261114-130627-1557/construction.py        # re-run every test on the delivered construction
python3 lab/predict.py gpt2_countdown lab/constructions/gpt2_countdown/20261114-130627-1557/construction.py              # held-out interventions against the baselines
python3 lab/constructor.py gpt2_countdown --told --no-weights --one-shot --steps 80   # a fresh run with the same settings (needs lab/.env)
```
