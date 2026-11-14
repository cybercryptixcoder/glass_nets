# Construction run `20261114-114450-62b2` on `gpt2_be_verb`

*Written by `lab/results.py` from the files in this folder. Network: gpt2_be_verb (124,439,808 weights). Known answer (never shown to a blinded reader): GPT-2 small prefers the form of 'to be' that agrees with the noun's number and the cue's tense: is / are / was / were..*

## The run

- Reader: deepseek-ai/DeepSeek-V4.1-Flash, reasoning effort high; step budget 120; used 44 steps, 8 submissions, 7459 s, $0.193.
- Weights: original. Reader was blinded (not told what the network does).
- Title: blinded run A. Status: done.

## Result

**VERIFIED AND COMPLETE**: every stage of the network has a verified site.

Run in place of the network, the construction makes the same decision on 100.0% of 360 held-out inputs (a constant output: 20.0%). Construction length: 2,663 characters.

## The mechanism, in the reader's words

Prompt: '<|endoftext|><cue>, the <noun>' (5 tokens; cue at position 1, noun at position 4 = last). Answer in [' is',' are',' was',' were'] = 2*TENSE + NUMBER, TENSE=1 iff cue index>=6 (Yesterday..Originally), NUMBER=1 iff noun is plural (noun index>=30). The two facts live at two positions: WHICH CUE WORD is at position 1 (only its tense half is used) and WHETHER the noun at position 4 is plural. Each named residual depends only on the tokens that produced it, so the cue identity at a cue-carrying position and the number at the last position explain it. Stage by stage: at the embedding the cue identity is at position 1 and the number at position 4. Layers 0-2 copy the cue into position 2 (the comma), so at depth 3 and 6 positions 1 and 2 both carry it; the number stays at position 4. Layers 6-8 keep the cue at positions 1-2 and start to write the answer class into the last position: at depth 9 position 4 already carries the 4-way class code, but only its NUMBER half is read by the later layers - the tense is still taken from the cue at positions 1-2 (swapping just the code at position 4 changes only the verb number). Layers 9-11 finish the job: at depth 12 the last position holds the answer class and the four answer logits are read straight off it. model() carries cue and number from stage to stage, takes the number from the class code at depth 9, and rebuilds the class for the last stage.

## The reader's final summary

**VERIFIED AND COMPLETE — all five stages.**

**Mechanism.** Prompt `<|endoftext|> <cue> , <the> <noun>`; the answer is a 2×2 fact table: `[' is',' are',' was',' were'] = 2·TENSE + NUMBER`, TENSE from the cue (past cues Yesterday…Originally → was/were) and NUMBER from whether the noun is plural. The two facts are computed and carried **independently, at separate positions**, and never mixed until the very end.

**Stage by stage (verified cuts):**
- **Embedding (resid0):** token+position embeddings only — the cue identity lives at position 1, the plural bit at the last position (4). Positions 0,2,3 are prompt constants.
- **Layers 0–2 (resid3):** the cue's identity is copied by attention into position 2 (the comma); the number bit stays at position 4.
- **Layers 3–5 (resid6):** unchanged layout — cue at positions 1 and 2, number at position 4.
- **Layers 6–8 (resid9):** still cue at 1–2, and now the last position has begun to accumulate a 4-way class code. Crucially, only its **number half** is consumed downstream — swapping just the code at position 4 flips the verb's number but not its tense, so the tense is still read from the cue at positions 1–2.
- **Layers 9–11 (resid12):** the last position assembles the full answer class from (tense at 1–2) + (number), and the four answer logits are read linearly straight off it.

So the network is a **copy-and-carry circuit**: the cue is duplicated forward through the comma position, the noun's number is held at the end position, and the two bits are only combined at the last layer — the model's answer is a lookup of the resulting 2-bit class.

## Stages

| Stage | Status | Sites |
|---|---|---|
| embedding | verified | cut0 |
| layers 0-2 | verified | cut3 |
| layers 3-5 | verified | cut6 |
| layers 6-8 | verified | cut9 |
| layers 9-11 | verified | cut12 |

## Where each variable lives

| Variable | Dimensions | Sites |
|---|---|---|
| `t0` | 12 | resid0@1 |
| `n0` | 2 | resid0@4 |
| `t3` | 12 | resid3@1, resid3@2 |
| `n3` | 2 | resid3@4 |
| `t6` | 12 | resid6@1, resid6@2 |
| `n6` | 2 | resid6@4 |
| `t9` | 12 | resid9@1, resid9@2 |
| `c9` | 4 | resid9@4 |
| `cls12` | 4 | resid12@4 |

## Site tests

Substitute: decisions kept when the site is replaced by the construction's prediction. Remove: decisions kept when the named subspace is removed. Mean-ablate: decisions kept when the whole site is replaced by its mean.

| Site | Named dims / PCs that suffice | Variance explained | Substitute | Remove | Mean-ablate | Pass |
|---|---|---|---|---|---|---|
| `cut0` | 12 / 36 | 0.37 | 100.0% | 50.0% | 20.0% | yes |
| `cut3` | 12 / 34 | 0.42 | 100.0% | 48.9% | 20.0% | yes |
| `cut6` | 12 / 29 | 0.53 | 100.0% | 50.3% | 20.0% | yes |
| `cut9` | 14 / 26 | 0.57 | 100.0% | 52.5% | 20.0% | yes |
| `cut12` | 3 / 12 | 0.09 | 100.0% | 42.8% | 20.0% | yes |

## Intervention tests

Share of cases where the predictor named the network's new decision, among cases where the decision changed ("no change" scores 0 on these by definition).

| Test | Variable | Site | Construction | Source output | Attribution patching | Pass |
|---|---|---|---|---|---|---|
| interchange | `all variables` | `cut0` | 95.4% | 92.0% | 0.0% | yes |
| partial | `t0` | `cut0` | 100.0% | 49.0% | n/a | yes |
| partial | `n0` | `cut0` | 100.0% | 44.7% | n/a | yes |
| interchange | `all variables` | `cut3` | 100.0% | 100.0% | 3.0% | yes |
| partial | `t3` | `cut3` | 100.0% | 49.0% | n/a | yes |
| partial | `n3` | `cut3` | 100.0% | 49.7% | n/a | yes |
| interchange | `all variables` | `cut6` | 98.2% | 96.7% | 4.0% | yes |
| partial | `t6` | `cut6` | 100.0% | 47.3% | n/a | yes |
| partial | `n6` | `cut6` | 100.0% | 49.3% | n/a | yes |
| interchange | `all variables` | `cut9` | 99.1% | 96.7% | 32.4% | yes |
| partial | `t9` | `cut9` | 100.0% | 36.0% | n/a | yes |
| partial | `c9` | `cut9` | 80.5% | 60.0% | n/a | yes |
| interchange | `cls12` | `cut12` | 99.6% | 99.7% | 99.6% | yes |

## Files

- `construction.py`: the delivered construction (runnable: `model(x, iv)` and `ALIGN`).
- `report.txt`, `report.json`: the judge's report on it.
- `submission_NN.py`, `submission_NN.json`: every construction the reader submitted, with its report.
- `run.json`: the whole run: prompt, every step (reasoning, code, outputs), usage.
- `predict.json` (if present): the held-out intervention benchmark.

## Reproduce

```bash
python3 lab/judge.py gpt2_be_verb lab/constructions/gpt2_be_verb/20261114-114450-62b2/construction.py        # re-run every test on the delivered construction
python3 lab/predict.py gpt2_be_verb lab/constructions/gpt2_be_verb/20261114-114450-62b2/construction.py              # held-out interventions against the baselines
python3 lab/constructor.py gpt2_be_verb --steps 120   # a fresh run with the same settings (needs lab/.env)
```
