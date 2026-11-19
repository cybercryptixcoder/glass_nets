# The mechanism atlas: what it is, what it does, and what it has shown

*Draft started Nov 19, 05:52 ET (clock). Held-out verdicts are filled in as they land; until then the sections marked PENDING are open. Detailed record: `docs/glass/DIARY.md`; change ledger: `docs/glass/CHANGES.md`; plan and status: `docs/glass/ATLAS.md`.*

## What it is, in one paragraph

Give it a trained network. It splits every weight matrix exactly into small rank-1 pieces ("mechanisms"), using stochastic parameter decomposition (SPD; Bushnaq, Braun & Sharkey 2025). For each mechanism it learns a readable rule for **when** the network needs it (a "trigger", e.g. "ON when the max arrived in the last step and x_t is ≥ 0.64 below it"). It measures **what** the mechanism does (removing it where needed flips the answer from the 2nd largest to the max, say). Then it checks the whole account: it rebuilds the network from its mechanisms, switches each one on and off by its rule, and counts how many of the real network's decisions survive. The output is a certificate (markdown and JSON), a catalog page, and a playground where you step through one input and watch which mechanisms switch on.

## How to use it

```
python3 scripts/glass/atlas_spd_job.py OUT 30000 512 cpu --coef-min=1e-4 TARGET      # decompose (minutes to an hour)
python3 lab/atlas.py TARGET OUT/spd_TARGET_original_30000_512.pt 4 --absorb [--control-spd SHUFFLED.pt]
python3 lab/server.py   # then open /catalog?t=TARGET
```

## What a certificate claims, and how each claim is checked

| claim | measured as | bar |
|---|---|---|
| (i') the mechanisms account for the network | decisions kept with all mechanisms on and nothing else | ≥ the task rule (how often "the task's own answer" matches the network) |
| (ii) the triggers say when each mechanism is needed | decisions kept when each trusted trigger switches its mechanism | ≥ the task rule |
| (iii) most mechanisms have a trigger you can trust | share of switched mechanisms whose trigger is ≥ 90% right on held-out clear-cut cases | ≥ 50% |
| (iv) mechanisms matter where they are said to | output change from removing a mechanism where needed vs where not (median) | ≥ 10× |
| (v) the structure is learned, not an artefact | mechanisms in functional groups, trained vs shuffled-weights network | trained > shuffled |
| (vi) the triggers do real work | share of not-needed calls the triggers switch off | ≥ 50% |

All bars were fixed before the networks they judge were run. The task-rule bar means an account must predict the network at least as well as the one-line description of its task.

## Evidence so far

**Known answers (networks built by hand, so the true mechanisms are known):**
- A recurrent network with 5 planted features: all 15 planted mechanisms recovered, triggers ≥ 0.98, rebuilt network 100%.
- A transformer that copies the previous token: all 16 planted "copy token v" mechanisms recovered across embedding, value, output and unembedding, triggers ≥ 0.997, rebuilt network 100%.

**Development networks** (they shaped the method, so they don't count as tests):
- Weak networks (task rules 46–71%) pass every criterion with the default settings.
- Accurate networks fail with the default settings. The cause: SPD's "not needed" labels are wrong for small activity, and the mechanisms leave a decision-relevant residual.
- **The fix (A26):** weaker sparsity pressure, then folding the residual into the mechanisms. It makes both accurate development networks pass every criterion:
  - argmedian 16_5 (98.4% accurate): rebuilt 98.9%, triggers switch off 93% of not-needed calls, 25 task-level explanations;
  - M16,10 (95.3%): rebuilt 96.8%.

**Held-out test A18** (six well-trained networks, 93.9–99.6% accurate, up to 1,376 parameters, default settings): **fails, 0 of 6.** Triggers (64–91% trusted) and specificity (11–103×) pass everywhere. Precision fails everywhere: the mechanisms alone keep 45–95% of decisions against bars of 94–99.6%. So either the switched network loses decisions, or almost nothing can be switched safely.

**Held-out test A27** (six new well-trained networks, up to 1,376 parameters, with the A26 fix): PENDING. This is the test of whether the method works on accurate networks.

## What the atlases find (examples)

- **ARC's M16,10** (the 2nd-argmax challenge network): 35 mechanisms with the default settings; 4 in functional groups, against 0 in the shuffled control. W_h.6 switches on when the max arrived in the last step and the current value is well below it; removing it makes the network return the max instead of the 2nd largest.
- **argmedian 16_5**: the mechanisms hold the median against both neighbours. With the default settings, 12 stop the answer moving to the next-larger value and 9 to the next-smaller; the shuffled control has 1. With the precise settings: 25 vs 12.
- **argmedian 8_7** (49.5% accurate): most recurrent mechanisms switch on by time step ("after step 3"). The network stages its computation over time, not by data content.
- **Recurrence across training runs: not confirmed.** ARC's M8,5 and zoo seed 0 shared 5 mechanisms with the same role and trigger variables, against 2 by chance (p = 0.003). That was exploratory. The pre-registered confirmation needed 3 of 4 more seeds to beat the chance control's 95th percentile. Seeds 1 and 2 have 3 matches each (null mean about 1.8, 95th percentile 3, P = 0.20 and 0.24), so it fails. There is a weak trend: every seed sits above the chance mean.

## Limits, stated plainly

- Transformers: the known answer passes, but real AlgZoo transformers decompose poorly with the default settings (rebuilt 53–77% vs task rules 81–95%). The A26 settings have not been tried on them yet.
- At the precise settings almost every component the decomposition offers is used (57 of 60 in argmedian 16_5), so mechanism counts partly reflect that budget.
- With absorption, criterion (i') becomes close to automatic whenever the alive mechanisms' input directions span the matrix's inputs: least squares can then make their sum exact on the data. Under A26 the informative criteria are (ii), the triggers keeping the decisions, and (vi), the triggers switching off most not-needed activity. Absorption does not make those easier: at minimality 1e-3 it made (ii) worse (0.876 → 0.723 on argmedian 16_5).
- Triggers for output-layer mechanisms are often not trusted: their importance is graded, not on/off.
- Findings are stated in the task's terms (ranks, steps, values), but many triggers still name other neurons. A unit-map gloss is shown where one exists.
