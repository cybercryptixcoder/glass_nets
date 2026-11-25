# Planets: status

Branch `planets`. Ledger row PL1 in docs/glass/CHANGES.md.

## Done

- Data: SPOCK 2.0 resonant and random datasets, initial conditions and N-body truth
  (scripts/planets/features.py, physfeat.py; data lives outside the repo in ~/pdata).
- Network baseline on 1,750 random systems (nn_batch.py), checked against SPOCK's default
  sampling (nn_settings_check.py).
- Initial-condition laws (fit_law.py, fit_law2.py, fit_tobit.py): best 1.23; information limit shown.
- Final law with short-integration summaries (final_law.py), frozen in
  results/planets/FROZEN_LAW.json: matches the network on 1,736 held-out systems.
- Controls (controls.py), information table (ic_only_same.py), recalibration check (recal_check.py).

## Fresh N-body verification: done

- Fresh set (fresh.py): 800 new systems, 693 integrated to 1e6 inner orbits by the stop
  (items 0-399 here, 293 of 400-799 on Colab, session 'planets', stopped at the end);
  short summaries and network predictions for them on Colab (scripts/planets/colab/, the
  Colab launchers as run). Scored by fresh_eval.py.

Scored at 21:45 (RESULTS.md, fresh section). The 107 unintegrated items can be added by
rerunning fresh.py on them and rescoring; the law must not be refit.

## To resume

1. Collect Colab outputs: featfresh_*.csv, nnfresh_*.csv, pdata/fresh_nbody_*.csv into
   ~/pdata/colab/; local outputs are ~/pdata/fresh_nbody_w*_0.csv.
2. `python3 scripts/planets/fresh_eval.py` (uses the frozen law; never refit it).
