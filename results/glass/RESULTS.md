# Glass Networks: AlgZoo results

*Regenerated 2026-11-17 21:11 EST by `scripts/glass/results.py` from `results/glass/queue.json` and each run's `report.json`. 'main' = blinded constructor run; 'prior' = prior-only control (told the task, no weights, one submission); 'shuffled' = the same reader on weights shuffled within each tensor (should fail).*

| network | params | run | job status | run id | verdict | stages | behaviour | all sites at once | worst slice | mistakes reproduced | steps | submissions | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2,2 | 10 | main | done | 20261118-015037-4e7d | COMPLETE | 2/2 | 100.0% | 100.0% | 100.0% | 100.0% | 12 | 2 | $0.04 |
| M4,3 | 32 | main | done | 20261118-015037-c928 | COMPLETE | 3/3 | 99.2% | 99.1% | 98.1% | 79.7% | 26 | 6 | $0.13 |
| M4,4 | 36 | main | done | 20261118-015037-a0e7 | not verified | 4/4 | 100.0% | 100.0% | 100.0% | 100.0% | 28 | 5 | $0.11 |
| M8,5 | 112 | main | running | 20261118-015314-6ab4 | - | - | - | - | - | - | 66 | 10 | $0.27 |
| M8,8 | 136 | main | running | 20261118-015831-acaf | - | - | - | - | - | - | 29 | 5 | $0.15 |
| M16,10 | 432 | main | done | 20261118-015037-99a9 | not verified | 8/10 | 100.0% | 100.0% | 100.0% | 100.0% | 80 | 11 | $0.38 |
| M2,2 | 10 | prior | done | 20261118-015915-a7c8 | COMPLETE | 2/2 | 100.0% | 99.6% | 99.9% | 0.0% | 1 | 1 | $0.04 |
| M4,3 | 32 | prior | done | 20261118-020353-1e31 | not verified | 1/3 | 98.2% | 98.7% | 95.8% | 0.0% | 7 | 1 | $0.07 |
| M4,4 | 36 | prior | done | 20261118-020602-4ec3 | not verified | 0/4 | 59.8% | 27.5% | 53.7% | 0.0% | 1 | 1 | $0.02 |
| M8,5 | 112 | prior | running | 20261118-020838-b52a | - | - | - | - | - | - | 0 | 0 | $0.00 |
| M8,8 | 136 | prior | running | 20261118-021012-9ae6 | - | - | - | - | - | - | 0 | 0 | $0.00 |
