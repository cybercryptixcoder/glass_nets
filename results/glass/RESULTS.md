# Glass Networks: AlgZoo results

*Regenerated 2026-11-17 21:41 EST by `scripts/glass/results.py` from `results/glass/queue.json` and each run's `report.json`. 'main' = blinded constructor run; 'prior' = prior-only control (told the task, no weights, one submission); 'shuffled' = the same reader on weights shuffled within each tensor (should fail).*

| network | params | run | job status | run id | verdict | stages | behaviour | all sites at once | worst slice | mistakes reproduced | steps | submissions | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M2,2 | 10 | main | done | 20261118-015037-4e7d | not verified | 2/2 | 100.0% | 100.0% | 100.0% | 100.0% | 12 | 2 | $0.04 |
| M4,3 | 32 | main | done | 20261118-015037-c928 | COMPLETE | 3/3 | 99.2% | 99.1% | 98.1% | 79.7% | 26 | 6 | $0.13 |
| M4,4 | 36 | main | done | 20261118-015037-a0e7 | not verified | 4/4 | 100.0% | 100.0% | 100.0% | 100.0% | 28 | 5 | $0.11 |
| M8,5 | 112 | main | done | 20261118-015314-6ab4 | not verified | 0/5 | 88.9% | 100.0% | 84.3% | 0.0% | 80 | 12 | $0.34 |
| M8,8 | 136 | main | done | 20261118-015831-acaf | not verified | 8/8 | 100.0% | 100.0% | 100.0% | 100.0% | 36 | 6 | $0.19 |
| M16,10 | 432 | main | done | 20261118-015037-99a9 | not verified | 8/10 | 100.0% | 100.0% | 100.0% | 100.0% | 89 | 13 | $0.41 |
| M2,2 | 10 | main2 | done | 20261118-023529-c7fd | COMPLETE | 2/2 | 100.0% | 100.0% | 100.0% | 100.0% | 7 | 1 | $0.03 |
| M4,4 | 36 | main2 | running | 20261118-023529-1a6a | - | - | - | - | - | - | 24 | 4 | $0.09 |
| M8,5 | 112 | main2 | running | 20261118-023529-46c8 | - | - | - | - | - | - | 23 | 3 | $0.08 |
| M8,8 | 136 | main2 | running | 20261118-023529-a0d6 | - | - | - | - | - | - | 23 | 3 | $0.07 |
| M16,10 | 432 | main2 | running | 20261118-015037-99a9 | not verified | 8/10 | 100.0% | 100.0% | 100.0% | 100.0% | 89 | 13 | $0.41 |
| M2,2 | 10 | prior | done | 20261118-015915-a7c8 | COMPLETE | 2/2 | 100.0% | 99.6% | 99.9% | 0.0% | 1 | 1 | $0.04 |
| M4,3 | 32 | prior | done | 20261118-020353-1e31 | not verified | 1/3 | 98.2% | 98.7% | 95.8% | 0.0% | 7 | 1 | $0.07 |
| M4,4 | 36 | prior | done | 20261118-020602-4ec3 | not verified | 0/4 | 59.8% | 27.5% | 53.7% | 0.0% | 1 | 1 | $0.02 |
| M8,5 | 112 | prior | done | 20261118-020838-b52a | - | - | - | - | - | - | 1 | 1 | $0.03 |
| M8,8 | 136 | prior | done | 20261118-021012-9ae6 | not verified | 0/8 | 46.0% | 100.0% | 39.9% | 0.0% | 1 | 1 | $0.04 |
| M16,10 | 432 | prior | done | 20261118-021154-f774 | not verified | 1/10 | 95.5% | 93.5% | 92.8% | 0.0% | 1 | 1 | $0.04 |
| M2,2 | 10 | shuffled | done | 20261118-021155-ff8b | COMPLETE | 2/2 | 100.0% | 100.0% | 100.0% | 100.0% | 7 | 2 | $0.02 |
| M4,3 | 32 | shuffled | done | 20261118-021345-e14a | not verified | 0/3 | 67.4% | 100.0% | 22.0% | 64.9% | 30 | 9 | $0.14 |
| M4,4 | 36 | shuffled | done | 20261118-021358-b385 | not verified | 4/4 | 100.0% | 100.0% | 100.0% | 100.0% | 27 | 6 | $0.09 |
| M8,5 | 112 | shuffled | done | 20261118-021502-76fc | not verified | 0/5 | 88.3% | 100.0% | 0.0% | 88.7% | 30 | 5 | $0.08 |
| M8,8 | 136 | shuffled | done | 20261118-021616-3ac7 | verified (partial) | 1/8 | 97.6% | 100.0% | 96.9% | 97.5% | 30 | 5 | $0.09 |
| M16,10 | 432 | shuffled | done | 20261118-022212-1ef6 | not verified | 0/10 | 100.0% | 100.0% | 100.0% | 100.0% | 30 | 5 | $0.11 |
