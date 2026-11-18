# Glass Networks: AlgZoo results

*Regenerated 2026-11-18 00:03 EST by `scripts/glass/results.py` from `results/glass/queue.json` and each run's `report.json`. 'main' = blinded constructor run; 'prior' = prior-only control (told the task, no weights, one submission); 'shuffled' = the same reader on weights shuffled within each tensor (should fail).*

| network | params | run | job status | run id | verdict | stages | behaviour | all sites at once | worst slice | mistakes reproduced | steps | submissions | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LC (hidden 3, length 4) | 108 | maina (v2) | done | 20261118-045127-234b | COMPLETE | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 29 | 4 | $0.15 |
| LC (hidden 3, length 4) | 108 | mainb (v2) | running | 20261118-045127-2b58 | - | - | - | - | - | - | 44 | 8 | $0.21 |
| LC (hidden 4, length 5) | 188 | maina (v2) | running | 20261118-045956-6821 | - | - | - | - | - | - | 20 | 2 | $0.06 |
| LC (hidden 4, length 5) | 188 | mainb (v2) | running | 20261118-050059-93e7 | - | - | - | - | - | - | 22 | 4 | $0.04 |
| M2,2 | 10 | v1a (v1) | done | 20261118-045127-9999 | COMPLETE | 2/2 | 100.0% | 100.0% | 99.9% | 0.0% | 14 | 2 | $0.10 |
| M2,2 | 10 | v2a (v2) | done | 20261118-045127-0fd7 | COMPLETE | 2/2 | 100.0% | 100.0% | 99.9% | 0.0% | 12 | 2 | $0.07 |
| M4,3 | 32 | v1a (v1) | running | 20261118-045626-6756 | - | - | - | - | - | - | 17 | 3 | $0.09 |
| M4,3 | 32 | v2a (v2) | done | 20261118-045809-3312 | COMPLETE | 3/3 | 99.4% | 99.5% | 99.0% | 82.4% | 7 | 1 | $0.03 |
