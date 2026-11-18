# Glass Networks: AlgZoo results

*Regenerated 2026-11-18 01:03 EST by `scripts/glass/results.py` from `results/glass/queue.json` and each run's `report.json`. 'main' = blinded constructor run; 'prior' = prior-only control (told the task, no weights, one submission); 'shuffled' = the same reader on weights shuffled within each tensor (should fail).*

| network | params | run | job status | run id | verdict | stages | behaviour | all sites at once | worst slice | mistakes reproduced | steps | submissions | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LC (hidden 3, length 4) | 108 | maina (v2) | done | 20261118-045127-234b | COMPLETE | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 29 | 4 | $0.15 |
| LC (hidden 3, length 4) | 108 | mainb (v2) | done | 20261118-045127-2b58 | not verified | 0/3 | 52.3% | 100.0% | 0.0% | 33.3% | 80 | 14 | $0.42 |
| LC (hidden 4, length 5) | 188 | maina (v2) | done | 20261118-045956-6821 | COMPLETE | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 79 | 12 | $0.53 |
| LC (hidden 4, length 5) | 188 | mainb (v2) | done | 20261118-050059-93e7 | COMPLETE | 3/3 | 99.9% | 99.9% | 99.6% | 99.7% | 57 | 9 | $0.23 |
| LC (hidden 6, length 5) | 378 | maina (v2) | done | 20261118-050628-dba7 | COMPLETE | 3/3 | 99.9% | 99.9% | 99.6% | 98.7% | 58 | 8 | $0.27 |
| LC (hidden 6, length 5) | 378 | mainb (v2) | running | 20261118-051434-ca67 | - | - | - | - | - | - | 90 | 12 | $0.68 |
| LC (hidden 8, length 6) | 656 | maina (v2) | running | 20261118-055556-e50d | - | - | - | - | - | - | 26 | 3 | $0.05 |
| LC (hidden 8, length 6) | 656 | mainb (v2) | running | 20261118-055610-d7df | - | - | - | - | - | - | 25 | 3 | $0.06 |
| M2,2 | 10 | v1a (v1) | done | 20261118-045127-9999 | COMPLETE | 2/2 | 100.0% | 100.0% | 99.9% | 0.0% | 14 | 2 | $0.10 |
| M2,2 | 10 | v2a (v2) | done | 20261118-045127-0fd7 | COMPLETE | 2/2 | 100.0% | 100.0% | 99.9% | 0.0% | 12 | 2 | $0.07 |
| M4,3 | 32 | v1a (v1) | done | 20261118-045626-6756 | COMPLETE | 3/3 | 99.5% | 99.5% | 99.1% | 83.8% | 23 | 5 | $0.13 |
| M4,3 | 32 | v2a (v2) | done | 20261118-045809-3312 | COMPLETE | 3/3 | 99.4% | 99.5% | 99.0% | 82.4% | 7 | 1 | $0.03 |
| M4,4 | 36 | v1a (v1) | done | 20261118-051642-3aea | COMPLETE | 4/4 | 98.3% | 100.0% | 96.6% | 97.8% | 76 | 11 | $0.39 |
| M4,4 | 36 | v2a (v2) | done | 20261118-052137-c0c7 | not verified | 4/4 | 100.0% | 100.0% | 100.0% | 100.0% | 71 | 10 | $0.29 |
| M8,5 | 112 | v1a (v1) | done | 20261118-053747-35b7 | COMPLETE | 5/5 | 99.7% | 99.8% | 99.4% | 98.9% | 43 | 6 | $0.21 |
| M8,5 | 112 | v2a (v2) | running | 20261118-054317-293b | - | - | - | - | - | - | 48 | 6 | $0.21 |
