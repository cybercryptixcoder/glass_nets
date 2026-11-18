# Glass Networks: AlgZoo results

*Regenerated 2026-11-18 01:42 EST by `scripts/glass/results.py` from `results/glass/queue.json` and each run's `report.json`. 'main' = blinded constructor run; 'prior' = prior-only control (told the task, no weights, one submission); 'shuffled' = the same reader on weights shuffled within each tensor (should fail).*

| network | params | run | job status | run id | verdict | stages | behaviour | all sites at once | worst slice | mistakes reproduced | steps | submissions | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LC (hidden 3, length 4) | 108 | maina (v2) | done | 20261118-045127-234b | COMPLETE | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 29 | 4 | $0.15 |
| LC (hidden 3, length 4) | 108 | mainb (v2) | done | 20261118-045127-2b58 | not verified | 0/3 | 52.3% | 100.0% | 0.0% | 33.3% | 80 | 14 | $0.42 |
| LC (hidden 4, length 5) | 188 | maina (v2) | done | 20261118-045956-6821 | COMPLETE | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 79 | 12 | $0.53 |
| LC (hidden 4, length 5) | 188 | mainb (v2) | done | 20261118-050059-93e7 | COMPLETE | 3/3 | 99.9% | 99.9% | 99.6% | 99.7% | 57 | 9 | $0.23 |
| LC (hidden 6, length 5) | 378 | maina (v2) | done | 20261118-050628-dba7 | COMPLETE | 3/3 | 99.9% | 99.9% | 99.6% | 98.7% | 58 | 8 | $0.27 |
| LC (hidden 6, length 5) | 378 | mainb (v2) | done | 20261118-051434-ca67 | not verified | 3/3 | 99.7% | 99.5% | 98.8% | 94.9% | 120 | 17 | $0.85 |
| LC (hidden 8, length 6) | 656 | maina (v2) | done | 20261118-055556-e50d | COMPLETE | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 72 | 11 | $0.32 |
| LC (hidden 8, length 6) | 656 | mainb (v2) | done | 20261118-055610-d7df | not verified | 1/3 | 92.4% | 100.0% | 0.0% | 90.7% | 120 | 17 | $0.47 |
| M2,2 | 10 | v1a (v1) | done | 20261118-045127-9999 | COMPLETE | 2/2 | 100.0% | 100.0% | 99.9% | 0.0% | 14 | 2 | $0.10 |
| M2,2 | 10 | v1b (v1) | running | 20261118-063701-8739 | - | - | - | - | - | - | 9 | 2 | $0.04 |
| M2,2 | 10 | v2a (v2) | done | 20261118-045127-0fd7 | COMPLETE | 2/2 | 100.0% | 100.0% | 99.9% | 0.0% | 12 | 2 | $0.07 |
| M4,3 | 32 | v1a (v1) | done | 20261118-045626-6756 | COMPLETE | 3/3 | 99.5% | 99.5% | 99.1% | 83.8% | 23 | 5 | $0.13 |
| M4,3 | 32 | v2a (v2) | done | 20261118-045809-3312 | COMPLETE | 3/3 | 99.4% | 99.5% | 99.0% | 82.4% | 7 | 1 | $0.03 |
| M4,4 | 36 | v1a (v1) | done | 20261118-051642-3aea | COMPLETE | 4/4 | 98.3% | 100.0% | 96.6% | 97.8% | 76 | 11 | $0.39 |
| M4,4 | 36 | v2a (v2) | done | 20261118-052137-c0c7 | not verified | 4/4 | 100.0% | 100.0% | 100.0% | 100.0% | 71 | 10 | $0.29 |
| M8,5 | 112 | v1a (v1) | done | 20261118-053747-35b7 | COMPLETE | 5/5 | 99.7% | 99.8% | 99.4% | 98.9% | 43 | 6 | $0.21 |
| M8,5 | 112 | v2a (v2) | done | 20261118-054317-293b | COMPLETE | 5/5 | 99.1% | 99.0% | 98.1% | 95.5% | 64 | 9 | $0.29 |
| M8,8 | 136 | v1a (v1) | running | 20261118-060905-2311 | - | - | - | - | - | - | 66 | 9 | $0.37 |
| M8,8 | 136 | v2a (v2) | running | 20261118-061528-0f74 | - | - | - | - | - | - | 47 | 7 | $0.28 |
| M16,10 | 432 | v1a (v1) | done | 20261118-062600-878d | COMPLETE | 10/10 | 97.2% | 97.2% | 94.5% | 74.2% | 33 | 5 | $0.12 |
| M16,10 | 432 | v2a (v2) | running | 20261118-063446-2d29 | - | - | - | - | - | - | 21 | 3 | $0.09 |
