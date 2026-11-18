# Glass Networks: AlgZoo results

*Regenerated 2026-11-18 03:59 EST by `scripts/glass/results.py` from `results/glass/queue.json` and each run's `report.json`. 'main' = blinded constructor run; 'prior' = prior-only control (told the task, no weights, one submission); 'shuffled' = the same reader on weights shuffled within each tensor (should fail).*

| network | params | run | job status | run id | verdict | stages | behaviour | all sites at once | worst slice | mistakes reproduced | steps | submissions | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LC (hidden 3, length 4) | 108 | maina (v2) | done | 20261118-045127-234b | COMPLETE | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 29 | 4 | $0.15 |
| LC (hidden 3, length 4) | 108 | mainb (v2) | done | 20261118-045127-2b58 | not verified | 0/3 | 52.3% | 100.0% | 0.0% | 33.3% | 80 | 14 | $0.42 |
| LC (hidden 3, length 4) | 108 | prior (v2) | done | 20261118-074816-d122 | not verified | 0/3 | 85.9% | 86.7% | 90.9% | 0.0% | 5 | 1 | $0.06 |
| LC (hidden 3, length 4) | 108 | shuffled (v2) | done | 20261118-075036-9482 | COMPLETE | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 27 | 4 | $0.12 |
| LC (hidden 4, length 5) | 188 | maina (v2) | done | 20261118-045956-6821 | COMPLETE | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 79 | 12 | $0.53 |
| LC (hidden 4, length 5) | 188 | mainb (v2) | done | 20261118-050059-93e7 | COMPLETE | 3/3 | 99.9% | 99.9% | 99.6% | 99.7% | 57 | 9 | $0.23 |
| LC (hidden 4, length 5) | 188 | prior (v2) | done | 20261118-075554-da05 | not verified | 0/3 | 80.8% | 84.4% | 56.6% | 0.0% | 7 | 1 | $0.07 |
| LC (hidden 4, length 5) | 188 | shuffled (v2) | done | 20261118-080002-f417 | not verified | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 30 | 5 | $0.09 |
| LC (hidden 6, length 5) | 378 | maina (v2) | done | 20261118-050628-dba7 | COMPLETE | 3/3 | 99.9% | 99.9% | 99.6% | 98.7% | 58 | 8 | $0.27 |
| LC (hidden 6, length 5) | 378 | mainb (v2) | done | 20261118-051434-ca67 | not verified | 3/3 | 99.7% | 99.5% | 98.8% | 94.9% | 120 | 17 | $0.85 |
| LC (hidden 6, length 5) | 378 | prior (v2) | done | 20261118-080226-a3ea | not verified | 0/3 | 86.9% | 35.2% | 69.1% | 6.4% | 10 | 1 | $0.07 |
| LC (hidden 6, length 5) | 378 | shuffled (v2) | done | 20261118-080637-3e88 | not verified | 0/3 | 25.2% | 58.5% | 0.0% | 17.6% | 30 | 7 | $0.11 |
| LC (hidden 8, length 6) | 656 | maina (v2) | done | 20261118-055556-e50d | COMPLETE | 3/3 | 100.0% | 100.0% | 100.0% | 100.0% | 72 | 11 | $0.32 |
| LC (hidden 8, length 6) | 656 | mainb (v2) | done | 20261118-055610-d7df | not verified | 1/3 | 92.4% | 100.0% | 0.0% | 90.7% | 120 | 17 | $0.47 |
| LC (hidden 8, length 6) | 656 | prior (v2) | done | 20261118-081020-5842 | not verified | 0/3 | 85.7% | 32.8% | 61.3% | 83.8% | 8 | 1 | $0.08 |
| LC (hidden 8, length 6) | 656 | shuffled (v2) | done | 20261118-081657-2679 | not verified | 0/3 | 94.3% | 94.3% | 0.0% | 95.7% | 30 | 5 | $0.10 |
| M2,2 | 10 | v1a (v1) | done | 20261118-045127-9999 | COMPLETE | 2/2 | 100.0% | 100.0% | 99.9% | 0.0% | 14 | 2 | $0.10 |
| M2,2 | 10 | v1b (v1) | done | 20261118-063701-8739 | COMPLETE | 2/2 | 100.0% | 100.0% | 99.9% | 0.0% | 12 | 4 | $0.08 |
| M2,2 | 10 | v2a (v2) | done | 20261118-045127-0fd7 | COMPLETE | 2/2 | 100.0% | 100.0% | 99.9% | 0.0% | 12 | 2 | $0.07 |
| M2,2 | 10 | v2b (v2) | done | 20261118-064322-3224 | COMPLETE | 2/2 | 100.0% | 100.0% | 99.9% | 0.0% | 14 | 2 | $0.08 |
| M4,3 | 32 | v1a (v1) | done | 20261118-045626-6756 | COMPLETE | 3/3 | 99.5% | 99.5% | 99.1% | 83.8% | 23 | 5 | $0.13 |
| M4,3 | 32 | v1b (v1) | done | 20261118-064501-97a8 | COMPLETE | 3/3 | 98.8% | 98.9% | 98.4% | 62.2% | 14 | 2 | $0.07 |
| M4,3 | 32 | v2a (v2) | done | 20261118-045809-3312 | COMPLETE | 3/3 | 99.4% | 99.5% | 99.0% | 82.4% | 7 | 1 | $0.03 |
| M4,3 | 32 | v2b (v2) | done | 20261118-064821-2f42 | COMPLETE | 3/3 | 99.7% | 99.7% | 99.4% | 90.5% | 14 | 2 | $0.06 |
| M4,4 | 36 | v1a (v1) | done | 20261118-051642-3aea | COMPLETE | 4/4 | 98.3% | 100.0% | 96.6% | 97.8% | 76 | 11 | $0.39 |
| M4,4 | 36 | v1b (v1) | done | 20261118-064925-6c8e | not verified | 4/4 | 100.0% | 100.0% | 100.0% | 100.0% | 44 | 6 | $0.25 |
| M4,4 | 36 | v2a (v2) | done | 20261118-052137-c0c7 | not verified | 4/4 | 100.0% | 100.0% | 100.0% | 100.0% | 71 | 10 | $0.29 |
| M4,4 | 36 | v2b (v2) | done | 20261118-065051-cf8c | COMPLETE | 4/4 | 99.1% | 99.0% | 98.8% | 98.8% | 61 | 9 | $0.34 |
| M8,5 | 112 | v1a (v1) | done | 20261118-053747-35b7 | COMPLETE | 5/5 | 99.7% | 99.8% | 99.4% | 98.9% | 43 | 6 | $0.21 |
| M8,5 | 112 | v1b (v1) | done | 20261118-065208-9151 | COMPLETE | 5/5 | 98.5% | 98.7% | 97.8% | 93.9% | 72 | 12 | $0.34 |
| M8,5 | 112 | v2a (v2) | done | 20261118-054317-293b | COMPLETE | 5/5 | 99.1% | 99.0% | 98.1% | 95.5% | 64 | 9 | $0.29 |
| M8,5 | 112 | v2b (v2) | done | 20261118-070518-7ae0 | not verified | 5/5 | 100.0% | 100.0% | 100.0% | 100.0% | 80 | 11 | $0.50 |
| M8,8 | 136 | v1a (v1) | done | 20261118-060905-2311 | not verified | 0/8 | 32.5% | 100.0% | 20.7% | 60.0% | 80 | 12 | $0.42 |
| M8,8 | 136 | v1b (v1) | done | 20261118-071402-b3b5 | COMPLETE | 8/8 | 95.6% | 97.7% | 89.3% | 95.0% | 21 | 3 | $0.06 |
| M8,8 | 136 | v2a (v2) | done | 20261118-061528-0f74 | not verified | 8/8 | 100.0% | 100.0% | 100.0% | 100.0% | 54 | 8 | $0.31 |
| M8,8 | 136 | v2b (v2) | done | 20261118-071801-82cd | not verified | 8/8 | 100.0% | 100.0% | 100.0% | 100.0% | 63 | 10 | $0.35 |
| M16,10 | 432 | v1a (v1) | done | 20261118-062600-878d | COMPLETE | 10/10 | 97.2% | 97.2% | 94.5% | 74.2% | 33 | 5 | $0.12 |
| M16,10 | 432 | v1b (v1) | done | 20261118-071942-7a33 | COMPLETE | 10/10 | 96.7% | 97.5% | 93.6% | 67.6% | 136 | 20 | $0.80 |
| M16,10 | 432 | v2a (v2) | done | 20261118-063446-2d29 | COMPLETE | 10/10 | 98.9% | 99.2% | 97.3% | 84.1% | 71 | 11 | $0.48 |
| M16,10 | 432 | v2b (v2) | done | 20261118-072351-bddd | not verified | 10/10 | 94.3% | 95.5% | 87.9% | 64.3% | 160 | 25 | $0.98 |
