# Change ledger (rule: LEARNINGS E12)

Every change to the judge, reader or constructor needs an entry here BEFORE code. An entry that can't fill every field is an experiment: behind a flag, off by default.

| id | failure (run id + file) | cause, reproduced | change | fails-before / passes-after test | held-out + adversarial cases | verdicts that must not change | status |
|---|---|---|---|---|---|---|---|
| F2 | round 1: exact output ties decided by float noise | reproduced: identical logits, different argmax | round outputs to 1e-9 | tie cases | - | all others | kept (pre-ledger; has evidence) |
| F1 | round 1: copies judged by changing rules (E2) | - | single non-integer-literal rule | 17 round-1 constructions only | **none** | - | **FAILED in use (E10)**; to be replaced via a full entry |
| F3/F4 | E4/E5 (one anecdote each) | **not reproduced** | reader v2 | **none valid** (leaky judge, n=2) | none | - | experiment; set back to off-by-default |
| I2 (intent v2) | see INTENT.md | see DIAGNOSIS_ABOVE_32.md | intent rewritten; the code is rebuilt from INTENT.md in its rebuild order | rebuild step 1 (measurement proof) | must-fail set written before the measure | M2,2, M4,3 | intent accepted pending Shreyas; no code yet |
| J3 (open) | 7 copies passed glass-2; shuffled lc_3_4 control passed | see REPORT_ROUND2 corrections §3 | not designed yet: must meet §3 (a)–(c) | all held copies must fail; M2,2/M4,3 must pass | copies in encodings written BEFORE the rule is designed | M2,2, M4,3 certified | open |
| T1 (glass-2.1) | lc_8_6 reports say the network is wrong on 66.9% of inputs (e.g. `lc_8_6/20261118-055610-d7df/report.txt` §6); its true accuracy is 85.4% | reproduced: `nets.domain_inputs` subsamples 20,000 of the 46,656 inputs at random, while `judge.load_labels` returned labels in full-file order | `load_labels` applies the same subsample | before: judge labels give 0.33 accuracy on lc_8_6; after: 0.854 (equal to the accuracy on the full file) | lc_6_5 and M16,10 (≤ 20,000 inputs) unchanged | every target with ≤ 20,000 inputs is unaffected (no subsample) | done (translation error) |
