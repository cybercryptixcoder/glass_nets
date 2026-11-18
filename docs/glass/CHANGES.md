# Change ledger (rule: LEARNINGS E12)

Every change to the judge, reader or constructor needs an entry here BEFORE code. An entry that can't fill every field is an experiment: behind a flag, off by default.

| id | failure (run id + file) | cause, reproduced | change | fails-before / passes-after test | held-out + adversarial cases | verdicts that must not change | status |
|---|---|---|---|---|---|---|---|
| F2 | round 1: exact output ties decided by float noise | reproduced: identical logits, different argmax | round outputs to 1e-9 | tie cases | - | all others | kept (pre-ledger; has evidence) |
| F1 | round 1: copies judged by changing rules (E2) | - | single non-integer-literal rule | 17 round-1 constructions only | **none** | - | **FAILED in use (E10)**; to be replaced via a full entry |
| F3/F4 | E4/E5 (one anecdote each) | **not reproduced** | reader v2 | **none valid** (leaky judge, n=2) | none | - | experiment; set back to off-by-default |
| I2 (intent v2) | see INTENT.md | see DIAGNOSIS_ABOVE_32.md | intent rewritten; the code is rebuilt from INTENT.md in its rebuild order | rebuild step 1 (measurement proof) | must-fail set written before the measure | M2,2, M4,3 | intent accepted pending Shreyas; no code yet |
| J3 (open) | 7 copies passed glass-2; shuffled lc_3_4 control passed | see REPORT_ROUND2 corrections §3 | not designed yet: must meet §3 (a)–(c) | all held copies must fail; M2,2/M4,3 must pass | copies in encodings written BEFORE the rule is designed | M2,2, M4,3 certified | open |
