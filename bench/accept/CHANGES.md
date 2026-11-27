# Change ledger: the acceptance suite's interface and scorer code

*Public ledger (`docs/spec/BENCH_BRIEF.md` §2). Only changes to the interface and to the scorer's code are logged here;
everything sealed is logged privately. A scorer bug is fixed in both copies as a new, named version.*

| version | date | change |
|---|---|---|
| interface bench-accept/v1 (draft) | Nov 27, 2026 | First version: task folder, blueprint, allowed map classes, invented primitives (200 AST nodes), the network's own choice, run alone, the `Box` protocol, the verifier hook. Adopts vocabulary v2 (f0b3b680) unchanged |
| scorer-v1 | Nov 27, 2026 | Development copy of the scorer: format check, run alone in a separate process, interchange on decisive pairs (at least 20 per tested part) with family-wise random-subspace controls (99th percentile of the best of K, K = parts tested), substitution, kept parts suffice (5th percentile), overview, recovery of sealed variables (agreement or correlation, and substitution at the sealed location), S5a checks (run alone against the task-rule baseline; a lookup at the decision must be at chance on fresh inputs). "No value" (an all-zero categorical reading, declared by `null_below`) agrees only with "no value" |
| interface bench-accept/v1 | Nov 27, 2026 | Completed: the reference true blueprint (`reference/frac_prevs/`, INTERFACE.md §8); the categorical encoding's optional `null_below` |
