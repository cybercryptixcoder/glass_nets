# Gate 1 (the finish line): progress report

*Draft, session 1 (Nov 27, 2026). Gate 1 passes when, through the bench's verifier hook, the builder's checker
certifies the sealed true blueprints of S0, S1 and S4 and rejects every S5a account (`docs/spec/BUILD.md` §6). No
bench exists yet, so no formal attempt has been made. This report covers the checker's development against public
and saved cases.*

## What passed

- **S0's kind (public Tracr programs):** 36 of 36 known cases as expected. True blueprints of frac_prevs, hist,
  reverse and sort_unique, written in the starting vocabulary, are certified; for each, a wrong definition, a wrong
  location, a random location, a rule mislabelled as a lookup, an account given as code, a definition outside the
  grammar, a sixth constant and a cycle are rejected.
- **Saved S5a cases, as given:** every saved copy, near-copy and disguise is Python code, and the checker rejects it
  as not a blueprint.

## What did not pass, and why

- **The three saved real accounts, as given, are also code,** so they are rejected too. They pass only once written
  as blueprints.
- **M2,2** (hidden state of two dimensions, two classes): interchange interventions cannot discriminate there.
  Random directions and the same directions in weight-shuffled networks already score 0.95 to 1.00, so neither the
  real account nor the integer copy clears the control lines. X0 also lists M2,2 as unable to decide anything.
- **M4,3, the deciding case:** written in the vocabulary, a near-copy of the network (its own units, exact weights)
  is certified (run alone 0.9998; 15 of 15 located parts above the lines), and the real account c928 is rejected
  (run alone 0.9902 passes; 22 of 27 parts pass; its channel magnitudes, IIA 0.71 to 0.88, and one class score, 0.84,
  fall below the random-direction line of about 0.90). A copy is true, so truth tests cannot reject it (SPEC R9);
  counting constants does not help (copy 32, account about 50).

## What this means for Gate 1

The checker's truth tests work as designed: they certify true blueprints and reject false ones, and they rejected
every structurally wrong account. They cannot, by design, separate a copy from an account. On networks as small as
AlgZoo's, the constants budget does not stop a copy either: each M4,3 unit has five weights, exactly the budget.
Copies of wider networks fail, because splitting a unit's sum creates parts that are not variables of the network,
and the checker rejects those.

So whether Gate 1 can pass on S5a depends on R9's own tests: answering without running (its outcome is X0's, run by
the bench-writer) and the bounded reader (which needs the frontier-model key). This is put to Shreyas as a decision
(`docs/build/STATUS.md`). The spec is unchanged.
