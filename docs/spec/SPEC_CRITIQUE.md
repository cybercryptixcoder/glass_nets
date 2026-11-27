# Critique of docs/spec/SPEC.md and docs/spec/GAP.md

*Independent read-only review, Nov 26, 2026, from 19:16 ET (clock). Checked against branch `rosetta`, `main:docs/literature/SYNTHESIS.md` and four web sources (linked at the end).*

## Problems, most serious first

**1. The blueprint is never required to run (SPEC §2.3, R3, R4).**
- *Text:* "putting the blueprint's version of the variable in place of the network's own preserves the behaviour" (R3). "If everything the blueprint calls irrelevant… is replaced by its value on other inputs, the behaviour survives" (R4).
- *Problem:* substitution is tested one variable at a time, and completeness keeps the network's real values for the named parts. Nothing requires all definitions to run together, from input to output.
  - That joint test is exactly where Rosetta broke: value R² 0.062, and 0.074 with richer definitions (RESULTS §4, §9).
  - NEXT_BUILD §3.1, which SPEC supersedes, required it ("Each level is executable… reproduces the network's intermediate states and its output").
  - SYNTHESIS §4b: agreement decays with depth.
- *Fix:* add a joint test. The blueprint run alone (lookups as tables, unresolved parts resampled and charged) must reproduce every output at a line set by true blueprints, level by level.

**2. The sealed suite is both the development set and the exam.**
- *Text:* "Final scores are computed once" (SPEC §6). But "Each gate has a pass test from the suite" (GAP §4), and a Gate 2 failure leads to retries on the same S2 and S3.
- *Problem:* each retry leaks the sealed answers through the scores. The atlas avoided this with fresh seeds per retry (A18 s0, A27 s1, A28 s2; glass DIARY lines 1457 and 1540).
- *Fix:* seal a generator, not a fixed set. Gates use development draws, and certification uses one fresh draw, scored once.

**3. Planted "true blueprints" may not match what the planted networks compute, yet lines come from them.**
- *Text:* "a small value network trained to imitate a sealed evaluation" (S3). "below the largest unresolved share any true blueprint shows on the planted rehearsal network" (R4).
- *Problem:* training on outputs does not create the sealed intermediates. InterpBench needed strict interchange-intervention training (SIIT) for exactly this reason. Rosetta's planted S2 reached only R² 0.656 (RESULTS §1). So:
  - a correct box can fail R6;
  - a sloppy planted network loosens the completeness line;
  - "largest" is the most lenient choice;
  - a toy network's number is transferred to LD2.
- *Fix:* train planted networks with interchange objectives and verify the sealed intermediates by intervention before sealing. Use a percentile, not the largest value. Call a real network complete only against a rehearsal of matching size.

**4. Two pass lines lower the bar (SPEC §6).**
- *Usability* ("the task-only floor, plus the floor's spread"):
  - It is the only line not set by the true account, so the task rule plus a sliver passes.
  - "Beats a copy" cannot fail, because the reader's budget is below the cost of hand-executing the copy.
  - The spec never says how a bounded reader uses a long "navigable" blueprint (§2.4).
- *Recovery* ("correlation of at least 0.9"): a fixed constant, against §0 ("Each line is named by the control that sets it"). Correlation also ignores scale and rare decisive cases.
- *Fix:* set the usability line at the true blueprint's reader score minus spread, and report the share of the floor-to-network gap closed. For recovery, require correlation and substitution, each at the sealed definition's own level.

**5. The typing can be gamed (R5).**
- *Text:* "agrees with its table on the memorised items and is at chance elsewhere".
- *Problem:* the items are not tied to training data, and Leela's and GPT-2's training sets cannot be checked. Any sparse feature the box cannot define can be listed as a table. "The network's own noise" is undefined for a deterministic network.
- *Fix:* table keys must be verified training items, and any part with an effect on fresh inputs is not a lookup. Define noise as the spread across retrained seeds. Headline the lookup and unresolved share, and use 20 or more cases per test (the A41 lesson).

**6. Completeness is underspecified, and the cited fix does not answer the pitfall.**
- *Text:* "the worst case, over a fixed set of resampling choices" (R4).
- *Problem:* the Pitfalls post locates cancellation across inputs and recommends per-sample loss; "worst case" appears only in a comment. The spec also does not say who picks the resampling, or which metric is used. That matters: at K = 64, value R² is 0.918 but top-move agreement is 0.402 (RESULTS §9). NEXT_BUILD §3.2's distribution-level measures were dropped.
- *Fix:* the bench fixes the resampling, both hypothesis-respecting and adversarial. Report the lower tail of per-input recovery on every output head, using KL or MSE.

**7. The flagships are not covered.**
- *Problem:*
  - No requirement asks why a fluid model is cheap. The current emulator costs about 47 times its solver's arithmetic (REVIEW_LOG F43), and a cheap rebuild is not a precondition.
  - A lying "fingerprint" means a detector that generalises, which nothing requires.
  - Lying and weight-reading are Tier B, where "fully" is never required.
- *Fix:* add a cost requirement, tested on a rehearsal with a sealed shortcut. Add a detector test on held-out and uninstructed lies. Say plainly that flagships 3 and 4 get partial accounts.

**8. GAP overpromises.**
- *Text:* "does give the system you described" and "It cannot end in 'marginally better'" (§1). "Gates 3 to 6: … methods the field has demonstrated… a translation error" (§4).
- *Problem:* the system Shreyas described includes all four flagships. With no k-of-n pass counts fixed, partial passes bring back "marginally better". Several steps are not demonstrated:
  - worst-case completeness at scale: Haiku's replacement leaves 21.7% error and gave satisfying insight on about a quarter of prompts;
  - beating attribution patching: in E7 it scored 90.8% and 57.0%, against the construction's 82.3% and 13.9%;
  - LLM-simulated layers, which SYNTHESIS §3.7 describes as "about as badly as deleting the layer".

  Labelling failures as translation errors in advance also breaks CLAUDE.md rule 6.
- *Fix:* say "Gate 2 answers whether the approach works; the system exists only if Gates 2 to 6 pass." Pre-register pass counts.

**9. Smaller holes.**
- *GAP §4, Gate 3 ("S6's four rehearsals pass"):* S6 tests all requirements, so it needs C7 to C10 from Gates 4 to 6. Run only S6's R3 to R7 parts there.
- *S2:* a frontier observer can recall known chess concepts. Add synthetic concepts that have no human name.
- *R7 and R11:* the box picks its own flips and edits, and "at least one 'remove a capability' edit per domain" is a count of one. The bench should choose 20 or more.
- *R12:* rivals are untested without planted cases that have two valid accounts.
- *R3:* a 99th-percentile line over thousands of variables passes some by chance (Rosetta's untrained network did, RESULTS §1). Use a family-wise line.

**10. Factual errors.**
- *SPEC §3:* the "about half" output match is the 18-layer model's, not Haiku's.
- *SPEC R2:* the "22% beyond textbook chess" is variance left by one gradient-boosted surrogate, including its own error.
- *SPEC R10 and GAP §5:* there were two reader tries, not one. The grid version scored 0.21 against 0.04 on "in check", and RESULTS §8 calls teachability "essentially untested".
- *SPEC R8 and GAP C5:* the line swing is not a located variable. None of the 16 carries it (r −0.12 to +0.13), so it fails R8's own R3 clause.
- *SPEC R6 and GAP C4:* the 56%, 66% and 91% figures are exploratory, with no controls at blocks 6 and 8. "A verdict is simple in terms of its ingredients" is the Gate 2 hypothesis, not a finding.
- *SPEC S7:* the capture variable passed only at the margin (0.501 against 0.5).
- *GAP:* R5 is called both "never built" (§1) and "partly built" (§2).

## What is sound
- *Spot-checks that hold:* 41% and R² 0.06; 45 proposals from 18 calls; the atlas's 5 of 6 held out; A41; E7; the 47 times figure; liar sprint probes only; every cited file exists (`helix_evolve` is a separate repo).
- *Design:* capabilities rather than implementation, sealing by a separate session, must-fail sets, lines set by controls, rival accounts, Tier A and Tier B, and the risky gate placed second.
- *Citations:* InterpBench (86 models), the quarter of prompts, 11–22% dark matter, Schut et al., MAIA, Nikankin et al. and Méloux et al. are all correct.

*Web sources:* [Pitfalls](https://alignmentforum.org/posts/DFarDnQjMnjsKvW8s/practical-pitfalls-of-causal-scrubbing), [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html), [Biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html), [InterpBench](https://arxiv.org/html/2407.14494v3).
