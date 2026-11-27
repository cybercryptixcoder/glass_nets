# The finished system: a capability specification

*Written Nov 26, 2026, from 19:03 ET (clock). This document defines **what** the finished interpretability system
must be able to do, and **how each capability is tested**. It never says how to build anything: any design that
passes the tests is acceptable. The tests, and the networks they run on, are to be built and sealed **before** the
build starts, so the finish line cannot move during the build.*

*Companions:*
- *`SPEC_SIMPLE.md`: the same thing in plain words.*
- *`GAP.md`: what exists today, what is missing, and the one build that closes the gap.*

*Revised the same evening after an independent critique (`docs/spec/SPEC_CRITIQUE.md`); every change it led to is
listed in `GAP.md` §8.*

*This supersedes `docs/review/NEXT_BUILD.md` §3 as the definition of the target. NEXT_BUILD, its review log, and
Rosetta's write-ups (`docs/rosetta/`) remain the evidence it rests on.*

---

## 0. How to read this

- Each requirement has three parts:
  - **Must:** what the system has to do, in plain words.
  - **Why:** the evidence that makes it necessary, from this project's record or the literature.
  - **Test:** how anyone can check it, and what is reported.
- **No pass line is chosen here by looking at results.** Each line is named by the control that sets it, as the
  project's rules require (`CLAUDE.md`). Most lines are set by one of these:
  - the **true answer**: on a network built with a sealed, known mechanism, the true account's own scores, with
    their spread across seeds and data splits, set the bar;
  - **controls**: untrained networks, shuffled activity, random directions and memorisers set the floor;
  - **baselines that use no explanation**: the task rule alone, attribution patching, "no change", and a copy of the
    network.
- "The box" means the finished system. "The network" means the trained model being explained.

---

## 1. The system in one paragraph

You give the box three things:
- a trained network;
- the behaviour you want explained, either what it was trained to do or a named behaviour such as "gives false
  answers when instructed to";
- inputs on which that behaviour happens.

It gives back a **verified blueprint** of how the network produces that behaviour:
- a one-page overview of the network's strategy;
- the systems that carry the strategy out;
- every working part, each with a definition and a test certificate;
- the wiring between the parts;
- a trace of any single decision, from input to output;
- a ledger of exactly how much is explained and what is left.

Nothing in the blueprint is taken on trust. Every statement has been checked by changing the network and seeing the
predicted effect.

---

## 2. What "fully interpretable" means here

This is the definition everything else serves. A blueprint makes a network **fully interpretable for a behaviour**
when all five of these hold, each by test.

1. **Complete.** Everything that matters to the behaviour is in the blueprint.
   - Scramble everything the blueprint says does not matter, and the behaviour survives.
   - The part that is not explained is measured, located, and below the completeness line (§6).
2. **Honestly typed.** Every part is exactly one of four types, and a test assigns the type, not the box's
   preference:
   - a **rule**: a definition that holds on inputs it was never fitted on;
   - a **lookup**: a memorised table, listed, with its size, whose effect does not carry over to fresh inputs;
   - **irrelevant**: removing it changes the behaviour less than the irrelevance line (set on planted networks by
     the effect of removing their sealed unused parts);
   - **unresolved**: the measured gap. The lookup and unresolved shares appear in the headline of every blueprint.
3. **Runs on its own, and is faithful at every level.**
   - **Run alone**, with every definition standing in at once, lookups as their tables, and unresolved parts
     resampled and charged to the ledger, the blueprint reproduces the network's outputs and its intermediate states,
     level by level. Substituting one definition at a time is not enough: Rosetta's definitions passed one by one,
     but run together they kept almost none of the evaluation (R² 0.06; `docs/rosetta/RESULTS.md` §4, §9).
   - Every variable the blueprint names exists in the network, and intervening on it in the network has the effect
     the blueprint predicts.
4. **Navigable, though not necessarily short.**
   - The top level fits on a page, and each level is defined in terms of the level below.
   - The bottom level reaches the network's actual components.
   - Real networks often compute with large collections of simple heuristics rather than one neat algorithm
     (Nikankin et al., ICLR 2025, "a bag of heuristics"). For them the full blueprint is long, but every entry is
     simple and checked, and the top level still fits on a page.
   - A building's blueprint works the same way: one drawing of the whole, and a specification for every beam.
5. **Usable by a mind.** A person, or a language model not allowed to run code, can use the blueprint to predict the
   network. It must predict better than with no blueprint, and better than with a copy of the network.

**"Provably"** adds a sixth condition:
- Every claim states the kind of evidence behind it: an exhaustive check, a proven bound, a statistical test with an
  interval, or an adversarial search that found no counterexample.
- The box's stated confidence is calibrated on networks whose answers are known.

**What the definition deliberately does not require:**
- **That the network be simple.** The box reports the network's real shape and never invents a neater one.
- **That the network contain new human knowledge.** Whether a chess engine knows a strategy nobody has named is a
  fact about the engine. The box's job is to find it if it is there. If it is not, the box can say so only within
  the explained share, with the unresolved share stated.
- **That every unit be explained for every behaviour.** The scope is the behaviour asked about.

**Why this definition, and not "a short program equal to the network":**
- Short programs equal to the network exist only for clean networks. The project showed this: readable approximate
  accounts cannot be faithful on fragile AlgZoo networks (`docs/glass/INTENT.md`, addendum), while exact
  decompositions can be.
- Real networks memorise. A complete account must be allowed to say "this part is a table", but it must not be
  allowed to hide a rule as a table, or a table as a rule. ARC's surprise accounting asks the same question: when a
  network memorises facts, is a lookup table a valid explanation of it?
- "Fewer bits than a copy" rejected every real explanation in the project's own pre-registered test (T2,
  `docs/glass/info_measure_T2.txt`). Completeness plus honest typing plus usability does not depend on shortness.

---

## 3. Scope: the networks and behaviours it must handle

| Tier | Domain | Examples | Size | Behaviours |
|---|---|---|---|---|
| A | Small algorithmic networks | AlgZoo RNNs, modular arithmetic, compiled and semi-synthetic transformers | up to about 10^6 parameters | their trained task |
| A | Game networks | Leela chess networks (LD2, 744706, t1); smaller trained players | millions of parameters (LD2 is about 3 million) | move choice and evaluation |
| A | Learned simulators | a fluid emulator trained on coarse-grained frames of a finer simulation (rebuilt; see GAP.md) | about 10^6 to 10^7 | next frame, and the correction it learned |
| A | Small language models | GPT-2 small and medium | about 10^8 | named behaviours (date arithmetic, verb agreement, planted behaviours) |
| B | Open language models | 7–8B models | about 10^10 | lying when instructed; reading another network's weights |

- **Tier A is the "narrow realm".** When the build is done, the box must make Tier A networks fully interpretable in
  the sense of §2, for the behaviours listed.
- **Tier B must run end to end,** with completeness measured and reported. "Fully" is not required there yet.
  - The reason is evidence, not caution.
    - On Claude 3.5 Haiku, the best published tools gave satisfying insight on about a quarter of prompts, and their
      replacement left about 22% reconstruction error.
    - On the smaller research model in the same work, the replacement matched the model's output in about half of
      cases.
    - Both leave error terms that cannot be interpreted (Anthropic, circuit tracing, 2025).
  - So the two language-model flagships (lying; reading weights) will get measured, partial accounts at Tier B, not
    complete ones, until Tier B completes.
- **GPT-2 behaviours count as Tier A only after S7 passes on GPT-2** (month rollover). Until then they are Tier B.
  - Tier B is where the box should be pushing the frontier, not where its completion is judged.
- **Out of scope for this build:** models above about 8B; accounts of everything a language model does at once;
  vision models (one can be added later by adapter).

---

## 4. The requirements

### Getting a network in

**R1. It takes any network in scope, with only an adapter.**
- *Must:* explain a new network without new pipeline code. All the box needs from a new domain is an adapter:
  - how to run the network;
  - how to intervene at a named place inside it;
  - what the inputs and outputs mean;
  - optionally, a starting vocabulary and domain operations (legal moves, spatial derivatives, token positions).
- *Why:* every sprint in this project built a new, bespoke pipeline (`docs/PROJECT_REVIEW.md`;
  `docs/review/NEXT_BUILD.md` §1). "Hand it any model" means one box.
- *Test:*
  - Every network in the acceptance suite (§6), across all domains, goes through the same box, with one start
    configuration and no change to its core between networks.
  - An adapter may contain only the four items above. A domain vocabulary is a list of primitives, not a pipeline.
  - Reported: the size and contents of each adapter.

### What it tells you

**R2. It says what the network does.**
- *Must:* state the network's input-to-output behaviour on the inputs given, in the task's own terms, including where
  the network departs from the task. That covers:
  - its mistakes, and its quirks;
  - what it knows beyond the reference rules;
  - the shortcuts it takes compared with a reference method, where there is one (for example, a fluid emulator
    against the solver it imitates);
  - where it will hold up, and where it will fail, on unusual inputs outside the ones it was given.
- *Why:* a network's real behaviour can differ from the task it was trained on. In LD2, about 22% of the variance
  in the evaluation is left unexplained by a surrogate fitted on 61 textbook concepts; that share includes the
  surrogate's own error (`docs/rosetta/RESULTS.md` §2). The review asked whether that part is knowledge or the network's own error
  (`docs/review/NEXT_BUILD.md`, flagship 1).
- *Test:*
  - agreement with the network on held-out inputs, with an interval;
  - agreement on the worst slice, and on the network's own mistakes;
  - predictions, made before running, of where the network fails on a pre-registered set of unusual inputs;
  - the **knowledge split**: of the network's departures from the reference, the share that predicts ground truth
    (game results, the fine simulation, true facts) and the share that does not.

  The line is the true account's own agreement on the planted networks. On real networks it is reported against the
  task-rule baseline.

**R3. It shows how, inside, faithfully, at every level.**
- *Must:* every variable in the blueprint is located in the network: a place in the network, plus a direction or
  feature under a declared, restricted map. Two things must then hold:
  - intervening there changes the network the way the blueprint predicts;
  - putting the blueprint's version of the variable in place of the network's own preserves the behaviour.
- *Why:*
  - Matching outputs is not matching mechanism. The task rule itself matches most outputs.
  - With unrestricted maps, any network "implements" any algorithm (Sutter et al. 2025, cited in
    `docs/glass/EVIDENCE_LADDER.md`).
  - The project's judge and Rosetta's substitution test already do this well.
- *Test:*
  - interchange-intervention accuracy for each variable;
  - substitution, as the share of the variable's effect restored.

  Both are reported against untrained networks, shuffled activity and random directions. The line is family-wise:
  the 99th percentile of the best control score among as many control variables as real ones tested. A
  per-variable line lets chance passes through when thousands are tested, and Rosetta's untrained network already
  produced one (`docs/rosetta/FINDINGS.md` 1). On planted networks the true account's spread also applies.

**R4. It accounts for everything, and says exactly what is left.**
- *Must:* cover the whole computation behind the behaviour, in two senses:
  - **Kept parts suffice.** If everything the blueprint calls irrelevant, or does not mention, is replaced by its
    value on other inputs, the behaviour survives.
  - **The blueprint alone suffices.** Run on its own (§2, condition 3), it reproduces the behaviour.

  What does not survive is the unresolved share, and it is measured and located by layer and by part.
- *Why:*
  - Partial accounts look complete.
  - Attribution graphs reach exact equivalence only by adding error terms that are unlabelled, not sparse, and never
    broken down further.
  - Redwood's causal scrubbing turned completeness into a number: how much of the network's performance survives
    when only what the explanation names is kept (Redwood Research, 2022).
  - An average over inputs can hide a wrong account by cancellation: better on some inputs, worse on others. The
    fix is to look at recovery input by input (Practical Pitfalls of Causal Scrubbing).
- *Test:*
  - **Resampling.** The bench, not the box, fixes the sets before the run. They include sets that respect the
    blueprint and adversarial sets.
  - **What is measured.** Recovery input by input, on every output the network has (both heads of a chess network,
    for example), as KL divergence or squared error.
  - **Reported:** the lower tail of recovery input by input (its 5th percentile), the mean, the joint run of §2
    condition 3, and the unresolved share by layer.
  - **The completeness line** is set on the planted networks: the 5th percentile of the true blueprints' recovery
    across the bench's development draws, minus its spread.
  - **For a real network,** "complete" is judged only against a rehearsal network of matched size and architecture
    (for LD2, a planted network with LD2's own architecture). Without one, its unresolved share is reported with an
    interval, and the word "complete" is not used.

**R5. It types every part honestly: rule, lookup, irrelevant or unresolved.**
- *Must:* assign each part's type by test:
  - **rule:** its definition holds on held-out inputs at the fidelity line;
  - **lookup:** it agrees with its table on the memorised items and is at chance elsewhere. The table is listed.
    Its keys must be verified training items where the training data is known. A part whose causal effect on fresh
    inputs is above the control line can never be typed lookup, whatever its keys;
  - **irrelevant:** replacing it changes the behaviour less than the irrelevance line;
  - **unresolved:** anything that is none of the above.
- *Why:*
  - Networks memorise, and they use bags of heuristics.
  - The atlas already labels rule against lookup on small networks, but its memoriser test passed on a count of one
    (`docs/glass/ATLAS_RESULTS.md`, A41).
  - Without the rules above, any part the box cannot define could be filed as a table over the inputs it was shown.
- *Test:*
  - on planted networks that mix a rule, a memorised table and unused parts, the box's types match the sealed types;
  - memoriser controls come out as lookup;
  - no sealed rule is labelled lookup, and no table is labelled rule;
  - at least 20 cases per type, per tier.

**R6. It goes from purpose to parts, and back.**
- *Must:* give at least three levels:
  - an **overview**: the network's strategy in the task's own terms;
  - **systems**: groups of variables that serve one function, such as look-ahead or material evaluation;
  - **components**: the individual variables.

  Each level is defined in terms of the level below and checked against the network at that level. Any statement can
  be zoomed into, down to the units that carry it.
- *Why:* this is where the last build broke.
  - Rosetta defined each of LD2's last-layer variables directly from board features in one leap.
  - The variables behind the network's evaluation are verdicts ("we are winning", "we are losing", "drawish"). They
    appear to be assembled over many layers: the top 16 carry 56%, 66% and 91% of the extra knowledge at blocks 6, 8
    and 10. That figure is exploratory, with no controls run at blocks 6 and 8.
  - Only 2 of those 16 could be defined, and the rebuilt layer kept almost none of the evaluation (R² 0.06;
    `docs/rosetta/RESULTS.md` §9).
  - The working hypothesis is that a verdict is simple in terms of its ingredients and nearly impossible to state in
    terms of the board. Rosetta's evidence is consistent with this but does not prove it; Gate 2 of `GAP.md` tests
    it.
- *Test:*
  - **Planted networks:** on networks whose sealed mechanism has intermediate variables computed in stages, every
    intermediate variable is recovered (its definition matches the sealed one at the recovery line), and so is the
    wiring between them. The overview matches the sealed top level.
  - **Real networks:** every level passes R3, and the overview alone predicts the behaviour better than the task-rule
    baseline.

**R7. It traces any single decision, and fingerprints any behaviour.**
- *Must:*
  - for any input, show the path of parts that produced the output, how much each contributed, and the smallest
    change (to the input, or to one part) that would flip the decision;
  - for a named behaviour (a lie, a blunder, a planted trigger), list the parts whose activity marks it: its
    fingerprint, usable as a detector.
- *Why:* when a structure fails, an engineer can name the member that gave way. The box must do the same for any
  decision: debugging, auditing a specific lie, explaining a specific move.
- *Test:*
  - **Flips.** The bench, not the box, chooses at least 20 decisions per domain. Predicted flips come true more
    often than the baselines' predictions (attribution patching, and "no change"). The bench's own search checks
    that each flip found is minimal.
  - **Fingerprints.** A fingerprint detects its behaviour on held-out cases better than a probe trained on the raw
    activations without the blueprint, the explanation-free baseline. For lying, that includes lies the network was
    not instructed to tell, where such cases exist.

  Both are reported per domain.

**R8. It names what nobody has named.**
- *Must:* when a variable cannot be defined in the current vocabulary, the box adds to the vocabulary:
  - It produces new concepts or new kinds of quantity, with names and definitions.
  - Each new concept then passes R3 like any other part.
  - Each joins a library the box uses on later networks.
- *Why:* every instrument in the project stopped at the edge of the vocabulary it was given, and the unexplained part
  came back as "residual":
  - Kepler's chess concept library left about a fifth of the value unexplained;
  - Rosetta's translations closed about 7% of that gap.

  The one new kind of quantity added by hand in the last build was the material won along the network's own planned
  line. It carried a sixth to a third of LD2's extra knowledge, yet no single one of the 16 evaluation variables
  holds it (correlations −0.12 to +0.13; `docs/rosetta/RESULTS.md` §9). So it is not a passing concept: it would
  fail R3. It is evidence that the missing ingredients are of a new kind. Discovery means adding such concepts. Concepts found inside AlphaZero have been taught to grandmasters, so the step can be
  done and tested (Schut et al., PNAS 2025).
- *Test:*
  - **Planted networks:** on networks built around concepts that cannot be written in the box's starting vocabulary,
    the concepts are recovered at the recovery line. The starting vocabulary is fixed and published before a
    separate session writes the sealed concepts.
  - **Half of those concepts are synthetic, with no human name,** and they are scored separately. A strong model can
    name a known chess idea from memory, which is recall, not invention.
  - **Real networks:** on each Tier A real network, the vocabulary-growth curve is reported: the unresolved share
    after each concept the box adds.

**R15. It explains what the network's efficiency comes from.**
- *Must:* where the network does a job that a known method also does, say which parts of the network stand in for
  which steps of the known method, which steps it skips or approximates, and what that costs in compute and in
  error. A fluid emulator against the solver it imitates is the example.
- *Why:*
  - "Why is the fluid model so cheap?" is one of the four questions this system exists to answer.
  - The current emulator is not cheap at all: by an arithmetic count it costs about 47 times its solver's work per
    frame (`docs/review/REVIEW_LOG.md`, F43). So the answer depends on a rebuilt target and on this capability.
  - No other requirement deals with compute.
- *Test:*
  - **Rehearsal (S6):** an emulator trained with a sealed shortcut, such as a longer time step with a sealed
    correction. The blueprint must name the shortcut, the parts that carry it, and its cost and error, at the
    recovery line.
  - **Real network:** the rebuilt emulator's blueprint is reported against the solver.

### How it guarantees what it says

**R9. It cannot be fooled by copies, surrogates or stories.**
- *Must:* reject any account that is the network in disguise, or a surrogate fitted to it. On networks with no
  structure, report lookup or unresolved, never a story.
- *Why:* a copy of a network is perfectly true, so it passes every test of truth. Every rule written against copies
  in this project was beaten:
  - rewritten numbers (`docs/glass/REPORT_ROUND2.md`);
  - the bits measure, which rejected real accounts (T2);
  - Goal 2, which gamed its score.
- *Test:*
  - **Rejected:** every case in the must-fail set (§6), including disguised copies, distilled small students,
    shuffled-weight and untrained networks, memorisers, and programs carrying tables tuned by selection.
  - **Two tests separate real accounts from copies**, on the saved runs and the planted suite:
    - **answering without running**: estimating the network's behaviour with less compute than sampling it;
    - **the bounded-reader test** (R10).

    Both are designed in `docs/review/PREREG_FIRST_EXPERIMENTS.md`, X0.

**R10. A mind can use it.**
- *Must:* a person, or a language model barred from running code, can use the blueprint to predict the network's
  behaviour on new inputs. It must beat the same reader given only the task, and the same reader given a copy. When
  the network knows something people do not, the blueprint must be able to teach it.
- *Why:*
  - Interpretability is for understanding: human-grounded evaluation is one of the field's three standard kinds
    (Doshi-Velez and Kim, 2017).
  - A correct blueprint nobody can use is a copy by another name.
  - Rosetta's reader tests barely worked: with positions as text, the cheap model applied board rules at 0.00 to
    0.13 against 0.05 chance; with boards drawn as grids, one rule reached 0.21 against a rival's 0.04. Its own
    write-up calls teachability "essentially untested" (`docs/rosetta/RESULTS.md` §8). Readability has to be
    measured, not assumed.
- *Test:*
  - **The reader test:** the reader may not run code, and its budget per question is below the cost of
    hand-executing the smallest opaque program in the comparison. It may open parts of the blueprint on demand
    within that budget, so a long blueprint is navigated, not read whole.
  - **The line** is the true blueprint's reader score on the planted networks, minus its spread. Also reported: the
    share of the gap between the task-only floor and the network that the blueprint closes.
  - **The copy condition** is a sanity check, not a discriminating test, because the budget is set so a copy cannot
    be run.
  - **Teaching new chess concepts:** for any new chess concept, a study modelled on Schut et al. (players improve on
    prototype positions after learning the concept).

**R11. You can change the network on purpose with it.**
- *Must:* make targeted edits through the blueprint (switch off, strengthen or redirect a part or a capability) and
  predict their effects, including side effects, beforehand.
- *Why:* control is the strongest proof of understanding, and it is the safety use case: switching off a lying gate.
  It must beat explanation-free methods. On held-out interventions, attribution patching once predicted the
  network's decision better than the project's own construction: 90.8% against 82.3% overall, and 57.0% against
  13.9% when the decision changed (`docs/EXPERIMENT_LOG.md`, E7).
- *Test:*
  - The bench, not the box, chooses at least 20 edits per domain, including "remove a capability" edits.
  - Each edit's effect is predicted before it is run, and compared with the outcome.
  - The predictions must beat attribution patching and "no change".

**R12. It says how sure it is, and it is right about that.**
- *Must:* every claim carries its kind of evidence: exhaustive, proven bound, statistical with an interval, or
  adversarially searched. On networks with known answers, claims made at a stated confidence are right at that rate.
  Where two different accounts pass the same tests, both are reported, with the experiment that would separate them.
- *Why:*
  - The project's history of over-claims (`docs/review/REVIEW_LOG.md`).
  - Mechanistic interpretability is not identifiable in general: several accounts can pass the same tests (Méloux et
    al. 2025, cited in `docs/glass/EVIDENCE_LADDER.md`).
- *Test:*
  - calibration on the acceptance suite;
  - the suite includes planted networks with two valid accounts, and both must be reported.

### Practical requirements

**R13. It works at the real size, in every domain in scope.**
- *Must:*
  - Tier A networks become fully interpretable (R1 to R12, and R15 where a reference method exists, pass).
  - Tier B behaviours run end to end, with completeness and cost reported.
  - Each network stays within a compute budget stated before it is run.
- *Why:* the flagships Shreyas named are real networks: a chess engine, a learned fluid model, a lying language
  model, a model that reads weights.
- *Test:* the acceptance suite plus the real Tier A targets, and the reported cost per network.

**R14. It does the research itself.**
- *Must:* given the network, the behaviour and the inputs, the box runs the whole loop:
  - finding parts;
  - proposing and testing definitions;
  - inventing concepts;
  - checking completeness;
  - stopping when the blueprint is complete, or when the budget runs out.

  It chooses its own experiments and logs every one. A person can steer it but is not needed.
- *Why:*
  - This is what "hand it a model" means.
  - So far in this project, the hard results came from a strong agent working by hand
    (`docs/review/NEXT_BUILD.md` §1).
  - Agents that design and run interpretability experiments themselves have been built for vision models (MAIA,
    ICML 2024).
- *Test:* the whole acceptance suite run end to end, from one start configuration, with no human input after the
  start, with a full log.

---

## 5. What a blueprint contains

The content, not a file format:

1. **Overview.** The network's strategy in the task's own terms, on one page, with how much of the behaviour that
   page alone predicts.
2. **Systems.** Groups of parts that serve one function. Each has its role, what feeds it and what it feeds.
3. **Components.** Every part, with:
   - where it is in the network;
   - its type (rule, lookup, irrelevant, unresolved);
   - its definition, in terms of earlier parts or the inputs;
   - its test certificate: the tests passed, the numbers, and the kind of evidence.
4. **Wiring.** Which parts feed which, with strengths.
5. **Traces.** For any input: the active path, each part's contribution, and what would flip the decision.
6. **Ledger.** The share of the behaviour carried by each type, and the unresolved share by layer and by part.
7. **Library links.** Which parts are instances of concepts already in the library, and which concepts are new.
8. **Rivals.** Accounts that pass the same tests, and the experiment that would tell them apart.

---

## 6. The acceptance suite: the finish line, built first

The suite is the certification. It is built, and its answers sealed, before the build starts.
- The sealed answers are written by a separate session that never shares context with the builder.
- The box never reads them.
- **What is sealed is a generator, not a fixed set of networks.**
  - The build works on development draws.
  - Certification uses one fresh draw, scored once.
  - Every retry of a gate gets a new draw, so no sealed answer leaks through repeated scoring. The atlas did the same,
    giving each retry fresh seeds.
- **Planted networks must actually compute their sealed mechanism.** Training a network on outputs does not make
  the sealed intermediate variables exist inside it.
  - Planted networks are trained with interchange objectives, as InterpBench's strict interchange-intervention
    training (SIIT) does.
  - Each sealed intermediate variable is checked by intervention before sealing.
  - Any network below a fixed fidelity is dropped. Rosetta's own planted network learned its rule only to R² 0.656
    (`docs/rosetta/RESULTS.md` §1).

| Tier | What it is | What it proves | Requirements tested |
|---|---|---|---|
| S0 | Compiled transformers (Tracr) and semi-synthetic transformers with known circuits (InterpBench, 86 models) | The box recovers exactly known circuits in realistic weights | R3, R4, R6 |
| S1 | Planted rules written in the box's starting vocabulary (Rosetta's yardstick, extended) | Search and scoring work | R2, R3 |
| S2 | Planted concepts that cannot be written in the starting vocabulary; half of them synthetic, with no human name, scored separately | It can invent, not only recall or search | R8 |
| S3 | Planted hierarchies: a verdict computed in stages from intermediate concepts, including a search-like step. The chess version is a small value network trained to imitate a sealed evaluation that looks two plies ahead along forcing moves and adds positional terms | It can define verdicts through their ingredients: the exact shape that broke the last build | R6, R7, R8 |
| S4 | Planted mixtures: part rule, part memorised random table, part dead units; plus networks with two valid accounts | It types honestly, and reports rivals | R4, R5, R12 |
| S5 | Must-fail cases (below) | It cannot be fooled | R9 |
| S6 | One rehearsal per flagship, sealed: chess (S3's network); a fluid emulator trained on data with a sealed closure term and a sealed shortcut; a small language model with a planted gate that switches it to false answers; a small reader network trained to report a property of another network's weights by a sealed method | Each flagship's shape is handled before the real target | all, including R15 |
| S7 | Real networks with partly known answers: LD2 and t1 (look-ahead along forcing lines; the capture variable that passed Rosetta's checks, at the margin and after a declared second step); GPT-2 month rollover; modular multiplication; AlgZoo M4,3 | It reproduces what is already known about real networks | all except R8 |

**The must-fail set (S5):**
- every saved copy and disguise from rounds 1 and 2;
- distilled small students of each network;
- shuffled-weight and untrained networks;
- memorisers;
- programs carrying literal tables tuned by selection;
- networks with no structure (which must return lookup or unresolved, not a story).

**How lines are set.** Every line comes from the true answer or from controls, never from the results it judges.
- **Recovery:** a sealed concept counts as recovered when the box's definition passes two tests on held-out inputs:
  correlation, and substitution in the network. Each must reach the sealed definition's own score minus its spread
  across development draws. A fixed constant such as Rosetta's 0.9 correlation is not used: it ignores scale, offset
  and rare decisive cases.
- **Completeness:** the 5th percentile of the true blueprints' recovery, input by input, minus its spread (R4).
- **Must-fail:** family-wise, the 99th percentile of the best control score (R3).
- **Usability:** the true blueprint's reader score, minus its spread (R10).

**Pass counts, fixed before the build.** These are defaults for Shreyas to confirm or change before the bench is
sealed; partial passes are failures:
- S0, S1, S3, S4 and S5: every case passes, and every must-fail case is rejected;
- S2: at least 80% of the named concepts, and at least 80% of the synthetic ones, each scored separately;
- S6: every rehearsal passes every requirement it tests;
- S7: every known answer is reproduced.

---

## 7. When the build is done

- **Done** means:
  - on the certification draw, every requirement passes on S0 to S6 at the pass counts fixed in §6;
  - S7 reproduces its known answers, with completeness reported;
  - the full suite runs end to end with no human input (R14).
- **Then the real flagships are run.** What they produce are findings, not pass or fail: the chess strategies, the
  fluid closure, the lying mechanism, how a model reads weights.
- **A failure names a requirement.** Each requirement maps to a component in `GAP.md`, so a failure points at one
  member, not at the whole structure.
- **The spec is frozen during the build.** A change is a new, named version with its evidence, as the project's rules
  require for measurements.

---

## 8. Limits, stated plainly

- **A network that is mostly memorised gets a complete blueprint that is mostly tables.** That is a true account of
  that network, not a failure of the box.
- **At Tier B, completeness may stop short within budget.** Sparse replacements of language-model layers still leave
  about 11–22% reconstruction error (`main:docs/literature/SYNTHESIS.md` §3, "dark matter"). The box must report where
  completeness stops, not paper over it.
- **Two different blueprints can pass the same tests.** The box reports rivals and the experiment that would separate
  them (R12).
- **A teachable new concept is a finding, not a guarantee.** The box can show whether the network holds one; it
  cannot put one there.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| Behaviour | What is being explained: the trained task, or a named behaviour, on stated inputs |
| Blueprint | The box's output: overview, systems, components, wiring, traces, ledger, library links, rivals |
| Part, component, variable | A quantity the network computes, located at a place in the network under a declared map |
| Rule, lookup, irrelevant, unresolved | The four types every part must have, each assigned by test (R5) |
| Interchange intervention | Copy a variable's value from one input's run into another's, and check that the network changes as the blueprint predicts |
| Substitution | Replace the network's variable with the blueprint's definition of it inside the running network; check that behaviour is restored |
| Resampling ablation | Replace a part's value with its value on a different input; used to test what matters (completeness) |
| Completeness line | The worst-case share of behaviour the true blueprint keeps on planted networks, minus its spread |
| Planted network | A network trained so that its mechanism is known, with the answer sealed |
| Must-fail case | Something that must never pass as an explanation: copies, students, memorisers, shuffled networks |
| Bounded reader | A language model barred from running code, with a budget per question too small to execute the network by hand |
| Starting vocabulary | The primitives the box begins with, fixed and published before the sealed concepts are written |
| Tier A, Tier B | The networks that must become fully interpretable (A), and those that must run with completeness reported (B) |

---

## 10. Sources

**This repository:**
- `docs/rosetta/RESULTS.md`, `FINDINGS.md` and `PREREGISTRATION.md`: the last build, and where it broke.
- `docs/glass/INTENT.md`, `EVIDENCE_LADDER.md`, `ATLAS_RESULTS.md`, `info_measure_T2.txt` and `REPORT_ROUND2.md`.
- `docs/EXPERIMENT_LOG.md` (E7); `docs/PROJECT_REVIEW.md`.
- `docs/review/NEXT_BUILD.md`, `REVIEW_LOG.md` and `PREREG_FIRST_EXPERIMENTS.md`.
- `main:docs/literature/SYNTHESIS.md`: the field's own measures, hurdles and testbeds, drawn from 193 core entries.

**Literature:**
- Nikankin, Reusch, Mueller and Belinkov, "Arithmetic Without Algorithms: Language Models Solve Math with a Bag of
  Heuristics", ICLR 2025 (arXiv 2410.21272).
- Schut, Tomašev, McGrath, Hassabis, Paquet and Kim, "Bridging the human–AI knowledge gap through concept discovery
  and transfer in AlphaZero", PNAS 2025 (arXiv 2310.16410).
- Ameisen et al. and Lindsey et al., "Circuit Tracing" and "On the Biology of a Large Language Model", Anthropic
  2025 (transformer-circuits.pub).
- Chan et al., "Causal Scrubbing", Redwood Research 2022; and "Practical Pitfalls of Causal Scrubbing" (Alignment
  Forum).
- Wang et al., "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small", 2022
  (arXiv 2211.00593): the criteria of faithfulness, completeness and minimality.
- Rott Shaham, Schwettmann et al., "A Multimodal Automated Interpretability Agent" (MAIA), ICML 2024 (arXiv
  2404.14394).
- Gupta, Arcuschin, Kwa and Garriga-Alonso, "InterpBench", NeurIPS 2024 Datasets and Benchmarks (arXiv 2407.14494).
- Ellis et al., "DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning", PLDI 2021.
- Hilton, "Formal verification, heuristic explanations and surprise accounting", ARC 2024.
- Sharkey, Chughtai et al., "Open Problems in Mechanistic Interpretability", 2025 (arXiv 2501.16496).
- Doshi-Velez and Kim, "Towards a Rigorous Science of Interpretable Machine Learning", 2017 (arXiv 1702.08608).
- Gross et al., "Compact Proofs of Model Performance via Mechanistic Interpretability", NeurIPS 2024.
- Sutter et al., "The Non-Linear Representation Dilemma", NeurIPS 2025; Méloux et al., "Is MI Identifiable?", ICLR
  2025.
