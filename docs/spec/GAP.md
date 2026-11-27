# From where we are to the finished system: the gap, and the one build that closes it

*Written Nov 26, 2026, from 19:03 ET (clock); revised the same evening after an independent critique (§8). It reads
against `SPEC.md`, which defines the finished system as 15 requirements (R1 to R15), each with a test. This
document:*
- *lists, requirement by requirement, what exists today and what is missing;*
- *defines the components the next build must deliver;*
- *orders them so the uncertain part is tested first;*
- *states plainly what passing would and would not guarantee.*

*It names what each component must achieve and how it is checked. It does not prescribe how to build any of them;
where the evidence points to a promising route, that is marked as a lead, not a requirement.*

*If you are the builder, start at `BUILD.md`. The sealed suite is written by a separate session from
`BENCH_BRIEF.md`.*

---

## 1. The short answer

**The three fixes named earlier are not enough on their own.** Those fixes were:
- defining verdicts through their ingredients, layer by layer;
- a language of operations rather than static features;
- an inventor with an observer that can perceive the domain.

They repair the member that broke in Rosetta, which is R6 and R8 in the spec. But the spec has 15 requirements.
Several of the others are only partly built, and some have never been built at all:
- complete accounting of a real network, including running the blueprint on its own (R4: never built);
- honest typing at real scale (R5: partly built, on small networks only);
- tracing single decisions and fingerprinting behaviours (R7: never built);
- usability by a mind (R10: never shown);
- control through edits (R11: never built);
- explaining where efficiency comes from (R15: never built);
- running the whole loop unaided (R14: never built).

**You get the system you described for the Tier A networks** (the narrow realm, `SPEC.md` §3) **only if Gates 2 to 6
all pass, and the certification draw then passes**, at pass counts fixed before the build. "Fully interpretable" then means what
`SPEC.md` §2 defines and tests: complete, honestly typed, runnable on its own, faithful at every level, navigable,
and usable by a mind.

**Gate 2 gives a definite answer, early, on whether this approach works at all.** It is the only research-grade gate.
Gates 3 to 6 rest on methods that have precedent in the field, but several have not been shown to the standard this
spec sets:
- worst-case completeness on real networks;
- beating attribution patching at predicting edits;
- complete accounts of GPT-2 behaviours.

A failure at any gate is diagnosed by the project's own routine (a translation error, or an intent error), not
labelled in advance.

**Two of your four questions sit at Tier B.**
- Lying, and reading weights, are about 7–8B language models. This build gives them measured, partial accounts, with
  completeness reported, not complete ones.
- The chess and fluid questions are Tier A. The fluid question also needs the rebuilt emulator, and R15.

**What no build can promise in advance** is the content of what it finds:
- how much of a given network will turn out to be rules rather than tables;
- whether a chess engine holds a strategy nobody has named;
- how far completeness gets within budget on 7–8B language models.

The box measures those things. It cannot decide them.

**How the next attempt avoids being open-ended:**
- The finish line, the acceptance suite, is built and sealed first, as a generator: the build works on development
  draws, and certification is one fresh draw, scored once.
- The pass counts are fixed before the build, so a partial pass is a fail, not "marginally better".
- The one genuinely uncertain part is tested second, before anything is polished.
- Every component has its own pass test.

So the attempt ends in one of two definite states: the suite passes, or a named component fails a named test.

---

## 2. Where we are, requirement by requirement

| Req | What it asks | What exists today | What is missing | Kind of gap |
|---|---|---|---|---|
| R1 | Any network in scope, adapter only | Rosetta's core (`rosetta/core.py`, `engine.py`) is meant to be domain-agnostic. But each domain still has its own pipeline (`lab/` for AlgZoo and GPT-2, `kepler/` for chess and fluid, `rosetta/` for chess) | One core with adapters; the separate pipelines folded in | Engineering |
| R2 | What it does, with the knowledge split | Kepler's behaviour rules and fluid law; Rosetta's residual against textbook chess, and its fresh-games knowledge tests (`docs/rosetta/FINDINGS.md` 4, 6) | A standard knowledge-versus-error split; predictions of the network's own mistakes and of where it fails on unusual inputs; the shortcut comparison against a reference method (for example, the fluid emulator against its solver), in every domain | Engineering |
| R3 | Faithful inside | Strong: the judge's interchange tests (`lab/judge.py`, `lab/lmjudge.py`), Rosetta's substitution, the atlas | Applied at every level of a hierarchy, not one layer at a time | Engineering |
| R4 | Complete, with the remainder measured | Rosetta's one-layer glass model, run as a whole (top move kept 41%, value R² 0.06); the atlas certified 5 of 6 held-out small RNNs | The blueprint run on its own through every level; input-by-input completeness accounting on every output, on real networks | Research at Tier B; engineering at Tier A, but never yet shown on a real network |
| R5 | Honest types | The atlas's rule-versus-lookup labels on small networks, with the memoriser test passing on a count of one | Typing at real scale, tested on planted mixtures | Engineering |
| R6 | Layered, from purpose to parts | None automated. Month rollover was traced by hand. Rosetta measured how the evaluation concentrates across blocks (56%, 66%, 91%) but defined only the last layer | Defining variables in terms of the variables that feed them, through the network, up to an overview | **Research: the member that broke** |
| R7 | Trace any decision | Rosetta's attribution to the value | Per-input traces with predicted flips, and behaviour fingerprints usable as detectors, scored against baselines | Engineering |
| R8 | Name what nobody named | None. Rosetta's "vocabulary growth" only recombined its 61 features. The one new kind of quantity (material along the network's own line) was added by hand. The cheap-model proposer: 45 proposals from 18 calls, none better than exhaustive search | A component that adds new kinds of concept when the vocabulary runs out, proven on planted concepts outside the vocabulary | **Research: the core of discovery** |
| R9 | Not fooled | The restricted program language; partial must-fail sets; the bits measure (T2) failed as a gate; X0 designed, not run | The full must-fail suite (S5a and S5b), and the two copy tests run and passing. Answering without running gates only if X0's main arm separates real accounts from copies on the saved runs | Engineering, with one open question (X0) |
| R10 | Usable by a mind | Two reader tries, both weak: with positions as text, the cheap model applied board rules at 0.00 to 0.13 against 0.05 chance; with boards drawn as grids, one rule reached 0.21 against a rival's 0.04. Rosetta calls teachability "essentially untested" | A reader that can perceive the domain, the reader test passing at the true blueprint's line, and a human study for chess | Resource (a frontier model; people) and engineering |
| R11 | Control by edits | Steering experiments (liar sprint, `lab/steer_calendar.py`). Attribution patching beat the project's construction on held-out interventions (E7) | Edits through the blueprint with predicted effects, beating attribution patching | Engineering |
| R12 | Calibrated confidence | Controls and pre-registration are strong habits. Evidence types are not assigned systematically, and nothing has been calibrated | Evidence labels on every claim, and calibration on the suite | Engineering |
| R13 | Real sizes, all domains | Chess (three Leela networks), AlgZoo, GPT-2 small. The fluid emulator must be rebuilt (as built it costs about 47× its solver's arithmetic). 7–8B models only probed (liar sprint), never decomposed | The fluid rebuild; a GPT-2 adapter on the new core; 7–8B decomposition (transcoders) for Tier B | Engineering and resource (GPUs) |
| R15 | Explains where efficiency comes from | Only an arithmetic count: the current fluid emulator costs about 47× its solver per frame (REVIEW_LOG F43) | The rebuilt emulator, a rehearsal with a sealed shortcut, and a component that maps network parts to solver steps | Engineering, after the rebuild |
| R14 | Does the research itself | The hard results came from a strong agent by hand. `probe/` (the Helix Probe loop) and `helix_evolve` (volume search; a separate repo, `cybercryptixcoder/helix_evolve`) exist | A director that runs the loop end to end on the suite | Engineering and resource (a frontier model) |

**In one line:**
- Verifying is strong (R3).
- Six requirements are partly built (R1, R2, R5, R9, R12, R13).
- Two are research-grade and decide everything (R6, R8).
- Six more have never been built or shown here: R4 on real networks, R7, R10, R11, R14 and R15.
- The rest are engineering that has never been done here, plus two missing resources: a model that can perceive the
  domain, and GPUs.

---

## 3. The components the next build must deliver

Ten default components. Only the tests are fixed: any design that passes every requirement's test is acceptable
(`SPEC.md`, header), so the builder may split, merge or replace these components, as long as every requirement is
still carried and tested. For each: what it delivers, which requirements it carries, the test it must pass (from
`SPEC.md` §6), and what in the repo it can start from. Leads are suggestions from the evidence; the builder may find
better.

### C1. The test bench: the finish line
- **Delivers:** the acceptance suite S0 to S7, with sealed answers, and a scorer.
  - The answers are written by a separate session that never shares context with the builder.
  - The scorer is plain code. Its one model call is the reader test (R10), with a reader model the bench fixes for
    the whole build.
  - What is sealed is a generator. Development draws are used during the build; certification uses one fresh draw,
    scored once; every retry gets a new draw.
  - Planted networks are trained with interchange objectives (as InterpBench's SIIT does), and every sealed
    intermediate variable is checked by intervention before sealing. Networks below a fixed fidelity are dropped.
  - The suite includes:
    - synthetic concepts with no human name (half of S2);
    - networks with two valid accounts (S4);
    - a fluid rehearsal with a sealed shortcut (S6);
    - matched-architecture networks for LD2, t1 and GPT-2 small (S6), without which those real networks are not
      called complete;
    - must-fail cases in two kinds: accounts to reject (S5a) and networks that must get no story (S5b);
    - the pass counts, fixed before the build.
  - The training for S6 (the fluid rehearsal, the planted language model and the matched-architecture networks) is
    budgeted inside this gate.
- **Carries:** every requirement's test.
- **Passes when:** true answers score at the ceiling and every must-fail case scores at the floor. This checks the
  bench itself before anything else is judged by it.
- **Starts from:**
  - `rosetta/yardstick.py`'s training code (S1). Its three rules, in `rosetta/sealed/`, are public, so they are not
    reused;
  - the T2 copies and disguises (S5a), listed with run ids in `scripts/glass/info_measure.py`;
  - InterpBench and Tracr, both public (S0);
  - Leela, GPT-2 and AlgZoo networks with partly known answers (S7).

  S2, S3, S4 and S6 are new.

### C2. The verifier: one place for every test
- **Delivers:** every check a blueprint needs, at any level of a hierarchy:
  - behaviour agreement;
  - interchange interventions;
  - substitution;
  - worst-case completeness;
  - the four-way typing;
  - the copy tests (answering without running; the bounded reader);
  - evidence labels and calibration.
- **Carries:** R3, R4, R5, R9, R12.
- **Passes when:** through the bench's verifier hook (`BENCH_BRIEF.md` §6), it certifies the sealed true blueprints of
  S0, S1 and S4 and rejects every S5a account. The builder sees only the counts.
- **Starts from:** `lab/judge.py`, `lab/lmjudge.py`, Rosetta's substitution (`rosetta/engine.py`), `lab/atlas.py`
  (rule against lookup), and the X0 design (`docs/review/PREREG_FIRST_EXPERIMENTS.md`).
- **If it fails:** stop. The measurement is wrong, as T2 was, and nothing built on it would count.

### C3. The map-maker: the network's own parts, and their wiring
- **Delivers:** the network's variables at every layer, plus the wiring between them. It works backward from the
  behaviour: which variables carry it at the last layer, which variables feed those, and so on down to the inputs.
- **Carries:** R6 (the structure), R7.
- **Passes when:** on S0 and S3 it locates every sealed intermediate variable and the sealed wiring.
- **Starts from:** `rosetta/sae.py` (sparse dictionaries), `lab/spd.py` (exact decomposition), and Rosetta's value
  attribution and cross-block concentration (`rosetta/glass2.py`, `rosetta/fresh_where_site.py`).
- **Lead:** attribution graphs over a sparse replacement (Anthropic's circuit tracing) do this per prompt for language
  models. The same backward reading of which parts feed a verdict applies to any network.

### C4. The definer: each part in terms of what feeds it
- **Delivers:** a definition for each variable in terms of the variables upstream of it, or the inputs, working from
  the bottom of the map upward.
  - The definitions are written in a restricted language: no weights, no fitted sub-models, a budget of constants.
  - The language contains operations as well as features: for example, "play out", "for each reply", "the exchange
    on a square", "a spatial derivative", "at the previous token".
- **Carries:** R6, R3.
- **Passes when:** on S1 and S3, every sealed variable, including the intermediate ones, is defined at the recovery
  line: correlation and substitution on held-out inputs, each at the sealed definition's own score minus its spread
  (`SPEC.md` §6).
- **Starts from:** `rosetta/core.py` and `rosetta/engine.py` (exhaustive search over restricted programs, control
  lines, substitution), and `rosetta/chess_lib.py`.
- **Why this should repair the break:** Rosetta could not define the verdict variables from the board (2 of 16).
  The working hypothesis, which Gate 2 tests, is that a verdict defined from its ingredients is a short combination
  of defined parts. Rosetta's evidence is consistent with this but does not prove it: the cross-block concentration
  figures are exploratory.

### C5. The inventor: new kinds of concept when the vocabulary runs out
- **Delivers:** when C4 cannot define a variable at the line, C5 adds to the language:
  - new primitives or new operations, each with a name and a definition, and with its provenance recorded;
  - every proposal is scored only by C2, never accepted on a model's say-so.
- **Carries:** R8.
- **Passes when:** on S2 it recovers the sealed concepts, none of which can be written in the starting vocabulary. The
  starting vocabulary is published before S2 is written.
- **Starts from:** `rosetta/llm.py` (a proposer), `rosetta/line.py` (the one hand-added kind of quantity that worked),
  and `kepler/targets/fluid/library.py` (physics terms).
- **Leads, in order of evidence:**
  - **Library learning:** compress the definitions already found into reusable abstractions, as DreamCoder grows its
    own language from solved problems.
  - **Search over operations, not only features:** the planned-line material was one such composition. It carried a
    sixth to a third of LD2's extra knowledge, though no single variable holds it. It is a pointer to the missing
    kind of ingredient, not a passing concept.
  - **An observer that can perceive the domain:** a strong model, shown boards rendered as images or contrast pairs
    (positions where the variable is high against low, with material matched), names what differs. Its proposal is
    then written in the language and scored. MAIA is the precedent for an agent that runs such experiments.
  - **Directions found without labels, kept only if teachable:** Schut et al. found new concepts inside AlphaZero
    this way, filtered them by whether a weaker network could learn them, and taught them to grandmasters.

### C6. The assembler: the blueprint
- **Delivers:**
  - the systems and the one-page overview, assembled from defined parts;
  - the account of what the network does, with the knowledge split (R2);
  - the ledger, with every part's type;
  - per-input traces;
  - rival accounts;
  - stated confidences on every claim;
  - where a reference method exists, the cost map: which parts stand in for which of its steps, with cost and error
    (R15).
- **Carries:** R2, R4, R5 (assigning the types, which C2 tests), R6, R7, R12, R15.
- **Passes when:**
  - on S0 to S4, blueprints run on their own and are complete within the line, input by input, on every output;
  - on S1 to S4, the behaviour account (R2), the types (R5) and calibration (R12) pass;
  - no S5b network gets a story;
  - on S3, the overview matches the sealed top level;
  - on bench-chosen decisions, traces' predicted flips beat attribution patching and "no change";
  - on S6's fluid rehearsal, the cost map names the sealed shortcut, its parts, and its cost and error (R15).
- **Starts from:** Rosetta's glass model (`rosetta/glass2.py`), the residual curve (`rosetta/residual_curve.py`), and
  the atlas's certificate.

### C7. The reader interface: blueprints for minds
- **Delivers:** the blueprint rendered for a reader, with plain-language definitions, examples and contrast pairs. The
  scorer runs the reader test on that rendering, with the bench's fixed reader model. The human study runs after the
  build, on findings.
- **Carries:** R10.
- **Passes when:**
  - the bounded-reader test passes on S1 to S4 and on S7's known answers;
  - for any new chess concept, a small human study follows the Schut et al. design.
- **Starts from:** `rosetta/teach_dict.py`. It needs a reader that can perceive boards; with the cheap model,
  teachability stayed essentially untested.

### C8. The editor: control through the blueprint
- **Delivers:** edits made through blueprint parts (switch off, strengthen, redirect), each with its effect and side
  effects predicted before it is run.
- **Carries:** R11.
- **Passes when:** on S3 and S6, on at least 20 bench-chosen edits per domain, its predictions beat attribution
  patching and "no change". The edits include "remove a capability" edits, such as switching off look-ahead in the
  chess rehearsal and switching off the planted false-answer gate.
- **Starts from:** the liar sprint's steering code (`scripts/liar/`), and `lab/steer_calendar.py`.

### C9. Adapters: every domain through one core
- **Delivers:** adapters for each Tier A domain:
  - chess (exists, `kepler/targets/chess/`);
  - small algorithmic networks (exists, `lab/`);
  - the rebuilt fluid emulator;
  - GPT-2.

  Plus a 7–8B adapter with a sparse replacement for Tier B.
- **Carries:** R1, R13.
- **Passes when:** S6's rehearsals, including the matched-architecture networks, and S7's real networks run through
  the same core.
- **Starts from:**
  - `kepler/targets/chess/leela.py` and `leela_t.py`;
  - `lab/lm.py` (GPT-2);
  - `kepler/targets/fluid/` (to be retrained on coarse-grained frames of a finer simulation, the known-answer setting
    that Kochkov et al. 2021 showed works).

### C10. The director: no hands
- **Delivers:** the loop run end to end, with every experiment logged and a stop when the blueprint is complete or the
  budget is spent:
  - pick the largest unresolved share;
  - map it, define it, invent where needed;
  - verify;
  - update the ledger.
- **Carries:** R14.
- **Passes when:** the full suite runs with no human input after the start.
- **Starts from:** `probe/` (the Helix Probe loop: predict first, then experiment), and `helix_evolve` (a separate repo,
  `cybercryptixcoder/helix_evolve`) for volume search where a score is closed to loopholes.
- **Note:** a strong model takes the judgment roles (director, observer, critic). A cheap model at volume fits only
  where the scorer is closed (`docs/review/NEXT_BUILD.md` §3.11).

---

## 4. The order, and why it makes this one attempt rather than twenty

The build runs as six gates. Each gate has a pass test from the suite, and a failure at any gate names the component
that failed, the way an engineer names the member that gave way.

| Gate | Components | Pass test | Why here |
|---|---|---|---|
| 1. The finish line exists | C1, C2 | The bench scores true answers at the ceiling and must-fail cases at the floor; through the bench's verifier hook, the builder's checker certifies the sealed true blueprints of S0, S1 and S4 and rejects every S5a account | Nothing can be judged without it, and the record shows measurements must be proven before use (T2) |
| 2. **The make-or-break** | C3, C4, C5 | S2 (concepts outside the vocabulary) meets its pass counts; on S3 (verdicts through their ingredients), every sealed intermediate variable is recovered and the sealed wiring found; S0's circuits are located and S1's rules defined. The exact counting rule is in `BUILD.md` §6 | This is where the last build broke, and it is the only research-grade part. Testing it second keeps the uncertain spend bounded and early |
| 3. Complete blueprints | C6 | S0 to S4 run on their own and are complete within the line; on S1 to S4, R2, R5 and R12 pass; no S5b network gets a story; S6's rehearsals pass their R3 to R7 parts, and the fluid rehearsal passes R15 | Assembling defined parts into levels, with a ledger. The full S6 comes at Gate 6, once readers, editors and adapters exist |
| 4. Real known answers | C9 with C2 to C6 | S7: look-ahead along forcing lines and the capture variable in LD2 and t1; month rollover; modular multiplication; M4,3 | Proves the box on real weights before any unknown target |
| 5. Minds and hands | C7, C8 | Reader and editor tests | Usability and control on blueprints that already pass |
| 6. No hands | C10 | The full suite, including the full S6, on a fresh development draw, end to end from one start configuration, with no human input | The box, not an agent working by hand |

**Then certification:** one fresh draw, generated by a separate session from Shreyas's seed and scored once
(`BENCH_BRIEF.md` §8). Gates are passed on development draws; only certification says the system is done
(`SPEC.md` §7).

**Then the flagships:**
- chess t1 and LD2;
- the rebuilt fluid emulator;
- the lying mechanism in a 7–8B model (Tier B);
- how a model reads weights (Tier B).

**What happens if a gate fails:**
- **Gate 1 fails:** the measurement is wrong. Fix the intent, not the code (`CLAUDE.md`), before anything else.
- **Gate 2 fails:** it fails at a named place. Either:
  - the map-maker did not find the ingredients;
  - the definer could not define them even with operations;
  - the inventor could not add the missing kind.

  Each points at different work, and none needs the rest of the build redone.
- **Gates 3 to 6:** these rest on methods with precedent in the field (references in `SPEC.md` §10), but several
  have not been shown to this standard. A failure is diagnosed by the project's routine: a translation error (the
  code did not match the intent) is fixed in that component; an intent error is recorded in the intent, with its
  evidence (`CLAUDE.md`).

**The spec and the suite are frozen during the build.** A change is a new, named version with its evidence.

---

## 5. What must be in place before starting

1. **A strong model that can perceive the domain,** for the observer, the reader, the director and the critic.
   - The only key in this workspace is DeepSeek V4.1 Flash. It could barely apply board rules (essentially untested
     teachability), and its proposals added nothing (`docs/rosetta/RESULTS.md` §8, `FINDINGS.md`).
   - This is the single most important missing resource. Without it, C5 is limited to library learning and search,
     and C7 and C10 cannot pass.
2. **GPU time**, for:
   - decomposition at chess and language-model scale;
   - the fluid rebuild;
   - the planted language-model rehearsal;
   - 7–8B models (Tier B).
3. **Shreyas's approval for the spend,** with a budget per gate, so that Gate 2's cost is known before it starts.
4. **A separate session to write the sealed answers** (C1), so that the builder never sees them.
5. **Time measured in days, not hours.** Rosetta's three hours built one chain for one network. This is ten components
   and a sealed suite. Gates 1 and 2 come first, so the time at risk is spent where the uncertainty is.

---

## 6. What passing guarantees, and what it does not

**If Gates 1 to 6 pass, and the certification draw then passes, for any Tier A network and behaviour in scope** the box
returns a blueprint that is:
- complete;
- honestly typed;
- runnable on its own;
- faithful at every level;
- navigable from a one-page overview down to components;
- usable by a reader;
- calibrated about its own confidence.

Or it states exactly where, and how much, is unresolved, with no silent gaps. That is "fully interpretable" as
`SPEC.md` §2 defines it, and it is testable by anyone.

**It does not guarantee:**
- **That a network is simple.** A mostly memorised network gets a complete but table-heavy blueprint.
- **That a network contains new knowledge.** The box finds it if it is there.
- **Completeness at Tier B within budget.** The field's best tools still leave substantial unexplained error at that
  scale; the box reports where completeness stops.

**The remaining risk after passing:**
- Real networks could hold structure unlike anything in the suite.
- S7 (real networks with known answers) and S6 (rehearsals shaped like each flagship) exist to shrink that risk
  before the unknown targets are run. They cannot remove it.

---

## 7. Decisions for Shreyas

Defaults for every number are fixed in `SPEC.md` §6, and `BUILD.md` says what the builder does about each missing
resource. So "Build" can start without answers to the questions below. The defaults hold unless Shreyas changes them
before the bench is sealed.


1. **The definition.** Is `SPEC.md` §2 what you mean by "fully interpretable"? In particular, do you accept:
   - "navigable, not necessarily short";
   - the four honest types (rule, lookup, irrelevant, unresolved)?

   Everything downstream depends on this.
2. **The scope.** Is Tier A the right narrow realm, with Tier B run and reported but not required to be complete?
3. **The resources.** A frontier-model key, a GPU budget, and a budget per gate.
4. **The intent.** By default, `SPEC.md` is this build's intent: its header says so, and `CLAUDE.md` item 0 points to
   it. `docs/glass/INTENT.md` is left unchanged as the record of the earlier target. Replacing it is your call.
5. **Who writes the sealed suite.** A separate session, so that the builder never sees the answers. By default you
   start it with `BENCH_BRIEF.md` as its whole brief; or you tell the builder to launch it as a fresh agent
   (`BUILD.md` §5).
6. **A private repository** for the sealed material (default name `glass_nets_bench_sealed`), and, at the end, the
   certification seed.

---

## 8. Changes made after the independent critique

A fresh agent attacked the first drafts of `SPEC.md` and this file (`docs/spec/SPEC_CRITIQUE.md`). Every point was
checked against its source. All held, and each led to a change:

| # | The critique's point | The change |
|---|---|---|
| 1 | The blueprint was never required to run on its own; Rosetta broke exactly there | Added: the blueprint run alone reproduces outputs and intermediate states, level by level (SPEC §2 condition 3, R4) |
| 2 | The sealed suite was both the development set and the exam | The bench seals a generator: development draws, one fresh certification draw, new draws for retries (SPEC §6, C1) |
| 3 | Planted networks may not compute their sealed mechanism; "largest unresolved share" was lenient; toy-to-real transfer | Planted networks trained with interchange objectives and checked before sealing; percentile lines; "complete" for a real network only against a matched-size rehearsal (SPEC §6, R4) |
| 4 | The usability line and the fixed 0.9 recovery line lowered the bar | Usability set by the true blueprint's reader score; recovery needs correlation and substitution at the sealed definition's own level; the copy condition demoted to a sanity check; readers navigate long blueprints on demand (R10, §6) |
| 5 | Typing could be gamed by filing hard parts as tables | Lookup keys must be verified training items where known; any part with an effect on fresh inputs cannot be lookup; irrelevance line from planted unused parts; 20 cases per type; lookup and unresolved shares in every headline (R5, §2) |
| 6 | Completeness underspecified; cancellation is across inputs | The bench fixes the resampling sets (respecting and adversarial); input-by-input recovery on every output, with its lower tail reported (R4) |
| 7 | Two flagships uncovered (efficiency; a generalising lie detector); Tier B partial | Added R15 (where efficiency comes from), behaviour fingerprints tested on uninstructed cases (R7); §1 says plainly which flagships get partial accounts |
| 8 | GAP overpromised | §1 rewritten: the system exists only if Gates 2 to 6 pass; pass counts fixed in advance; no failures pre-labelled as translation errors |
| 9 | Gate 3 needed components from later gates | Gate 3 runs S6's R3 to R7 parts; the full S6 moves to Gate 6; S6 training budgeted in Gate 1 |
| 10 | Smaller holes | Synthetic S2 concepts with no human name; the bench chooses flips and edits (20 or more per domain); planted two-account cases for R12; family-wise control lines (R3); adapters capped and one start configuration (R1, R14) |
| 11 | Factual errors | Fixed: the 50% match belongs to the 18-layer model, not Haiku; the 22% is variance left by one surrogate; two reader tries, not one; the line swing is not a located variable; the cross-block figures are exploratory; the capture variable passed at the margin; R5 is partly built |

