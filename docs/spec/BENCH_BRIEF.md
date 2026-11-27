# Brief for the bench-writer: the sealed acceptance suite

*For a separate session, not the builder. Written Nov 26, 2026, from 19:30 ET (clock). You write the finish line of
the build: the sealed acceptance suite of `SPEC.md` §6, its scorer, and the certification procedure. You never share
your context with the builder. Your final report to whoever started you describes the interface only, never sealed
content.*

---

## 1. What you produce

1. **A generator** that, given a seed and a list of tiers, produces one draw of those tiers (by default every tier,
   S0 to S7), together with its sealed true answers.
2. **Development draws** for the builder to use while building.
   - At least three full draws. The lines' spreads and control lines are computed from these and then frozen (§5).
   - Plus fresh draws on request: every formal gate attempt needs a draw that no earlier formal attempt used.
   - A draw for a formal attempt holds only the tiers its gate scores.
   - S6 is drawn afresh only for Gate 3, Gate 6 and certification. Within one gate's retries its networks may be
     reused, and the scorer then prints only pass or fail for them.
3. **A scorer** that takes the box's output for a network and prints scores and pass/fail per requirement. It never
   prints sealed content.
   - **On development draws** it prints, per network and per requirement: pass or fail, the box's score and the line.
     For Gate 2's diagnostics, it also prints how many sealed intermediate variables were recovered out of how many,
     and which of the box's own parts (by the box's own identifiers) matched one.
   - **On the certification draw** it prints only pass or fail, per requirement and per tier.
   - It never prints a sealed definition, a family or a mechanism.
4. **A certification procedure**, run once at the end by a separate session: a fresh draw from a seed Shreyas gives,
   including concept families never used in any development draw, scored once.
5. **`bench/accept/README.md`** (how to call the scorer) and **`bench/accept/INTERFACE.md`** (what the box must hand the scorer;
   §6 below).

**Where things go, and how the two sessions connect.** The repo is public, so nothing sealed is committed before
certification.
- **Shared branch.** Both sessions work against the build branch on origin: `rosetta-v2`, unless Shreyas names
  another. If it does not exist yet, create it from `rosetta` and push it.
- **Order of pushes:**
  1. You push a draft `bench/accept/INTERFACE.md` (everything in §6 except the reference true blueprint) and
     `bench/accept/README.md`, as early as you can.
  2. The builder pushes `bench/accept/VOCABULARY.md` and its code in `bench/accept/vocab/`, and tags that commit
     `vocab-frozen`. The tag is the freeze. The vocabulary fixes how definitions are encoded, and `INTERFACE.md`
     adopts that encoding.
  3. You complete `INTERFACE.md` with its reference true blueprint, written in that grammar, then write S1 and S2.

  Before the tag, you can build S0, S7, the scorer, and the parts of S5 that do not depend on S1 to S4 (the saved
  copies and disguises, and the networks with no structure).
- **Two private repositories,** which Shreyas creates (ask him if they do not exist yet):
  - `glass_nets_bench_sealed` holds the generator, the held-out concept families, the certification procedure, the
    scorer of record, and your diary and ledger. The builder never gets access to it.
  - `glass_nets_bench_dev` holds the sealed answers of the development draws, and nothing else.
- **In the builder's workspace,** the scorer reads the development answers from `bench/accept/sealed/`, which is
  gitignored.
  - The builder clones or pulls `glass_nets_bench_dev` there when a draw is ready, and never opens it. This is an
    honour rule.
  - It is backed by the design: the certification draw is generated only at the end, from a seed the builder never
    saw, and it includes held-out concept families.
  - Shreyas attaches only `glass_nets` and `glass_nets_bench_dev` to the builder's session.
- **Draws.**
  - The builder asks for one with a line "Draw request: <gate>, <tiers>" in `docs/build/STATUS.md`, and tells
    Shreyas.
  - When a draw is ready, you push its answers to `glass_nets_bench_dev` and its networks to the build branch, then
    add the line "Draw <id> ready: <tiers>" to `docs/build/STATUS.md`.
  - If you have ended, a session Shreyas starts with this brief serves the request.
- **If the builder launched you** and you share its workspace:
  - write the development answers straight into `bench/accept/sealed/`;
  - keep the generator and the held-out concept families outside the repository tree;
  - push them only to `glass_nets_bench_sealed`.
- **If the private repositories do not exist yet,** keep the sealed material outside the repository tree, tell
  Shreyas, and never commit it to the public repo.
- **Committed to the build branch:** the development networks themselves (their weights are what the box must
  explain), a development copy of the scorer, and the interface files.
  - Weights over 50 MB go as release assets of the public repo (tag `bench-dev-<draw id>`), with checksums in
    `bench/accept/README.md`. GitHub refuses files over 100 MB.
- **The scorer of record** is the copy in `glass_nets_bench_sealed`.
  - Fix a scorer bug in both copies, as a new, named version, logged in `bench/accept/CHANGES.md`.
  - Certification scores with the scorer of record.
- **Vendored code.** The scorer keeps its own frozen copies, under `bench/accept/vendor/`, of anything it needs from
  `lab/`, `kepler/` or `rosetta/`, so the builder's edits cannot move the measurement.
- **After certification,** everything is committed, for transparency.

---

## 2. Before you start

- **Read `SPEC.md`:** §2 (the definition), §4 (every requirement's test), §5 (the blueprint's contents) and §6 (the
  suite, the lines, the defaults).
- **Wait for the builder's `bench/accept/VOCABULARY.md`,** its frozen starting vocabulary, before writing S1 (written in
  it) or S2 (written to be impossible in it). Until it exists, build S0, S7, the scorer, and the parts of S5 that do not depend on S1 to S4.
- **Reuse what exists:**
  - Rosetta's yardstick training code (`rosetta/yardstick.py`) for S1, but not its three rules:
    `rosetta/sealed/yardstick_rules.py` is public. Rosetta's write-ups call those rules S1 to S3; they are not this
    suite's tiers.
  - the saved copies and disguises for S5a:
    - the round 1 and 2 constructions, in `lab/constructions/<target>/<run>/construction.py`;
    - the disguises D1 to D6 generated from them.

    Both are listed, with run ids, in `scripts/glass/info_measure.py` (`CASES`), and their results are in
    `docs/glass/info_measure_T2.txt`. Its rows marked "fail" are the starting S5a list. Its rows marked "pass" are
    real accounts the checker must certify;
  - AlgZoo networks, GPT-2 and the Leela networks for S7;
  - Tracr and InterpBench, both public, for S0. Their code is on GitHub. If downloading pretrained models is
    blocked, retrain from their code.
- **Data:** the chess networks, games and puzzles, and where they go, are in `BUILD.md` §4, step 3. Use the same
  files as the builder.
- **Your records.** `CLAUDE.md`'s rules bind you too.
  - Your diary and change ledger live in `glass_nets_bench_sealed`, so that nothing sealed is published.
  - On the public branch, log only changes to the interface and to the scorer code, in `bench/accept/CHANGES.md`.
- **Times:** from the clock only (`TZ=America/New_York date`, in its own call).
- **Commits:** under Shreyas's identity (cybercryptixcoder, official.shreyas28@gmail.com), with no AI co-author or
  session lines.
- **Spending:** ask Shreyas before any. You ask for the bench's own GPU time; the builder asks for its own.
  - The S6 rehearsals (a fluid emulator, a small language model, the matched-architecture networks) and the SIIT
    training may need it.
- **Run X0's main arm first** (`docs/review/PREREG_FIRST_EXPERIMENTS.md`, X0), on the saved runs. Report whether
  answering without running separates the real accounts from the copies.
  - X0's estimators must be written by a session that has only the accounts and the tasks. You hold the networks,
    so start a fresh agent for that step and give it only those.
  - If X0 does not separate them, R9's estimator test is reported but does not gate. R9 then gates on the must-fail
    set and the bounded reader.

---

## 3. What each tier contains (sizes are the defaults of `SPEC.md` §6)

**S0. Known circuits in realistic weights.**
- 10 Tracr-compiled programs and 10 InterpBench (SIIT-trained) models.
- The true answer is the published circuit.

**S1. Planted rules inside the starting vocabulary.**
- 10 networks, each trained on a sealed rule written in `bench/accept/VOCABULARY.md`'s terms.
- Rosetta's yardstick is the pattern.

**S2. Planted concepts outside the starting vocabulary.**
- 10 concepts per draw:
  - **5 named:** ideas a domain expert would recognise, such as a chess pattern. A strong model may know these.
  - **5 synthetic:** combinations or operations with no human name, built so that no expert or model could recall
    them.
- Every concept must be provably impossible to write in the starting vocabulary within its length budget: no
  expression of length at most L, within the constants budget, reaches the recovery line. Check this by exhaustive
  search over the vocabulary (fitting each expression's constants), and record the check.
- Draw concepts from families.
  - A certification draw includes at least 3 families never used in any development draw.
  - At least 5 of its 10 concepts come from those families, with named and synthetic both represented.
- Include both kinds of concept:
  - some reachable by composing named expressions beyond length L;
  - some that need a new operation (§6).

  Record which kind each is, sealed.

**S3. Planted staged computations.**
- 5 networks per draw, each computing a verdict in stages from intermediate concepts. At least one stage is
  search-like: playing moves forward, iterating, or taking a minimum over options.
- One of the five is the chess rehearsal: a small value network trained to imitate a sealed evaluation that looks two
  plies ahead along forcing moves (checks and captures) and adds sealed positional terms.

**S4. Planted mixtures, and rival accounts.**
- 5 networks per draw mixing a rule, a memorised random table (keyed by listed training items) and unused parts.
- Plus 2 networks with two valid accounts: the input distribution is built so that two different definitions agree
  on it, and both are sealed.

**S5. Must-fail, in two kinds.**
- **S5a, accounts the checker must reject:**
  - every saved copy and disguise from rounds 1 and 2;
  - for each S1 to S4 network, a distilled small student offered as an account, and a program carrying a literal
    table tuned by selection.

  Write each S5a account in `INTERFACE.md` form, as an adversary would.
  - Make it grammatical and within budget wherever possible: for example, the weights spread over many small parts,
    or stored as lookups keyed on training items.
  - An account that fails only the format check does not count as rejected.
  - Record, sealed, which test rejects each one.
- **S5b, networks the box must not explain with a story:**
  - for each S1 to S4 network, a shuffled-weight version, an untrained version and a memoriser (shuffled labels);
  - 3 networks with no structure.

  Passing means the box types their parts as lookup, irrelevant or unresolved, with no rule above the family-wise
  control line.

**S6. One rehearsal per flagship, each sealed:**
- **chess:** the S3 chess network;
- **fluid:** an emulator trained on coarse-grained frames of a finer simulation, with a sealed closure term and a
  sealed shortcut (a longer time step, with a sealed correction), so R15 can be scored;
- **lying:** a small language model with a planted gate that switches it to false answers when a sealed context
  feature is present. Include held-out cases where the feature appears without being instructed;
- **weight reading:** a small reader network trained to report a property of another network's weights by a sealed
  method;
- **matched architectures:** a planted network with LD2's architecture, and one with t1's, each trained to a sealed
  staged evaluation; and one with GPT-2 small's architecture, trained to a sealed behaviour.
  - These are needed before "complete" can be claimed for LD2, t1 or GPT-2 (`SPEC.md` R4).
  - They need GPU time.
  - If they cannot be built, say so. They are then left out of S6 for every gate and for certification. The real
    networks they match get their completeness reported with an interval, and are not called complete.

**S7. Real networks with partly known answers.**
- LD2 and t1: look-ahead along forcing lines, as confirmed on fresh positions (`docs/rosetta/PREREGISTRATION.md`, R2),
  and the capture variable (`docs/rosetta/FINDINGS.md` 3).
- GPT-2 small: month rollover (`lab/constructions/gpt2_month_rollover/`).
- Modular multiplication (`lab/constructions/modmul_p113/20261113-190435-ec84/`).
- AlgZoo M4,3 (`lab/constructions/zoo_2nd_argmax_4_3/20261118-015037-c928/`, and `docs/glass/M43_VS_ARC.md`).
- The known answers are written as partial true blueprints. The box must reproduce them, not be limited to them.
  - Two chess answers are defined through the network's own move. Write them as §6's rule for the network's own
    choice allows: through a part that computes the move.
  - A known answer is reproduced when one part of the box's blueprint, or one composite part it defines, passes the
    recovery tests against the known definition.
  - Its line is the known definition's own score on each draw's fresh held-out inputs, minus its spread across the
    first three development draws.
  - On S7, reproduction gates (Gate 4, and done), and R10 gates at Gate 5. The other requirements are reported.

**Bench-chosen probes, sealed per draw:**
- at least 20 decisions per domain, for R7's flips;
- at least 20 edits per domain, for R11, including "remove a capability" edits;
- a set of unusual inputs per domain, for R2's failure predictions;
- the resampling sets for R4: some that respect the true blueprint, and some adversarial.

---

## 4. Admission checks before anything is sealed

A planted network must actually compute its sealed mechanism. Training on outputs alone does not guarantee that.
- **Train with interchange objectives,** as InterpBench's strict interchange-intervention training (SIIT) does.
- **Admit a network only if both hold:**
  - its outputs match the sealed mechanism on at least 98% of held-out inputs (R² at least 0.95 for continuous
    outputs);
  - every sealed intermediate variable passes interchange interventions at its sealed site on at least 90% of
    held-out pairs;
  - for a lookup variable, both checks use its memorised items instead: the outputs on them, and interchange on pairs
    of them;
  - once the lines are frozen, its own true blueprint clears every one of them (§5).
- **Otherwise,** drop it, draw a replacement, and record the drop rate.
- **Report** the admission statistics, but not the mechanisms.

---

## 5. True blueprints and the lines

For every planted network, write its **true blueprint**: the sealed mechanism expressed in the form of
`bench/accept/INTERFACE.md` (§6). If it needs an operation the frozen vocabulary lacks, write it as a new operation
under §6's rules and record which. Do not ask for a change to the vocabulary.

The lines of `SPEC.md` §6 compare the box with the true answer on the same network. Each is the true answer's own
score on the draw being scored, moved by the tier's spread:
- **Spread** is the standard deviation, across the first three development draws, of the tier-level statistic (for
  recovery, the mean over the tier's sealed definitions of their own scores). The spreads, and the control lines
  below, are computed from those three draws and then frozen.
- **Recovery line:** each sealed definition's own correlation and substitution scores, on held-out inputs, minus the
  tier's spread. It is computed on the draw being scored, so it exists for held-out families too.
- **Completeness line,** per network: the 5th percentile, over that network's inputs, of its own true blueprint's
  recovery, minus the tier's spread.
- **Irrelevance line,** per network: the 95th percentile of the behaviour change caused by resampling that network's
  sealed unused parts, plus the tier's spread.
- **Family-wise control lines:** the 99th percentile of the best score among as many control variables (untrained
  networks, shuffled activity, random directions) as real ones tested.
- **Usability line,** per network: its true blueprint's score in the bounded-reader test, minus the tier's spread.
  - It uses the scorer's own fixed, plain rendering of the true blueprint.
  - It needs the frontier-model reader. If no key is available yet, mark it "pending", and compute it before Gate 5.

A planted network whose own true blueprint does not clear every line, the control lines included, is dropped and
redrawn (§4).

The lines live inside the scorer. On development draws the builder sees each score next to its line and "pass" or
"fail", never the sealed definitions. On the certification draw it sees only "pass" or "fail".

---

## 6. The interface: what the box must hand the scorer

The scorer runs every test itself, on the networks, with its own intervention code. Write `bench/accept/INTERFACE.md` as
an **executable contract**, so that a builder and a bench-writer working apart produce compatible work. It needs three
things:
- a Python protocol with typed signatures;
- the task-folder format;
- one reference true blueprint, for a public S0 model, that the scorer passes.

It must specify:
- **The task folder** the bench hands the box for each network:
  - the weights, and a loader that runs the network;
  - the behaviour: which outputs, and which metric;
  - the inputs;
  - the training items, where known (R5's lookup keys need them);
  - the named intervention sites.
- **The allowed map classes** for a part's location (R3). Each is linear in the activations or the weights, and is
  declared in the blueprint:
  - a direction, or a subspace of declared dimension, in an activation;
  - a feature of a declared sparse dictionary, with its encoder and decoder supplied;
  - a parameter component of a declared decomposition.

  Anything else is rejected, because unrestricted maps make any network "implement" any algorithm.
- **Parts.** Each part gives:
  - an identifier;
  - its location (layer, position, and the map);
  - its type (rule, lookup, irrelevant, unresolved);
  - for a rule, its definition;
  - for a lookup, its table;
  - the identifiers of its upstream parts;
  - for each claim, its evidence label.
- **Definitions** are expressions in the grammar of `bench/accept/VOCABULARY.md`, in the encoding it fixes, not
  arbitrary code, so the scorer can check each against the language and its constants budget.
- **Invented primitives** (R8). The box may add primitives of two kinds:
  - a named expression in the grammar, which later definitions may use as one symbol;
  - a new operation: Python code that does not call the network, read its weights or activations, or do any input
    or output, and whose size is at most a fixed number of AST nodes. Fix that number here before S2 is written
    (default 200).

  Length counts each invented primitive as one symbol, so every definition stays within L. Constants are counted
  after expansion: a definition, with every primitive it uses expanded, holds at most 5 numeric constants. So
  invention adds structure, never fitted numbers. The scorer records each primitive's provenance.
- **The network's own choice.** Some variables are about the network's own choice on the same input: the square its
  chosen move goes to, or its planned line.
  - A definition may refer to such a choice only through a part of the blueprint that computes it, such as the
    blueprint's own move prediction. It never calls the network.
  - When the scorer tests such a part on the network (R3), it uses the network's actual choice. When the blueprint
    runs alone (R4), it uses the blueprint's own.
- **Run alone.** Given inputs, the blueprint produces outputs and every part's value without the network.
  - The scorer calls this entry point in a process with no access to the network's weights.
  - It resamples unresolved parts from other inputs, by its own choice, and feeds lookups their tables.
- **Behaviour account (R2).**
  - The scorer gives the bench's unusual inputs only to the run-alone entry point, before the network is run on
    them. The box's predicted outputs there are its predictions of where the network fails.
  - For each departure from the reference rule that the box reports, it names the parts responsible.
  - The scorer computes agreement, the worst slice and the knowledge split against ground truth.
- **Budget (R13).** The box states its compute budget in its start configuration, as a function of the network's size
  (GPU-hours and model-call tokens). The scorer records what each run used. R13 passes on a network when the run
  stays within its stated budget.
- **Estimator.** For answering without running (R9): given a question, such as a quantity over a distribution, the
  box returns an estimate and its operation count.
- **Claims** for calibration (R12): each part's type, and each predicted test outcome (a test passing, a flip, an
  edit's effect), each with a stated probability.
- **Overview and systems.** The one-page overview, with a callable top-level predictor, and groups of parts with their
  roles.
- **Trace and flips.**
  - The bench poses decisions; for each, the box returns each part's contribution and a predicted minimal flip.
  - Flip size is the number of changed input elements (squares, tokens) for discrete inputs, and L2 distance for
    continuous ones.
  - For a continuous output, the bench defines the decision per network, as a sign or a threshold.
  - The bench's own search checks minimality among its candidates.
- **Edits.** The bench poses edits in terms of capabilities ("switch off look-ahead", "remove the gate"). The box
  maps each to its parts and returns the edit and its predicted effects. The scorer applies the edit to the network
  and measures.
- **Fingerprint.** For a named behaviour, a detector the scorer can run on activations.
- **Ledger.** Shares by type; the unresolved share by layer.
- **Reader rendering.** A navigable rendering that a bounded reader can open part by part.
- **Rivals.** Alternative blueprints, if any.
- **Cost map (R15).** For networks with a reference method: which parts stand in for which steps, with cost and
  error.
- **Entry point.** The box's entry point and its single start configuration. Both are used unchanged for
  certification.
- **Verifier hook (Gate 1).** A named entry point to the builder's checker. The scorer feeds it the sealed true
  blueprints and the S5a accounts. It prints only counts, per tier: "certified X of N true blueprints, rejected Y of
  M accounts".

---

## 7. Defaults: the bounded reader (R10), calibration (R12), and other numbers

- **The reader** may not run code.
- **The reader model** is fixed by the bench for the whole build: by default, the frontier model whose key Shreyas
  supplies, at a fixed version recorded in `bench/accept/README.md`.
  - The scorer runs the reader test. The builder supplies only the rendering.
  - This is the only model the scorer calls.
- **Its budget** in output tokens per question is at most 10% of the multiply-adds in one forward pass of the
  smallest opaque program compared (a copy, or a distilled student). That is too few to hand-execute it.
- **Where that smallest program needs fewer than 10,000 multiply-adds,** the reader test is reported but does not
  gate, because hand-execution is cheap there.
- **Calibration** passes when the expected calibration error of the box's stated confidences is at most 0.10,
  computed over 10 equal-width bins of stated probability, on each draw it is scored on. The certification draw is
  the one that decides "done".
- **The overview match** (R6) is the top-level predictor's agreement with the sealed top level, at the recovery
  line.
- **R5's "20 cases per type"** applies to S4, over the three types a planted network can contain (rule, lookup,
  irrelevant). It counts parts, pooled across S4's networks in one draw. Design S4 so that each draw holds at least
  20 of each.
  - On S1 to S3, R5 is scored on the parts present.
  - "Unresolved" has no planted instances. It is scored through the ledger (R4) and on S5b.

---

## 8. Certification

1. Shreyas gives a seed.
2. A separate session (you, or a fresh one with this brief and the private generator) draws a fresh suite. The draw
   includes the held-out concept families.
3. It runs the finished box from the commit the builder tagged `cert-candidate`, from its one start configuration,
   with no human input, and scores it with the scorer of record from `glass_nets_bench_sealed`.
4. It scores the draw once, against the pass counts in `SPEC.md` §6.
5. It reports to Shreyas and the builder together.

- The box may start certification with the concept library it grew during development. The library must be in the
  `cert-candidate` commit.
- S2's held-out families are scored separately, and the 80% pass count applies to them on their own.
- There are no retries on a certification draw. A retry needs a new seed, and Shreyas's decision.

---

## 9. What you report back

Report only these:
- the interface (`bench/accept/README.md`, `bench/accept/INTERFACE.md`);
- the tiers built and their sizes;
- the admission statistics;
- the bench's self-check, per tier: true answers at the ceiling, and must-fail cases at the floor (the first half of
  Gate 1);
- X0's outcome, and so whether R9's estimator test gates;
- which lines are computed and which are pending;
- where the private repositories are.

Never report concept families, definitions or mechanisms.
