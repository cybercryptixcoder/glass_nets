# Brief for the bench-writer: the sealed acceptance suite

*For a separate session, not the builder. Written Nov 26, 2026, from 19:30 ET (clock). You write the finish line of
the build: the sealed acceptance suite of `SPEC.md` §6, its scorer, and the certification procedure. You never share
your context with the builder. Your final report to whoever started you describes the interface only, never sealed
content.*

---

## 1. What you produce

1. **A generator** that, given a seed, produces one draw of every tier, S0 to S7, together with its sealed true
   answers.
2. **Development draws** (at least three) for the builder to use while building.
3. **A scorer** that takes the box's output for a network and prints scores and pass/fail per requirement. It never
   prints sealed content.
4. **A certification procedure**, run once at the end by a separate session: a fresh draw from a seed Shreyas gives,
   including concept families never used in any development draw, scored once.
5. **`bench/README.md`** (how to call the scorer) and **`bench/INTERFACE.md`** (what the box must hand the scorer;
   §6 below).

**Where things go.** The repo is public, so nothing sealed is ever committed before certification.
- `bench/sealed/` holds the sealed answers of the development draws. Add it to `.gitignore`. It lives only in the
  workspace the builder uses.
- The generator, and everything needed for certification, live outside the public repo: a private repository, or
  files handed to Shreyas.
- Committed: the development networks themselves (their weights are what the box must explain), the scorer code, and
  the README and INTERFACE files.
- After certification, everything is committed, for transparency.

---

## 2. Before you start

- **Read `SPEC.md`:** §2 (the definition), §4 (every requirement's test), §5 (the blueprint's contents) and §6 (the
  suite, the lines, the defaults).
- **Wait for the builder's `bench/VOCABULARY.md`,** its frozen starting vocabulary. S2's concepts must be impossible
  to write in it.
- **Reuse what exists:**
  - Rosetta's yardstick (`rosetta/yardstick.py`) for S1;
  - the saved copies and disguises (`docs/glass/info_measure_T2.txt`, the round 1 and 2 files) for S5;
  - AlgZoo networks, GPT-2 and the Leela networks for S7;
  - Tracr and InterpBench, both public, for S0. Their code is on GitHub. If downloading pretrained models is
    blocked, retrain from their code.
- **Times:** from the clock only (`TZ=America/New_York date`, in its own call).
- **Commits:** under Shreyas's identity (cybercryptixcoder, official.shreyas28@gmail.com), with no AI co-author or
  session lines.
- **Spending:** ask Shreyas before any. The S6 rehearsals (a fluid emulator, a small language model) may need GPU
  time.

---

## 3. What each tier contains (sizes are the defaults of `SPEC.md` §6)

**S0. Known circuits in realistic weights.**
- 10 Tracr-compiled programs and 10 InterpBench (SIIT-trained) models.
- The true answer is the published circuit.

**S1. Planted rules inside the starting vocabulary.**
- 10 networks, each trained on a sealed rule written in `bench/VOCABULARY.md`'s terms.
- Rosetta's yardstick is the pattern.

**S2. Planted concepts outside the starting vocabulary.**
- 10 concepts per draw:
  - **5 named:** ideas a domain expert would recognise, such as a chess pattern. A strong model may know these.
  - **5 synthetic:** combinations or operations with no human name, built so that no expert or model could recall
    them.
- Every concept must be provably impossible to write in the starting vocabulary within its length budget. Check this
  by exhaustive search over the vocabulary, and record the check.
- Draw concepts from families. Certification draws include at least 3 families never used in any development draw.

**S3. Planted staged computations.**
- 5 networks per draw, each computing a verdict in stages from intermediate concepts. At least one stage is
  search-like: playing moves forward, iterating, or taking a minimum over options.
- One of the five is the chess rehearsal: a small value network trained to imitate a sealed evaluation that looks two
  plies ahead along forcing moves (checks and captures) and adds sealed positional terms.

**S4. Planted mixtures, and rival accounts.**
- 5 networks per draw mixing a rule, a memorised random table (keyed by listed training items) and unused parts.
- Plus 2 networks with two valid accounts: the input distribution is built so that two different definitions agree
  on it, and both are sealed.

**S5. Must-fail.**
- Every saved copy and disguise from rounds 1 and 2.
- For each S1 to S4 network: a distilled small student, a shuffled-weight version, an untrained version, a memoriser
  (shuffled labels), and a program carrying a literal table tuned by selection.
- 3 networks with no structure.

**S6. One rehearsal per flagship, each sealed:**
- **chess:** the S3 chess network;
- **fluid:** an emulator trained on coarse-grained frames of a finer simulation, with a sealed closure term and a
  sealed shortcut (a longer time step, with a sealed correction), so R15 can be scored;
- **lying:** a small language model with a planted gate that switches it to false answers when a sealed context
  feature is present. Include held-out cases where the feature appears without being instructed;
- **weight reading:** a small reader network trained to report a property of another network's weights by a sealed
  method.

**S7. Real networks with partly known answers.**
- LD2 and t1: look-ahead along forcing lines, as confirmed on fresh positions (`docs/rosetta/PREREGISTRATION.md`, R2),
  and the capture variable (`docs/rosetta/FINDINGS.md` 3).
- GPT-2 small: month rollover (`lab/constructions/gpt2_month_rollover/`).
- Modular multiplication; AlgZoo M4,3.
- The known answers are written as partial true blueprints. The box must reproduce them, not be limited to them.

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
    held-out pairs.
- **Otherwise,** drop it, draw a replacement, and record the drop rate.
- **Report** the admission statistics, but not the mechanisms.

---

## 5. True blueprints and the lines

For every planted network, write its **true blueprint**: the sealed mechanism expressed in the form of
`bench/INTERFACE.md` (§6). The lines of `SPEC.md` §6 are computed by running the scorer on true blueprints, controls
and baselines, across at least three development draws:
- **Recovery line:** each sealed definition's own correlation and substitution score, minus its spread.
- **Completeness line:** the 5th percentile of the true blueprints' recovery, input by input, minus its spread.
- **Irrelevance line:** the effect of removing the sealed unused parts.
- **Family-wise control lines:** the 99th percentile of the best score among as many control variables (untrained
  networks, shuffled activity, random directions) as real ones tested.
- **Usability line:** the true blueprint's score in the bounded-reader test, minus its spread. This needs the
  frontier-model reader. If no key is available yet, mark it "pending", and compute it before Gate 5.

The lines live inside the sealed scorer. The builder sees each score next to "pass" or "fail", not the sealed
definitions.

---

## 6. The interface: what the box must hand the scorer

The scorer runs every test itself, on the networks, with its own intervention code. Write `bench/INTERFACE.md`
stating what a blueprint must provide, in information terms, so the builder can implement it any way it likes:
- **Parts.** Each part has:
  - an identifier;
  - a location (layer, position and a direction or feature vector, in a form the scorer's intervention code
    accepts);
  - a type (rule, lookup, irrelevant, unresolved);
  - for a rule, a definition the scorer can evaluate on inputs, in terms of the inputs or other parts;
  - for a lookup, its table;
  - the identifiers of its upstream parts;
  - for each claim, its evidence label and stated confidence.
- **Run alone.** Given inputs, the blueprint produces outputs and every part's value without the network.
- **Overview and systems.** The one-page overview, with a callable top-level predictor, and groups of parts with
  their roles.
- **Trace.** For an input, each part's contribution and a predicted minimal flip.
- **Edit prediction.** For an edit (a part and an operation), the predicted effect. The scorer applies the edit to the
  network itself.
- **Fingerprint.** For a named behaviour, a detector the scorer can run on activations.
- **Ledger.** Shares by type; the unresolved share by layer.
- **Reader rendering.** A navigable rendering a bounded reader can open part by part.
- **Rivals.** Alternative blueprints, if any.
- **Cost map (R15).** For networks with a reference method: which parts stand in for which steps, with cost and
  error.

---

## 7. The bounded reader (R10) and calibration (R12): defaults

- **The reader** may not run code.
- **Its budget** in output tokens per question is at most 10% of the multiply-adds in one forward pass of the
  smallest opaque program compared (a copy, or a distilled student). That is too few to hand-execute it.
- **Where that smallest program needs fewer than 10,000 multiply-adds,** the reader test is reported but does not
  gate, because hand-execution is cheap there.
- **Calibration** passes when the expected calibration error of the box's stated confidences, on the certification
  draw, is at most 0.10.

---

## 8. Certification

1. Shreyas gives a seed.
2. A separate session (you, or a fresh one with this brief and the private generator) draws a fresh suite. The draw
   includes the held-out concept families.
3. It runs the finished box from the builder's branch, from one start configuration, with no human input.
4. It scores the draw once, against the pass counts in `SPEC.md` §6.
5. It reports to Shreyas and the builder together.

There are no retries on a certification draw. A retry needs a new seed, and Shreyas's decision.

---

## 9. What you report back

Report only these:
- the interface (`bench/README.md`, `bench/INTERFACE.md`);
- the tiers built and their sizes;
- the admission statistics;
- which lines are computed and which are pending;
- where the private generator is kept.

Never report concept families, definitions, mechanisms or line values.
