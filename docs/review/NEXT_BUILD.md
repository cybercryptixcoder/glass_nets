# Glass Networks: what the system is not yet, and what the next build must be

*A research review, not an implementation plan. Written Nov 25–26, 2026 by Claude (Opus 5.5) from the whole record on
branch `liar` (with `kronecker`, `planets`, `kepler`, `decompiler`, `glass-networks`, `constructor-attempt`) and the
literature log on `main`. Every claim about the past points to a file. The running log of how this review was done
is `REVIEW_LOG.md`.*

*Status: first complete draft at 23:39 ET Nov 25 (clock); revised through 00:17 ET Nov 26 (final version of the first hour). Every revision is listed in
REVIEW_LOG.md.*

**How to read it.** §0 is the whole answer on one page. Then, in order of importance: §1b (what the system interprets
and why that framing limits it), §3.1–§3.4 (what the box hands back, how understanding is measured, how it
discovers, where language models reason), §6 (the four flagships and what "decoded" means for each), §10 (the first
five experiments). §2 is the full diagnosis, with evidence for every gap; §11 is a glossary.

---

## 0. The answer in one page

**What the system is today.** A set of strong instruments and a discipline of honesty:
- a calibrated causal judge;
- controls that caught real errors, and a detector for memorisation;
- an exact parameter decomposition with "rule or lookup" labels;
- a blind decompiler for group-structured networks;
- a law extractor and a teachability test.

What turned those instruments into results on hard targets was a Claude session doing research by hand, in a new
bespoke pipeline for almost every target. The automated reader/judge loop the code is organised around succeeded only
where the network was clean and its mechanism was of a kind the reader already knew (toys, compiled programs, a game
policy, modular multiplication, English verb agreement, AlgZoo networks of at most 32 parameters). It failed on
every dense or unfamiliar target, and the last three sprints did not use the Glass code at all.

**What stops it from being the magic box is not size. It is five assumptions built into how it interprets, and
what:**

1. **That an explanation is one artifact, written in one leap and accepted or rejected by one judge.** What a
   network was trained to do lives in a hierarchy (variables → operations → algorithm → knowledge), and it has to be
   built by many small verified steps. Today each instrument produces a single level: a top-level program, a parts
   list, or a law read from behaviour. Nothing connects the levels.
2. **That the concepts are known in advance.** Every instrument explains in a vocabulary someone supplied (task
   features, Fourier angles, 61 chess concepts, 38 physics terms, group theory). When a network uses something new,
   the only possible output is "unexplained residual". The chess residual (about a fifth of the value, located in
   tactical positions and in two causal channels, never named) is exactly where discovery was waiting.
3. **That the object is the decision and the weights, not the computation at inference.** Almost every test asks "is
   my quantity Q at site S, and is it used for the decision?" The steps the network takes, its own intermediate
   results, its control flow by regime and its approximations, are tested only as far as they hold an analyst's
   variable. "How it works" is the inference process, and that is what must be interpreted. The record agrees:
   every result that held up rested on inference-time evidence, and weights alone predict edit effects at Spearman
   0.17 against 0.97 from the hidden state (section 1b).
4. **That difficulty is parameter count, and the way up is a size ladder.** A clean 226k-parameter network got a
   short, verified account; dense 112-parameter ones never did, and yielded only to an exact decomposition.
   Difficulty is structure. The way in is top-down from what the network does, and bottom-up from decompositions,
   meeting in the middle.
5. **That networks trained on functions we already wrote down are where to look.** Almost every target computed a
   function people defined first (argmax, groups, Tracr programs, templated behaviours), so only the implementation
   could be new. Knowledge comes from networks that know something we do not (chess, physics emulators trained on
   finer truth), and the one physics target was built so that its answer was already inside it.

**What the box must be** (section 3):
- **The output: a Glass Model.** A hierarchical, executable, verified model of the model, with a residual ledger and
  a typed guarantee on every claim. A person reads it as an execution trace of the network's own algorithm, for any
  input.
- **The measures: six numbers instead of "verified/complete":**
  - faithfulness under intervention at every level;
  - completeness (the residual ledger);
  - compression against a shared, growing library (a rate–distortion curve of understanding, which replaces the
    copy rules with a theory);
  - proof compression (does the explanation make checking the network cheaper?);
  - transfer (can a fresh model or a person predict the network from the explanation alone?);
  - control (can the explanation be used to edit the network with every effect predicted in advance?).
- **The method: a residual-driven discovery loop that invents its own vocabulary.** Cartography → the network's own
  variables → synthesised definitions → composition into an algorithm and comparison with the textbook one →
  abstraction into new library primitives → knowledge a person can learn → confirmation under frozen tests.
- **Language models reason at every step**, in series, one role at a time: cartographer, observer, theorist,
  formaliser, experimenter, critic, abstractor, translator, librarian, prover, reader, planner, an informant (the
  target model's own reports, treated as hypotheses), and a puzzle-maker that writes the test networks. They propose
  and argue; code decides. That makes even the words of an explanation
  testable.
- **Hypothesis spaces with structure** (geometry, algebra, automata, dynamics, numerical operators, search,
  inference), each with a recogniser and a verifier. This is how "provably" extends past finite tables.
- **Learning across networks:** a motif library with recognisers, trained readers, and developmental, comparative and
  data-level lenses, so the hundredth network is cheaper than the first.
- **The public product becomes that library** (3.10): the concepts and mechanisms networks have learned, each a
  program with a recogniser anyone can run, with per-network Glass Models as the evidence.

**Verdict:** rebuild the centre, keep the instruments as parts (section 4).

**How to know it is working:**
- **the discovery score:** blind recovery of algorithms that are known to the experimenters and published nowhere;
- **the residual curve:** on a flagship, each round adds named, verified, teachable concepts, and the structured
  residual falls below what the classical library reaches.

The flagships that should pull the build are the ones Shreyas named (section 6): a chess network's strategies, how a
fluid model is cheap, what gives an LLM the ability to lie, and how an LLM reads weights. The fluid target has to be
rebuilt first, because as built it cannot pose the question (section 6). The research order is in section 9:
calibration ladder first, then the loop on it, then one flagship end to end with its residual curve. Section 10
lists the first five small experiments that test the premises the design rests on.

### Why the box is buildable: the argument from the record

Every stage of the box already works somewhere, in this repo or in the literature it has read. What does not exist
yet is their composition and the loop that invents vocabulary.

| Stage of the box | Where it already works |
|---|---|
| Recover structure blind, without being told what to look for | The decompiler: exact group tables, including A5's 60 × 60, from weights under a secret relabelling (decomp/RESULTS) |
| Decompose exactly, and say honestly what is memorised | The atlas: SPD with rule/lookup labels, known answers recovered on fresh 227k-parameter networks, a memoriser flagged (ATLAS_RESULTS A41–A44) |
| Read a law out of a network blind | Kepler: 5 true terms of 38, coefficients within about 2%, regimes and a resolution limit (kepler/FINDINGS §1) |
| Verify a full causal account of a real language-model behaviour, with a rival representation ruled out | Month rollover: 5 of 5 stages in GPT-2 small; the circle present but unused, replicated in three models (DISCOVERY.md) |
| Make a weak model strong with a predict-first experimental loop | Helix Probe on FIND (RESULTS_LOG R5) |
| Find verified strategy-level knowledge in a superhuman network | Kepler: a counter-textbook rule, confirmed on fresh games across three networks and in play (kepler/FINDINGS §5) |
| Invent and name abstractions with a language model plus compression | LILO-style library learning (section 7) |
| Turn an explanation into a proof that cannot be talked around | Compact proofs on group networks (section 7) |

---

## 1. What has been tried: the whole record in one table

"Level reached" uses one scale throughout, so the targets can be compared:

| Level | What it means |
|---|---|
| **B** | Behaviour only: what the network does, measured from outside (inputs → outputs) |
| **V** | A variable located: some quantity is shown to live at a place inside the network, causally |
| **M** | Mechanisms: the parts that compute, each causally tested |
| **A** | A composed algorithm: the parts put together into a program that runs in place of the network, aligned to it |
| **K** | Knowledge a person can learn: a law, concept or strategy stated in human terms, shown to be the network's |

"Guarantee" says what kind of evidence backs it: **S** = statistical on held-out samples, **E** = exhaustive over a
finite input set, **P** = a proof or sound bound. No result in the repo has a P.

| Phase (dates) | Target | Instrument and method | Who did the thinking | Level reached | Guarantee | Source |
|---|---|---|---|---|---|---|
| Helix Probe (Nov 3–5) | FIND functions, ARC-AGI-3 games, toy boxes | Experimenter / updater / theorist loop; the box judges predictions | Cheap LLM roles | B (a model of the box's behaviour) | S | README (history), RESULTS_LOG |
| Goal 1 (Nov 5–7) | Qwen 0.8B / 9B: remove addition, gender forms | LLM editor loop on weights | Cheap LLM | removal 2 of 12 runs; editors break the cue ("+"), not the skill | S | FINDINGS.md |
| Goal 2 (Nov 7) | Qwen: map parts → abilities | Open-ended loop | Cheap LLM | failed 4 of 5 criteria; gamed its score | S | FINDINGS.md, PROJECT_REVIEW §2 |
| Constructor (Nov 13–14) | Boolean nets, Tracr programs, mod-add and mod-mul (226k params), Breakout policy | Reader (DeepSeek Flash) writes `model(x, iv)` + `ALIGN`; causal judge | Cheap reader, judge calibrated by Claude | A on toys, compiled nets, Breakout and mod-mul (complete); mod-add 3 of 4 stages | S | EXPERIMENT_LOG E0–E12; `lab/constructions/INDEX.md` |
| Constructor, LM (Nov 14–17) | GPT-2: be-verb, countdown, **month rollover** | Same, on a GPT-2 engine with cuts; `lm_circuit` | Be-verb: cheap reader (complete). Countdown: stalled. Month: **Claude by hand**; readers 1 of 5 stages | Month: A (5/5 stages) + circuit; replicated in 2 more models | S | RESULTS_LM, DISCOVERY.md (month), E21–E40 |
| Glass rounds 1–2 (Nov 17–18) | AlgZoo RNNs 10–432 params; longest-cycle transformers | Constructor + anti-copy rules | Cheap reader | A only ≤ 32 params (M4,3 = rediscovery of ARC); copies above | S | LEARNINGS E1–E12, REPORT_ROUND2 |
| Units program (Nov 18–19) | ~90 networks (AlgZoo, Tracr, noise-trained) | Unit library fits, replacement tests, noise tolerance | Claude | V/M partial (comparator units recur; fragility measured) | S | UNITS_FINDINGS |
| Atlas (Nov 19–20) | AlgZoo RNNs to 1,376 params; lc transformers; 227k mod-arith transformers | SPD → triggers (trees over a feature vocabulary) → effects → rule-gated rebuild | Claude (pipeline); no LLM in the loop | M (5 of 6 held-out RNNs certified); mod-arith frequencies recovered from a vocabulary that contains them | S | ATLAS_RESULTS, SHOWCASE |
| Decompiler (Nov 21–22) | Group-law transformers under secret relabelling; a^b mod p | Geometry of embeddings → group theory → program; M1 exhaustive, M2 causal | Claude (tool); blind at run time | A, blind, on 8/12 abelian, 2/2 S4, A5, S3×S3; new mechanism (nested discrete logs) | E (M1 over every input) + S (M2) | decomp/RESULTS, POW_MECHANISM |
| Kepler (Nov 24) | Fluid video CNN; smoke-only model; Leela LD2, 744706, t1 | Concept probes, law extraction (sparse regression on a 38-term library), velocity swap; chess concept atlas, rule mining vs humans, play-outs | Claude; DeepSeek as "scientist" (0 of 59 concept programs accepted over three rounds) | Fluid: K (law with coefficients and regimes) + V/M (Biot–Savart kernel, velocity at block 5). Chess: B/K (behaviour rules; 78% of value explained by 61 classical concepts; residual unnamed) | S | kepler/FINDINGS, RESULTS |
| Planets (Nov 24) | SPOCK instability network | Fit a physics formula that matches/beats the network | Claude | B (network replaced, not opened) | S | planets/DISCOVERY |
| Kronecker (Nov 25) | ML claims on Kronecker vanishing | Recursive known theorems vs LightGBM | Claude | No network interpreted | E (ground truth exact) | SPRINTS_NOV25_REPORT Part 1 |
| Liar (Nov 25) | Pacchiardi lie detector on Qwen, Llama, Mistral | Failed-lie test; probe groups; attention knockout | Claude | B + V (lying "mode" carried by chat-template tokens; real-lie trace model-specific) | S | liar/DISCOVERY |

**What the table says, before any interpretation:**
- Level A with a blind, exhaustive check was reached once: the decompiler, on group laws.
- Level K was reached for a fluid model's law (by reading its behaviour against a term library) and for chess
  behaviour rules (by mining its choices against humans). Neither came from reading the network's internals.
- The cheap reader reached level A on clean networks with familiar kinds of mechanism: toys, compiled programs, a
  hand-written game policy, modular multiplication (226k parameters, complete, 97.3% behaviour; prior-only control
  0 of 4 stages), GPT-2's be-verb agreement (complete; prior-only 1 of 5), and AlgZoo networks of at most 32
  parameters (`lab/constructions/INDEX.md`). It failed on every dense AlgZoo network above 32 parameters, on GPT-2
  countdown, and on month rollover (1 of 5 stages). Everything else above level B was produced by a Claude session
  acting as the researcher. The stronger-reader test (R3) was never run.
- The last three sprints did not use the Glass code at all (`scripts/planets`, `scripts/liar`, Kronecker scripts are
  standalone; only Kepler reuses the Luminal client).

## 1b. What the system actually interprets, what it probes for, and how that framing limits what it can find

*(Added at Shreyas's request at 23:32. This framing sits under every gap in section 2.)*

The size of the networks is not the main limit. A more important one is the *question* each instrument asks:
it fixes what kind of answer can come back.

### What is being interpreted, instrument by instrument

| Instrument | The object it interprets | The question it asks | The answer it can return |
|---|---|---|---|
| Constructor and judge (`lab/judge.py`) | The network's **decision** (largest output) on a fixed input set | Is there a program whose variables, placed at named sites by one affine map each, predict the decision and the effect of swapping those variables? | A map from the program's variables to sites, with pass/fail per site |
| LM tasks (`lab/lm.py`, `lab/lm_tasks/`) | One behaviour on **templated prompts of equal length**; the construction's input is not the text but **one row of integer features per prompt** chosen by the analyst (month1, year, month2) | How does the network compute f(analyst's variables)? | An account in the analyst's variables. Anything the network uses that is not one of those variables is invisible by construction |
| Units program (`lab/unitmap.py`) | Each neuron at each step | Which unit from a fixed library (running max, 2nd max, rank, …) does it compute? | A label per neuron, with confidence |
| Atlas (`lab/atlas.py`) | Rank-1 pieces of weight matrices | **When** is each piece needed, as a rule over a supplied feature vocabulary, and which output does removing it move? | A parts list with input-conditions |
| Kepler probes (`kepler/core/probe.py`) | Activations at a site | Is a **known** concept linearly readable here, above an untrained copy? | Where known concepts live. Kepler's own caveat: linear quantities are readable even from random filters |
| Kepler laws (`kepler/core/law.py`) | The model's implied time derivative | Which terms of a **candidate library** reproduce it? | A law in library terms |
| Decompiler (`lab/decomp/`) | Embedding geometry and the MLP's input/output | Which **algebraic structure** do the symbols form, and what is the law? | A group and a program, for group-like tasks |

### The common pattern

Almost every question has the form: **"is quantity Q, which I chose, located at place S, and is it used?"** That is
localisation plus necessity and sufficiency. It is the field's standard question (causal abstraction, probing,
patching), and it is answerable and testable. But it has three built-in ceilings:

1. **The analyst's prior is the ceiling.** Q comes from the analyst: a task variable, a library unit, a textbook
   concept. The network's own intermediate results that nobody has named can only appear as "residual".
2. **It interprets a decision, not a computation.** The unit of evidence is "the same decision on this input". The
   steps the network takes to get there are tested only as far as they hold an analyst's variable.
3. **It interprets one behaviour at a time.** A templated task, a fixed input set. How the network handles the space
   of inputs as a whole, its regimes and how it switches between them, is outside the question. The project already
   met the cost of this on Nov 7: **the model is not organised by our behaviour labels.** Qwen's late layers do one
   shared "turn this word into a related word" job across translation, capitals, word forms and young-animal names,
   and layers expected to handle arithmetic affected gender words (FINDINGS.md, Goal 2). A behaviour-by-behaviour
   study slices across the model's own operations.

### Today's workflow, drawn

```
 analyst picks a behaviour ──► analyst picks a vocabulary ──► instrument asks ──► certificate
 (a template, a task,          (task features, a unit          "is Q at S,          (pass / fail,
  a decision)                   library, 61 concepts,           and is it used?"     or a residual
                                38 physics terms)                                    nobody can name)
```

Every arrow carries the analyst's choice forward. The network contributes answers to questions it was asked, never
questions of its own.

### What "how the system works" requires instead: interpret the inference

Shreyas's point, "the information lies in inference", names what is missing. A forward pass is a computation. What a
person wants to understand is that computation as a **process**:
- which intermediate results the network computes, in its own terms, whether or not anyone has named them;
- which operation turns each into the next (compare, route, look up, accumulate, search, approximate a kernel);
- which path the computation takes for which input (its control flow, the regimes);
- what is preserved and what is thrown away (its invariants and its approximations);
- and why it computes this way and not the textbook way (what the method buys it: cheapness, robustness, margin).

The month-rollover write-up is the one place in the repo with this shape, written by hand. Its route section reads
like a trace (paraphrased from `lab/constructions/gpt2_month_rollover/DISCOVERY.md`): month1 is bound to the year at
the year token, the final token reads it from there, by depth 8 each month is reduced to one position on a line, the
two positions are compared, and the answer is decoded as a year token. It is also the one hard-target result that
held up.

**So the product should be an execution trace of the network's algorithm, for any input, in human terms, with every
step aligned to the network and verified, plus the program whose traces those are.**

Think of a debugger for networks that works at the level of the algorithm. Step through its computation on a chess
position and read something like this (invented, to show the level):
- today's level: "W_h.6 is ON when x_t is 0.64 below the max";
- the level wanted: "it is checking whether the knight fork on f7 survives the reply …xf7, along these lines, using
  these features".

The catalog's playground (per-step mechanisms on one input) is a primitive form of this view. The rest of this
document is about what it takes to fill it with the network's own steps rather than the analyst's.

### The framing is certification, not discovery

The system's own vocabulary shows its stance: a **judge**, a **certificate**, a **catalog** of certified entries
(INTENT: "a public library of constructions … each one ships with a certificate").
- Most of the engineering went into the measuring side: the judge's code changed in 22 commits in three days, then
  five copy rules, an information measure, a copy frontier (PROJECT_REVIEW §7).
- That is the stance of quality assurance: something is produced elsewhere, and the system decides whether to
  accept it.
- But nothing can be certified until it has been found, and the finding was left to a reader that could only recall
  or copy.

Certification is the last stage of a discovery instrument, not its centre. Kepler is the one place where the stance
turned to discovery (its intent: "a single tool that can make real discoveries", kepler/INTENT), and it is where the
most interesting results came from.

### Weights first was the founding frame; inference is where the evidence came from

The project's definition of a white box, from its own survey, is "an executable model derived from N's **weights**"
(`main:docs/literature/SYNTHESIS.md` §10), inherited from weight-space programming. The reader reads weights, SPD
decomposes weights, and the decompiler starts from the geometry of the embedding weights (it also reads the MLP's
activations over the finite table, which is part of why it works). Yet every result that held up rests on evidence from
inference:
- the month line: behaviour models, then interchange on activations;
- Kepler's fluid law: the model's implied dynamics; its velocity: probes and a swap;
- the atlas's triggers: importance measured over inputs;
- the liar sprint: the probes' answers and attention knockouts.

The reading literature says the same thing in numbers. From weights alone, edit effects are predicted at Spearman
0.17 (0.11 with no weights at all); from the true hidden state, at 0.97 (`docs/literature/wsp/K1_STATE_OF_ART.md`).
**So: inference first, for discovery; weights second, to generalise a mechanism once found to every input and to
prove things about it.** The weights are where a mechanism's reach can be proven. The activations are where it can be
found.

### What is probed for should flip

- From **"is my concept here?"** (supervised probes, fixed libraries) to **"what is here, and what is it for?"**:
  - find the variables the downstream computation actually uses;
  - then ask what they are (synthesise definitions, compare with known concepts, name them);
  - then ask what they are for (which later steps read them, what breaks without them).
- From **"which decision?"** to **"which computation?"**: score an explanation on whether it predicts the network's
  intermediate states and the effects of interventions at every step, not only the final answer.
- From **one templated behaviour** to **the model's whole competence**: start from a map of what the model does and
  where it behaves differently. Find its shared operations first, and explain behaviours as uses of them. Choose
  what to explain from that map, not from a template.

### What the targets were: mostly networks trained on functions we already knew

Most targets so far were trained to compute a function that people wrote down first: 2nd argmax, argmedian, longest
cycle, a + b mod p, a group law, a Tracr program, a templated language behaviour. On such a network the only thing
left to discover is *how* it implements the known function. That can be surprising (nested discrete logarithms for
a^b mod p), but the knowledge itself was ours to begin with.

The discoveries Shreyas is after need the other kind of target: **a network that knows something people do not**,
because it learned from nature's data or from self-play beyond human level. Chess networks, physics emulators trained
on fine simulations or observations, and models trained on mathematical data whose general rule is unknown (the
Kronecker setting) are of that kind. Only on those can interpretation hand back knowledge, not just mechanism.

So the two kinds of target have two jobs:
- **networks trained on known functions** are the calibration ladder: they test whether the box recovers mechanisms;
- **networks that know more than we do** are the discovery targets: the place where the box's output can be new
  knowledge.

### What the record has already taught about interpretability in general

Before the diagnosis, the positive side. These lessons came out of the runs, hold beyond this project, and the next
build should treat them as known:

| Lesson | Evidence |
|---|---|
| Whether an approximate account *can* be faithful looks measurable in advance: a network's noise tolerance (how much noise flips 5% of its decisions) predicts it. The repo calls this a candidate result, replicated across 13 networks plus 15 training runs | UNITS_FINDINGS F3, P5, U11: AlgZoo nets flip at ~1% noise, the modular nets tolerate ~80× more |
| A representation can be present and unused: the month circle is readable at R² 1.00 and causally inert for this decision | lab/constructions/gpt2_month_rollover/DISCOVERY.md, E32 |
| Models are not organised by our behaviour labels; one operation serves many behaviours | FINDINGS.md, Goal 2 |
| Editors that do not understand a model fake success by breaking the cue, not the skill | FINDINGS.md, Goal 1 |
| An optimiser pointed at a score games the score, and every rule written against the gaming gets beaten by a new disguise | Goal 2; LEARNINGS E1, E10, E11 |
| A good predictor need not implement the law, and whether it does changes by regime (the fluid model follows the law in equilibrium, not in transients or out of distribution) | kepler/FINDINGS §1 |
| Memorisation can be detected and labelled instead of explained away (rule vs lookup) | ATLAS_RESULTS A41 |
| Blind structure recovery is possible when the hypothesis space has deep structure (groups, under a secret relabelling) | decomp/RESULTS |
| A celebrated black-box detector can be reading the setup rather than the property (the lie instruction, not the lie) | liar/DISCOVERY |
| Machine-learning "edges" must be compared with what known theory already decides | SPRINTS_NOV25_REPORT Part 1 (Kronecker) |
| Predict-first, experiment-choosing loops make weak models strong on black boxes | RESULTS_LOG R5 |

---

## 2. The diagnosis: what the system is not

Fifteen gaps, in four groups. Each says what the system does now, the evidence, why it blocks the magic box, and
what is needed instead. None of them is a bug. Each is a capability or a stance that is missing.

### Group I. The explanation itself: what the box hands back

**Gap 1. There is no explanation object that spans levels.**
- *Now.* Each instrument produces one artifact at one level:
  - the constructor, a top-level program;
  - the atlas, a bottom-level parts list (rank-1 pieces with "when it fires" rules);
  - Kepler, a top-level law or behaviour rule;
  - the decompiler, a top-level program for one family.
- *Evidence.* The atlas's own limits: triggers are often time gates ("after step 2"), and what a mechanism computes
  is told only as a profile of the unit types it reads (ATLAS_RESULTS, Limits). Chess rules "are not a switch inside
  the network" (kepler/FINDINGS §5).
- *Why it blocks.* What a network was trained to do lives in the middle: concepts built from features, sub-routines
  built from concepts, strategies built from sub-routines. A parts list is not understanding, and a top-level law
  that is not connected to the parts is not an explanation of *this network*.
- *Needed.* A hierarchical explanation: every level is an executable model, and each level is verified against the
  level below it by causal alignment. The top level is the behaviour; the bottom is the weights; understanding is the
  verified chain between them.

**Gap 2. "Explanation versus copy" was attacked with rules about the program's text, not with a theory.**
- *Now.* Five literal-counting rules, each beaten (decimals, rescaling, integers over a scale, tables plus indices).
  Then an information measure against a copy frontier. Its first version failed its pre-registered test on tiny
  networks (DIARY 11:35); a revised version (T2) placed every one of 30+ saved copies and disguises at S ≥ 0.91
  (PLAN_SCALING, findings 1), but it was kept as a reported signal, not a gate. That measure is the right instinct;
  the two ideas below generalise it.
- *Evidence.* LEARNINGS E1, E3, E10, E11; PROJECT_REVIEW §6; INTENT change 2 "REOPENED".
- *Why it blocks.* Without it, "certified" can mean "copied", and the reader is rewarded for copying.
- *Needed.* Two ideas that the repo's own literature log already holds, put at the centre:
  1. **Compression against a shared library.** An explanation's length is measured in a vocabulary (a library of
     concepts and operations) whose cost is paid once and shared across many networks and inputs, the way a
     mathematician's definitions are. A copy cannot be short in such a vocabulary, because raw weights are not
     reusable primitives: the total cost is the library's length plus each explanation's length given the library,
     so a "primitive" that only one network uses pays its full length, and a disguised weight table is exactly such
     a primitive. Only structure that recurs (across inputs, regimes, layers or networks) gets cheap. This is the
     minimum-description-length idea of library learning (DreamCoder-style wake/sleep), applied to networks
     instead of programming tasks.
  2. **Compression of the proof.** An explanation is real to the extent that it makes *checking* the network cheaper:
     it turns into a verifier that bounds the network's behaviour faster than brute force (Gross et al. 2024; Wu,
     Jaburi, Drori & Gross 2025, `main:docs/literature/aether/arxiv-2410.07476.md`). A copy's verifier costs as much
     as running the network. This measure "cannot be talked around", because an over-claiming account is refuted by
     arithmetic.

  Both ideas meet, by construction, the three conditions the project itself set for any replacement of the copy
  rules (REPORT_ROUND2, corrections §3): they measure information or computation, not how numbers are written; their
  pass lines come from controls (the copy frontier and shuffled networks on one side, the task rule on the other);
  and a copy in any encoding fails them, because a weight table is neither reusable across networks nor cheaper to
  check than the network. Condition (c), rejecting every saved copy and every new disguise before adoption, still has
  to be shown on the saved runs before either is used as a gate.

**Gap 3. No guarantees beyond held-out statistics.**
- *Now.* Every certificate is a held-out sample statistic. Early on, the project decided that "'proven accurate'
  isn't possible beyond tiny nets" (glass/DIARY, Background). Even the decompiler, which checks every input of a
  finite table, does not call that a proof.
- *Why it blocks.* Shreyas's standard is "provably, not suggestively". A sample statistic cannot rule out the rare
  input where the account and the network part ways, and that is exactly where a chess network's "extra knowledge"
  sits (tactical positions: 36% explained vs 81% in quiet ones).
- *Needed.* Typed guarantees, attached to every claim, and an explicit ladder from cheap to strong:
  - **exhaustive** over a finite domain (a proof by enumeration; the decompiler already has it);
  - **regional proofs** (bound the network against the explanation over a region of input space, e.g. one
    ReLU-pattern region or one game phase);
  - **compact proofs** (a verifier faster than brute force with a stated bound);
  - **probabilistic bounds** with stated sample complexity on a stated distribution;
  - **adversarial search** for counter-examples, reported with its budget.
  "Provable" then has a precise meaning per claim, and the certificate says which kind each claim has.

### Group II. Discovery: how explanations are found

**Gap 4. The system can test an explanation but cannot generate one.**
- *Now.* The judge is a strong falsifier. The generator was a cheap reader, which wrote the textbook algorithm and
  patched toward the network (E7) or copied it (E1). Every success had its hypothesis supplied from outside:
  Fourier features, the circle-vs-line rivalry, group theory, a physics term library, a chess concept library.
- *Evidence.* E7, E1, F17 in REVIEW_LOG; PLAN_SCALING: R6 showed total-variance analysis misleads, and R7–R8 show
  the cheap reader "can't find decision-relevant low-dimensional quantities alone". The proposed tool for that (R9) was never built.
- *Why it blocks.* A magic box must find the variables of a network nobody has thought about. Testing is the easy
  half.
- *Needed.* A hypothesis engine whose first job is to find the network's *own* variables without being told what
  to look for: subspaces that downstream computation actually uses (found by substitution, not variance),
  decompositions of weights and activations, invariants across training runs. Its second job is to propose what
  those variables are (Gap 5).
- *The unit question it must settle operationally.* The field has no agreed unit: sparse-dictionary latents depend
  on the seed (30–42% shared), one group finds native neurons mostly clean, another blames the basis for
  non-sparse weights (`main:docs/literature/SYNTHESIS.md` §3, §9). The box should not pick a side by assumption.
  It should define a variable by what it does: **read by later computation** (substitution), **stable across
  training runs and decompositions** (invariance), and **compact** (few dimensions). Whatever basis meets those
  tests is the network's own.

**Gap 5. A closed vocabulary: it can confirm known concepts but not invent new ones.**
- *Now.* Every instrument explains in a vocabulary the analyst supplies:
  - atlas triggers are trees over task features, and the A35 failure was answered by adding the answer's own
    vocabulary ("angles k·v mod p", A37). That was done properly, as a new version judged on fresh networks, but it
    shows the shape of the method: when the network's concept is missing, a person adds it;
  - Kepler: 61 classical chess concepts, 38 physics terms, 144 situations × 42 move types;
  - the decompiler: circles, groups, and three closed forms (k·t, k·t², k·log(t + s)).
- *Evidence.* Chess: classical concepts explain 78% of LD2's value and the learning curve flattens at 79–80%. The
  residual was located (tactical positions; channels 22 and 15, which are causal) but never named. The LLM
  "scientist" wrote 59 concept programs over three rounds (0 of 24; 0 of 15 after refinement with round-1 scores;
  0 of 20 from minimal pairs on channel 22), and 8 rules synthesised from the channels also failed
  (kepler/FINDINGS §4; kepler/DIARY, Nov 24). The agent's own conclusion: "The residual is mostly not a simple static
  feature." The one feature accepted was a calculation feature (capture resolution). Both point the same way: what is
  missing from the vocabulary is dynamic (calculation, search), not another static board pattern.
- *Why it blocks.* Discovery *is* the invention of a concept that was not in the vocabulary. A system that can only
  say "unexplained residual" when the network uses something new cannot discover.
- *Needed.* **Library learning**: new primitives are defined as programs from the network's own units, verified by
  alignment, given a name and examples, and added to the vocabulary, so the next explanation can use them. The
  residual is the engine: each round, take what the current vocabulary cannot explain, find the internal variables
  that predict it, synthesise definitions for them, verify, add, repeat. The residual should shrink round by round.
  That curve is the measurable signature of discovery.

**Gap 6. Bottom-up by parameter count, on the wrong axis.**
- *Now.* "The path is a ladder of whole models" (glass/DIARY, Background): 10 → 32 → 112 → 432 → 1,376 parameters.
- *Evidence.* Difficulty tracks structure, not size. The 226k modmul transformer is cleanly explainable (fidelity
  97.4%, 7/7 claims); 112-parameter AlgZoo nets are not, because they are dense and precise (5% of decisions flip at
  ~1% noise) (PLAN_SCALING findings 2–3; UNITS_FINDINGS F3). Days went into ARC's deliberately hard tiny networks.
- *Why it blocks.* The targets that matter (chess, fluids, LLMs) are millions to billions of parameters. At the rate
  of the ladder they are unreachable, and they are not harder in the way the ladder assumes: they hold many clean
  features plus a messy remainder.
- *Needed.* Top-down and bottom-up at once, meeting in the middle:
  - **top-down** from the network's competence (what it does, in which regimes, which concepts its output depends on);
  - **bottom-up** from decompositions (features, mechanisms);
  - **the middle** is where they are joined by alignment.
  The messy remainder is not a reason to stop. It is accounted for in the residual ledger (3.1).

**Gap 7. Knowledge and mechanism are never connected, so the work drifts outside-in.**
- *Now.* Kepler's fluid law was read from the model's input→output dynamics; its chess strategy came from mining
  choices against humans, and the cause tests showed the rules are not internal switches. Planets replaced the
  network with a formula. Kronecker compared ML with known theorems and opened no network. The liar sprint tested a
  black-box detector.
- *Evidence.* REVIEW_LOG F12–F14.
- *Why it blocks.* Shreyas's questions are about the inside: *how* is the fluid model cheap, *what* gives an LLM the
  ability to lie. When the inside-out route has no capability advantage, a rational researcher takes the
  outside-in route, which is cheaper and always yields a publishable comparison.
- *Needed.* Every knowledge claim (a law, a strategy) gets an implementation claim (where and how the network
  computes it), tested by alignment. **The discovery is usually the difference between the two**: where the
  network's algorithm departs from the textbook algorithm. For the fluid model that difference is its approximation
  scheme (what it truncates, what it coarse-grains, what closure it learned for the scales it cannot resolve). The
  resolution limit Kepler measured (advection fading at wavenumbers 9–13) is a symptom of that scheme, seen from
  outside.

**Gap 8. Passive observation: no experiment design.**
- *Now.* The instruments analyse fixed input sets. Minimal pairs (K9) and board edits (K11h) were run a few times,
  by hand.
- *Why it blocks.* Telling rival hypotheses apart needs inputs and interventions chosen to separate them, the way the
  month finding was settled by removing the line versus removing the circle.
- *Needed.* An experimenter that chooses the next stimulus or intervention to discriminate the live hypotheses.
  This is Helix Probe's original insight (an experimenter, an updater and a theorist, judged by the box's answers),
  which was dropped when the project moved to the constructor. It belongs back in, one level down: the "box" is now
  the network's insides.

**Gap 9. The analyst picks the behaviour.**
- *Now.* Each study starts from a behaviour someone chose (month rollover, 2nd argmax, lie vs truth). Kepler found
  the fluid model's regimes (equilibrium, transient, out of distribution) by hand.
- *Needed.* Behaviour cartography as the box's first stage: map what the model does and where it behaves
  qualitatively differently (regimes), using its outputs *and* its internal states, before any mechanism work. For
  "hand it any model" this stage cannot be optional.

### Group III. Reach: what it can be pointed at

**Gap 10. No instruments at the scale of the targets that matter.**
- *Now.* SPD runs 25–90 minutes on networks of hundreds of parameters and hours at 227k; attention query/key weights
  get no mechanisms (an open limit after three redesigns). The LM engine covers the GPT-2 family. There are no
  sparse dictionaries, transcoders, cross-layer transcoders or attribution graphs. The liar sprint used black-box
  probe features and attention knockouts. Kepler's chess "units" are raw channels.
- *Why it blocks.* Lying in a 7–8B model, strategy in a Leela transformer, physics in a video model all need
  feature-level decompositions of large networks.
- *The query/key gap is not a corner case.* In a transformer, queries and keys decide routing: which information
  moves to which position at inference. The month finding's route (month1 → the year token → the final token) is a
  routing story, and the lying mode the liar sprint found is carried by specific template tokens, which is routing
  too. The atlas could not give query/key weights any mechanisms (an open limit after three redesigns), and the
  field names query/key the hardest part (SYNTHESIS §3, hurdle 6). A box that interprets inference has to explain
  routing, so this is on the critical path.
- *Needed.* A decomposition layer that runs at that scale (dictionaries per layer and across layers, attribution
  graphs, parameter decomposition where it scales), with attention handled, and trained once per model, then reused
  by every question.

**Gap 11. A static view of one network.**
- *Now.* One trained network at a time. No training checkpoints, no data attribution, no model diffing. Multi-seed
  comparison appears once (units across 5 seeds).
- *Why it blocks.* Some of the strongest evidence about what a mechanism is for comes from *when* it forms in
  training, *which data* creates it, and *what differs* between a model that has an ability and one that does not.
  For "what gives an LLM the ability to lie", a diff between an honest and a deceptive variant, or a base and an
  instruction-tuned model, is a sharper instrument than probing one model.
- *Needed.* Developmental (checkpoints), comparative (seeds, sizes, variants; what is invariant is the essential
  algorithm, what varies is incidental) and data-level (attribution of a mechanism to training examples;
  `main:docs/literature/aether/arxiv-2601.21996.md`) views, as standard lenses.

**Gap 12. No accumulation, no amortization.**
- *Now.* Every network is interpreted from scratch. Findings accumulate as prose (DIARY is 2,166 lines). The unit
  library and the atlas vocabularies are per-study. The trained reader (DIRECTIONS K2) and a learned decompiler were
  never built, though the literature log holds the precedent (Neural Decompiling of Tracr Transformers,
  arxiv-2410.00061: 73% functionally equivalent programs; graph metanetworks).
- *Evidence that the lever exists.* Function-level units recur: comparator units in 5 of 5 training runs at 8
  neurons, none in shuffled controls (UNITS_FINDINGS F1, U8). Shreyas's own hypothesis (Nov 18): networks are built
  from shared basic units composed into larger tools.
- *Needed.* An executable, growing library of motifs, each with a recogniser that finds it in a new network, so that
  the hundredth network is cheaper than the first. And readers trained with verifiable rewards on synthetic networks
  with known programs and on verified explanations. This is also where Shreyas's "how do LLMs read weights" question
  becomes a capability: the reading skill is trained and measured, not assumed.

### Group IV. The researcher: who runs it

**Gap 13. The real engine is a frontier agent, but the system is not built for one.**
- *Now.* The code is organised around a reader/judge loop that succeeded only on clean networks with familiar
  kinds of mechanism (section 1). Every result on a hard target came from a Claude session writing bespoke scripts
  (the last three sprints did not import the Glass code). The agent's understanding of a network lives in diaries
  and chat summaries, which are lost or compressed between sessions.
- *Also.* A large share of the agents' time went into plumbing: relaying Colab results to GitHub by hand (Colab
  had no push credential), restarting jobs lost when a session paused, multi-hour SPD runs with the agent waiting
  (glass/HANDOFF; ATLAS §6; the DIARY's many "waiting" entries). The compute substrate shaped which questions were
  affordable, and it favoured small ones.
- *Why it blocks.* Discovery compounds only if what is understood about a network is kept in a form that can be
  queried, tested and built on.
- *Needed.* The product's core should be a persistent **model of the model**: a structured graph of claims
  (variables, mechanisms, concepts, laws), each with its alignment, its evidence, its guarantee type, its residual
  and its open questions. The agent's job is to extend that graph; instruments are its standard verbs; the catalog is
  a view of it. It also serves Shreyas's standing rule that "the repo is the handoff" (WORKING_METHOD §3): today that
  rule is kept by updating four or more prose files after every event; a structured store that is current by
  construction is a handoff that cannot drift, and the prose files become views of it.

**Gap 14. One mode of work, tuned against a gaming optimiser.**
- *Now.* `CLAUDE.md`'s rules (ledger entry before any code, frozen measures, one change per round, pre-registration)
  came from rounds where a cheap reader gamed the judge. They are the reason the results are credible.
- *Why it limits.* Discovery needs wide, cheap exploration: hundreds of hypotheses, most wrong. Under one
  confirmatory mode, exploration is slow and small, and experiments shrink to what can be pre-registered.
- *Needed.* Two explicit modes. **Explore**: cheap, wide, unregistered, always labelled as exploratory. **Confirm**:
  frozen, pre-registered, held out, the only mode that produces claims. The honesty rules stay; they move to the
  boundary between the modes. (Kepler already did this informally: mining, then confirmation on fresh games.)

**Gap 15. Targets were searched for, not chosen to pull capability.**
- *Now.* After the instruments, a filtered search for "head-turner" targets (planets, Kronecker, liar) produced
  results Shreyas judges modest, and targets he does not care about. Meanwhile instruments stopped at validation:
  the decompiler's own plan said "if yes: point it at networks whose algorithm is unknown" (decomp/PLAN, Intent),
  and after its held-out passes the project moved on to Kepler instead.
- *Why it limits.* The targets he does care about (chess strategies, how a fluid model is cheap, what lets an LLM
  lie, how an LLM reads weights) need capabilities that do not exist yet. Searching for targets that today's
  instruments can finish selects small ones.
- *Needed.* Fix a few flagship targets and let each one pull the capabilities it needs (section 6).

### Each gap, and where the design answers it

| Gap | Answered in | Measured by |
|---|---|---|
| 1. No explanation object that spans levels | 3.1 Glass Model (L0–L4); 3.3 D composition | Faithfulness at every level (3.2 item 1); Q4 |
| 2. Copy vs explanation by rules | 3.2 items 3–4 (shared-library compression; proof compression) | The rate–distortion curve of understanding |
| 3. No guarantees beyond statistics | Gap 3's ladder; 3.5 hypothesis spaces with verifiers | Guarantee type per claim; Q5 |
| 4. Cannot generate explanations | 3.3 B variables; 3.4 theorist, observer, experimenter | Q1 on the calibration ladder |
| 5. Closed vocabulary | 3.3 C definitions, E abstraction; 3.6 library | Q2, Q3; the residual curve (Q9) |
| 6. Wrong axis (size) | 3.3 A cartography (top-down) + B (bottom-up) | Flagships at real scale (section 6) |
| 7. Knowledge and mechanism unconnected | 3.1 L4 linked to L2–L3; 3.3 D comparison with reference algorithms | Every L4 claim carries an implementation claim |
| 8. No experiment design | 3.4 experimenter; the Helix skeleton (3.3) | Information gained per experiment |
| 9. Analyst picks the behaviour | 3.3 A cartography | Coverage of the model's competences |
| 10. No instruments at scale | 3.7 verbs; decompositions trained once per model | Flagships 1, 3, 4 |
| 11. Static view of one network | 3.6 developmental, comparative, data lenses | Used in flagship 3 (lying) first |
| 12. No accumulation | 3.6 motif library, trained readers | Q3; cost of the n-th network |
| 13. The agent is the engine but unsupported | 3.1 Glass Model as the persistent store; 3.4 roles; 3.9 questions in words | Whether results compound across sessions |
| 14. One mode of work | 3.7 explore / confirm | Claims only from confirm mode |
| 15. Targets searched for, not chosen | Section 6 flagships; section 9 order | The residual curve on a flagship |

---

## 3. What the magic box must be

This section describes the box by what it produces, how it measures understanding, how it discovers, where language
models reason inside it, and what it stands on. It is a design of the research system, not a tech stack.

### 3.1 What it hands back: a Glass Model (a verified model of the model)

For a network N, together with the inputs or environment it works in, the box returns one structured object with five
levels. Each level is executable, and each is aligned to the level below by tested interventions.

| Level | Content | Illustration, chess network (hypothetical, not a finding) | Illustration, fluid model (Kepler's actual findings where marked ✓) |
|---|---|---|---|
| **L0 Network** | weights, activations, training checkpoints if given | Leela t1 | the video CNN |
| **L1 Variables** | the quantities the network's later computation actually reads, found without a supplied vocabulary, each with where it lives | a feature that is high when a piece is overloaded on two defensive duties | the internal velocity field, readable from block 5 and used there (velocity swap) ✓ |
| **L2 Operations** | how each variable is computed from earlier ones: small programs, each aligned | "overload = defends(X, a) ∧ defends(X, b) ∧ attacked(a) ∧ attacked(b)", computed at layers 3–5 from attack maps | velocity from vorticity through a learned, nonlocal Biot–Savart-like kernel (correlation 0.974 with the exact response) ✓; the advection product built in blocks 3–6 ✓ |
| **L3 Algorithm** | the composed program: dataflow plus control flow by regime. Its execution trace on any input reproduces the network's intermediate states and its output | "evaluate material; if forcing moves exist, expand them to depth 2 along these features; …" | "reconstruct velocity, advect, then handle the scales it cannot resolve (it fades above wavenumber 8 ✓) in some way not yet known": that unknown is the discovery target |
| **L4 Knowledge** | laws, concepts and strategies the algorithm embodies, in human terms, each linked to the L2/L3 parts that implement it, with the departures from the textbook made explicit | "when lost against strong play, trade queens", and where in the network that preference is computed | the law with coefficients, and the approximation scheme that makes it cheap |

The same thing as a picture (chess labels invented, to show the level of each line):

```
 L4 KNOWLEDGE    "when lost against strong play, trade queens"          residual: what no rule covers
        │  implemented by                                    ▲
 L3 ALGORITHM    trace: evaluate → expand forcing lines → compare → choose   residual: steps not yet aligned
        │  composed of                                       │  every link is a tested alignment
 L2 OPERATIONS   overloaded(sq) = …, material(board) = …    │  (substitution, interchange,
        │  computed from                                     │   partial/edit, necessity)
 L1 VARIABLES    component k at layer 5, per square          │  + a guarantee type on each claim
        │  living in                                         │
 L0 NETWORK      weights, activations, checkpoints     ──────┘   residual: memorised / noise / unnamed
```

These levels are Marr's three levels of analysis for an information-processing system (Marr, *Vision*, 1982),
applied to a network and each linked to the next by tests: L4 is the computational level (what problem is solved,
and why this solution), L2–L3 the algorithmic and representational level (which representations, which algorithm),
L0–L1 the implementation. Today's instruments each sit at one of Marr's levels; the Glass Model is the verified
chain through all three.

Two things run through every level:
- **A residual ledger.** What is not yet explained, where, how much (share of loss, of decisions, of intermediate
  state), and of what type: memorised cases (A41's "lookup"), noise, or structured but unnamed. "Complete" means the
  ledger is accounted for, not that it is empty: a memorised table is an honest, complete explanation of a part that
  memorises.
- **A guarantee per claim**, typed (section 2, Gap 3): exhaustive, regional proof, compact proof, probabilistic bound
  with sample size, or adversarial search with budget. Plus its provenance: which exploration proposed it, which
  frozen test confirmed it.

The trace view (section 1b) is how a person reads a Glass Model: pick an input, step through L3, open any step to see
its L2 program and L1 variables, see where the network and the model agree and where the residual sits.

### 3.2 How understanding is measured: six numbers instead of "verified / complete"

1. **Faithfulness under intervention, at every level.** The judge's tests (substitution, interchange, partial and
   edit, necessity), generalised from "the same decision" to "the same intermediate states", so that an account is
   scored on the computation and not only on the answer. Scored with measures that fit the output (a distribution
   over moves or tokens, a field, a value), not only "the same largest output", which is all the current judge
   compares. Scored in distribution and outside it (the law-or-lookup
   question, answered regime by regime as Kepler did for the fluid law), with the class of alignment maps declared
   and restricted, because with unrestricted maps any network matches any algorithm (Sutter et al., in
   EVIDENCE_LADDER).
2. **Completeness.** The residual ledger, per level, against the task rule and against controls.
3. **Compression against a shared library.** Description length of the explanation in the library's vocabulary,
   with the library's own cost amortised over every network and input it explains. Plotting fidelity against
   length gives a **rate–distortion curve of understanding** for each network. A copy is one point on it (high
   fidelity, long); the textbook rule is another (short, low fidelity); an explanation is a point that dominates
   both. This replaces the literal-counting rules with a theory. Concretely:
   - the horizontal axis is description length in the library's terms (bits, with the library's cost amortised);
   - the vertical axis is faithfulness under intervention at intermediate states (item 1);
   - the references are the copy frontier (the network's own weights rounded to 1, 2, … bits, as EVIDENCE_LADDER
     already defines it), the task rule, and shuffled-network controls;
   - a network's understanding gain is how far its best explanations sit above the copy frontier at each length.
4. **Proof compression.** Turn the explanation into a verifier that bounds the network's behaviour, and report
   (bound tightness, verifier cost) against brute force. A copy cannot make checking cheaper; an explanation can.
5. **Transfer.** Give the explanation, and only the explanation, to a fresh reader: a language model that has not
   seen the network, or a person. Can it predict the network on new inputs and under new interventions, or rebuild
   a network that behaves the same? The control is the same reader given only the task's description (the
   prior-only control, moved to the reader's side): the explanation's worth is the gain over it. This is what makes
   the *words* testable, closing the gap that the DESCRIPTION field never was. The same test, run with people, is teachability: Schut et al.'s grandmaster test, or Kepler's
   "does the rule help a classical evaluator".
6. **Control.** Use the explanation to make a targeted change to the network: remove one ability, move one variable,
   switch one strategy off. Predict every effect of the edit before making it, including the side effects, and score
   the prediction. Understanding that cannot be used to edit precisely is incomplete. The project met the opposite
   case on Nov 5–7: editors that did not understand the network "succeeded" by breaking the cue ("+") instead of the
   skill (FINDINGS.md, Goal 1). This is also the measure behind Shreyas's Nov 7 vision of a model whose parts can be
   turned up, down or off with predictable results.

### 3.3 How it discovers: a residual-driven loop that invents its own vocabulary

```
        ┌─────────────────────────────────────────────────────────────────┐
        │                 THE GLASS MODEL (graph of claims)               │
        │   L1 variables · L2 operations · L3 algorithm · L4 knowledge    │
        │   residual ledger · guarantees · provenance · open questions    │
        └───────▲───────────────────────────────────────────────┬─────────┘
                │ verified claims                                │ largest structured residual
                │                                                ▼
   G  CONFIRM (frozen tests,         A  CARTOGRAPHY: what does it do, in which regimes?
      typed guarantees, proofs)      B  VARIABLES: what does the downstream computation read?
                ▲                    C  DEFINITIONS: what is each variable? (synthesise programs)
                │                    D  COMPOSITION: how do they make the algorithm? vs the textbook?
                │                    E  ABSTRACTION: what recurs? → new library primitives
                └──────── explore ── F  KNOWLEDGE: what can a person learn? (teach, test transfer)
```

- **A. Cartography.** Map the model's behaviour and internal states over its inputs. Find the regimes where it
  behaves differently, the competences it has, and where it fails. This replaces "the analyst picks a template".
  Measure each regime's noise tolerance (the units program's ε95: how much noise flips 5% of decisions; seconds to
  compute, no AI calls) to decide the route before any explanation is attempted: tolerant regimes can take
  approximate accounts; fragile ones need exact structure (UNITS_FINDINGS, "what it means").
- **B. Variables.** Find the network's own variables without a vocabulary:
  - decompositions of activations and weights (sparse dictionaries, transcoders, parameter decomposition);
  - the subspaces that later computation actually uses, found by substitution and interchange, not by variance;
  - what is invariant across training runs and sizes (the essential variables) versus what varies (incidental ones);
  - **simple stand-ins as hypothesis generators:** ladders of behavioural models (the month finding began this
    way: constant, circular distance, linear order, additive, full table), and interpretable twins trained to
    imitate the network (trees, sparse networks, short programs). A stand-in's variables are candidates only; the
    literature log's warning applies (weight-sparse circuits can be interpretable yet unfaithful), so every candidate
    is aligned with the original network before it counts.
- **C. Definitions.** For each variable, synthesise a program that computes it from the inputs or from earlier
  variables, in the current library:
  - structure recognisers for mathematical objects (the decompiler's group recogniser is the first; section 3.5);
  - symbolic regression and enumerative search over the library;
  - language-model proposals, each one checked by prediction on held-out inputs and by intervention.
  A variable with no short definition stays in the ledger as structured-but-unnamed, with its examples, until the
  library can express it.
  Where the domain has a formal language (boards, fields, numbers, symbols), a definition is a program. Where it does
  not (much of natural language), a definition is a description that has passed the simulation test: a fresh model,
  given only the description, predicts the variable on new text. Both kinds enter the library, marked as which kind
  they are.
- **D. Composition.** Find the modules first: group mechanisms by how they co-fire and which feed which (the step
  the survey lists as open for parameter decomposition: "grouping pieces into mechanisms", SYNTHESIS §9). Then
  assemble variables, operations and modules into the algorithm: dataflow, plus control flow by regime.
  Compare it with reference algorithms for the same task (the textbook method, a numerical scheme, a search
  procedure). The places where the network departs from the reference are flagged as candidate discoveries.
- **E. Abstraction.** Definitions that recur across variables, regimes or networks become new primitives, with a
  name, examples and counter-examples. The library grows, so later explanations get shorter, and the residual
  shrinks. This is DreamCoder's wake/abstract/dream cycle with networks as the tasks, and it is the operational form
  of Shreyas's Nov 18 hypothesis (basic units, shared across networks, composed into larger tools).
- **F. Knowledge.** State what the algorithm embodies in human terms, and test transfer (section 3.2, item 5).
- **G. Confirmation.** Everything above runs in explore mode. A claim enters the Glass Model only through frozen,
  pre-registered, held-out tests that assign its guarantee type.

The measurable signature of discovery is the **residual curve**: the structured residual falls round by round
because the library learns the network's concepts. Chess is the test case: the share of LD2's value that classical concepts
explain plateaus at about 79–80% however much data is added (kepler/FINDINGS §4). A box that discovers pushes the
explained share past that plateau, and every point gained is a named, verified, teachable concept.

**The loop's skeleton already exists in the repo, and it earned its place.**
- *What it was.* Helix Probe's experimenter, updater and theorist, with three rules: the model predicts every
  experiment before the box answers; a fix must explain the past better; a new theory must predict the future
  better, on trial.
- *What it did.* It made DeepSeek Flash (thinking off) score 0.85–0.88 on FIND's numeric functions, against 0.38 for
  the same model in FIND's own protocol (RESULTS_LOG R5; the advantage narrows to about +0.10 at a matched number of
  inputs).
- *What replaced it.* The constructor swapped that structure for one long conversation and a forced submission every
  six steps.
- *What the box does with it.* Brings the Helix skeleton back, pointed at the network's insides: experiments become
  stimuli and interventions, the model being revised is the Glass Model, and every intervention is predicted before
  it is run. The skeleton exists as code (`probe/`, about 1,200 lines, one adapter file per box), so the network's
  insides become one more kind of box.

### 3.4 Where language models reason inside the box

*(Shreyas, 23:32: use a language model in many more parts of the system, for reasoning and inference.)*

Today a language model appears in two places: the cheap reader that writes a whole construction in one leap, and
Kepler's scientist that proposed chess concepts as code (0 of 59 accepted over three rounds, the last at low effort
with a brevity instruction because long answers ran out of tokens; kepler/DIARY). Everything else that needed reasoning was done by
the Claude session by hand. The box should make that reasoning explicit and plentiful.
- **Roles run in series, one call at a time, each with its own system prompt** (Shreyas's definition of
  multi-agent).
- **Language models propose and argue; code decides.** Every output is a hypothesis to test, a program to run, or a
  prediction that gets scored.

| Role | What it reasons about | What checks it |
|---|---|---|
| **Cartographer** | Proposes regimes and competences from behaviour and state clusters; designs probe sets that cover them | Coverage and predictive gain of the proposed regimes |
| **Observer** | Looks at a found variable (its top examples, contrasts, minimal pairs) and says what it is | A simulation test: from the description alone, predict the variable on new inputs |
| **Theorist** | Proposes rival algorithms for a part (at least two, so the experimenter has something to separate) | The experimenter's discriminating tests |
| **Formaliser** | Turns words into executable library programs plus an alignment (the construction format) | The judge, at every level |
| **Experimenter** | Chooses the next stimulus or intervention that best separates the live hypotheses (Helix Probe's role, one level down) | Information gained, measured afterwards |
| **Critic** | Attacks each candidate claim: confounds, "decodable but unused", ceiling effects, a rival that fits equally well; writes must-fail cases *before* a measure is built | Its predicted failures are run; a critic that cries wolf is scored too |
| **Abstractor** | Reads many verified definitions and proposes shared primitives | Compression gain on held-out networks and variables |
| **Translator / teacher** | States L4 knowledge in human terms, with examples and counter-examples; writes teaching sets | Transfer tests on fresh models, then people |
| **Librarian** | Matches each claim against prior work (the 793-entry Aether log plus the 193-entry core); turns known mechanisms into hypotheses to test, not answers to assume | A novelty statement that a second pass must not overturn |
| **Prover** | Writes the verifier or proof sketch that turns an explanation into a bound | Soundness is checked by code; tightness and cost are measured |
| **Reader** | Reads weights directly, with tools, as a source of first hypotheses | Measured like every other hypothesis source; later trained with verified rewards (section 3.6) |
| **Planner** | Keeps the Glass Model, chooses what to work on next (largest structured residual, highest expected information) | The residual curve |
| **Informant** (language-model targets only) | The target model itself, asked about its own processing ("were you told to lie?", "what are you tracking here?"); introspection adapters are the trained version (core literature entry `arxiv-2604.16812`) | Its reports are hypotheses, never evidence: each one is checked against the internals. The liar sprint shows why: the follow-up answers read the instruction, not the lie |
| **Puzzle-maker** | Invents algorithms published nowhere and writes networks that implement them (weight-space programming), or designs training data with a planted rule, for the calibration ladder; seals the answer key | The networks must behave as specified; the key stays sealed until scoring. Kept strictly apart from every role that interprets |

Two consequences:
- **Capability matters per role.** The record says the cheap reader cannot do the theorist's or formaliser's work
  on dense networks, and the stronger-reader test was never run. The box should use the strongest available model
  where reasoning is the bottleneck (theorist, critic, formaliser, prover), and cheap models where volume is
  (observer drafts, simulation students). Which role needs which model is itself a measurement.
- **The language layer becomes verified.** Every sentence in the final Glass Model is attached to a test (simulation,
  transfer or alignment). The words stop being the untested part.

### 3.5 The hypothesis spaces it needs (where "provable" comes from)

The decompiler worked blind and exhaustively because it searched a hypothesis space with deep structure: the
representation theory of groups. It found nested discrete logarithms in a^b mod p because that structure let it ask
the right next question. The box needs many such spaces, each with a recogniser that finds the structure in
activations or weights and a verifier for it:
- **geometry:** lines, circles, tori, hierarchies and trees in representations (month line vs circle; Fourier
  circles);
- **algebra:** groups and their representations, products and factorisations (built);
- **factorisation into independent sub-computations:** the decompiler's factor mode finds, blind, when a network
  secretly computes several small independent things (repeated difference vectors expose each part's subspace; the
  parts' logit tables must add up to the network's answer). It found a^b mod 109 split into Z4 and Z27, the
  factorisation of p − 1 read from the weights (`lab/decomp/factor.py`; decomp/RESULTS). The idea is not specific to
  groups: any network whose representation is a sum of parts can be tested this way;
- **automata and programs:** finite-state machines and pointer structures (MIPS-style; the longest-cycle transformers'
  pointer chasing);
- **dynamical systems:** recurrences, rotations, fixed points, attractors (M4,3's 120° rotation);
- **numerical operators:** convolution kernels and Green's functions, multigrid hierarchies, spectral truncation,
  upwind schemes, closure terms (the fluid model's learned Biot–Savart kernel is the first instance);
- **search and planning:** look-ahead, candidate generation and pruning, quiescence (Leela's learned look-ahead,
  Jenner et al. 2024);
- **inference:** belief states, evidence accumulation, Bayesian updates (an LLM's representation of what is true
  versus what it asserts).

Plus **domain libraries that grow** through abstraction (3.3 E): board predicates and tactical motifs for chess,
operators for physics, relations and states for language.

### 3.6 What it learns from, so the hundredth network is cheaper than the first

- **A library of motifs with recognisers**, shared across networks, from the abstraction step. Recognisers must match
  by function and topology, not by wiring: the repo's own units review found that units recur at the level of
  function (dynamical motifs, fixed-point structure, the same abstract algorithm) while exact wiring mostly does not
  (Chughtai et al. 2023; Maheswaranathan et al. 2019; Driscoll et al.; summarised in `docs/glass/UNITS_PROGRAM.md`
  §2).
- **Trained readers.** A model trained with verifiable rewards to propose definitions and algorithms from weights and
  activations, on three sources:
  - compiled networks (Tracr-style) with known programs;
  - networks written by frontier models through weight-space programming, which can implement *new, unpublished*
    algorithms on demand, so they test discovery rather than recall (`main:docs/literature/aether/arxiv-2611.18114.md`;
    `arxiv-2611.21449.md`);
  - real networks whose explanations the box has already verified.
  This is DIRECTIONS K2, never built. It is also the practical answer to "how do LLMs read weights, and can I use
  that": measure the skill, study its mechanism (flagship 4, section 6), and train it.
- **The developmental and comparative record.** Checkpoints, seeds, sizes, base vs fine-tuned variants and training
  data, used as lenses: what forms when, what is invariant, what differs between a model that has an ability and one
  that does not, which data created a mechanism.

### 3.7 The substrate and the way of working

- **One adapter per target**, as Helix Probe intended: the network, its input distribution or environment, and
  optionally checkpoints, sibling models and data. Blindness stays as a validation mode (hide the task, relabel the
  symbols), not as a limit on the product.
- **Standard verbs** that every role and every study uses: run, record, decompose, patch, interchange, steer,
  ablate, attribute, synthesise, prove. Decompositions are trained once per model and reused by every question, so
  large models stay affordable.
- **Two modes.** Explore: cheap, wide, labelled exploratory, never a claim. Confirm: frozen, pre-registered, held out.
  The honesty rules of `CLAUDE.md` move to the boundary between them.
- **Known-answer ladders that test discovery, not recall.** The repo calibrated mostly on published mechanisms
  (Fourier addition, ARC's M4,3, textbook chess). The box needs networks whose true algorithm is known to the
  experimenters but published nowhere: planted, compiled or written by weight-space programming. Recovery rate on
  those is the box's discovery score.

### 3.8 One cycle of the loop, worked through on the chess residual (illustrative)

This shows how the roles and stages fit together on the case where the record stops. Everything after the first
bullet is a description of what the box *would* do, not a result. Kepler's K9 already tried a one-shot version of
the observer step on these very pairs (channel 22's minimal pairs: 0 of 20 programs accepted). What differs below is
the cycle around it: a simulation test, refinement with the failures shown, rivals separated by designed positions,
a frontier model in the reasoning roles, and a library that contains calculation and search primitives, which the
agent's own K9 conclusion says the residual needs.

- **Where the record stops.** On LD2, 61 classical concepts explain 78% of the value; quiet positions 81%, tactical
  positions 36%; channels 22 and 15 track the unexplained part, and resetting channel 22 removes 32% of the residual
  in quiet positions (kepler/FINDINGS §4).
- **Planner.** Picks the largest structured residual: the value the classical library misses in tactical positions.
- **Variables (B).** Decompose the residual stream over squares at the layers where the residual is formed. Keep only
  the components that later layers read (substitution on the value), and that recur in the replication network.
  Each kept component comes with its top positions and minimal pairs (the same position with one piece moved, the
  component on and off).
- **Observer.** Looks at one component's positions and writes: "a defender is tied to two attacked targets". The
  simulation test runs at once: from that sentence alone, a fresh model predicts the component on new positions. If
  it cannot, the sentence is rejected and the observer tries again with the failures shown.
- **Formaliser.** Writes the sentence as a board program in the chess library: `overloaded(sq) = defends(sq, a) and
  defends(sq, b) and attacked(a) and attacked(b)`. The program must predict the component on held-out positions.
- **Theorist and critic.** The theorist offers a rival: "it is just the count of attacked pieces". The critic adds
  the confounds: decodable-but-unused, a material-balance proxy, a tactic the classical library already holds. The
  experimenter builds positions that split the rivals (overloaded with few attacks; many attacks with no overload)
  and predicts the network's value on them before running it.
- **Composition (D).** The surviving definition joins the value's L3 program where the network reads it, with its
  effect on the value. The residual is re-measured.
- **Abstraction (E).** If "tied to two duties" recurs elsewhere (a pawn guarding two squares; a rook on two lines),
  the abstractor proposes a general primitive, `tied(piece, duties)`, and the library shortens every explanation that
  uses it.
- **Knowledge and confirmation (F, G).** A teaching set (positions, explanation, counter-examples) goes to a fresh
  model, and later to players, as a transfer test. The claim enters the Glass Model only through the frozen tests on
  fresh games, with its guarantee type.
- **The headline.** One point on the residual curve: the share of tactical-position value the library explains,
  before and after this concept.

### 3.9 How a person uses it

The person's way in is a question, asked in words, about the whole system: "why does it trade queens here?", "what
does it do with the scales it cannot resolve?", "what in the model makes this answer a lie?".
- **If the Glass Model already holds the answer,** the box gives it with the trace, the evidence and the guarantee.
- **If it does not,** the planner turns the question into a gap in the model and runs the loop on it. The answer
  comes back with the same evidence once it is confirmed.
- **Every answer separates** what is confirmed from what is exploratory.
- **The playground is where the person tests an answer themselves:** change the input, intervene on a variable, and
  watch the network and the Glass Model move together, or fail to.

### 3.10 What the world gets: the library is the public product

Shreyas's Nov 17 picture (stated in conversation, not in a repo file) was a public, playable portal of
constructions, grown over time, in the spirit of AlphaFold's database. The design above changes what that portal
holds.
- A catalog of per-network certificates helps whoever owns those networks.
- A **library of the concepts and mechanisms networks have learned** helps everyone. Each entry is defined as a
  program, with a recogniser anyone can run on their own network, examples, counter-examples, the networks it was
  found in, and its guarantees. It grows with every network decoded:
- a researcher points the recognisers at a new model and gets its known parts for free, so the work starts at the
  residual;
- a scientist reads the library to learn what networks have found in physics or in chess that people had not named;
- the library's growth curve is itself a scientific result about universality: how much of what networks compute is
  shared, and at which level.

Per-network Glass Models stay as the evidence behind each library entry.

---

## 4. What to keep, what to scrap

**Verdict: rebuild the centre, keep the instruments as parts.**
- **The current centre is close to the opposite of what discovery needs.** A reader writes a whole explanation in one
  leap and a judge accepts or rejects it: a language model *authors* the explanation and the network *confirms* it.
- **The box runs the other way.** The network's own structure proposes (variables found without a vocabulary,
  structures recognised), language models interpret and argue, and interventions confirm. The decompiler already
  runs this way, and it is the one instrument that worked blind.
- **Discovery needs many small hypotheses,** generated from the network's own structure, composed upward, and
  accumulated in a vocabulary that grows.
- **The measurement core, the controls and several instruments are sound,** and become parts of the new centre.

| Part | Verdict | Why |
|---|---|---|
| The judge's causal tests (substitution, interchange, partial/edit, necessity, controls beside every number) | **Keep, generalise** | Calibrated on known answers, never shown wrong (INTENT, "deliberately not changed"). Generalise from "the same decision" to "the same intermediate states", so it scores computations |
| Controls: shuffled, untrained, permuted, prior-only, the memoriser | **Keep** | They are why the results are believable. The memoriser test (A41) is the template for "honest about lookup" |
| The decompiler | **Keep, as the first structure recogniser** | Blind, exhaustive and causal: the standard every recogniser in 3.5 should meet |
| SPD and the atlas | **Keep, as one decomposition instrument and one honesty label** | Exact decomposition and the rule/lookup label are valuable. Its explanation layer (trees over a fixed vocabulary) is replaced by the definition step (3.3 C) |
| Kepler's law extraction, concept probes with controls, and teachability test | **Keep, wire inside-out** | Each law or strategy must get an implementation claim (Gap 7) |
| The change ledger, frozen measures, pre-registration | **Keep, at the confirm boundary** | Credibility. Exploration happens before the boundary, labelled |
| The Helix Probe loop (`probe/`) | **Keep, as the loop's skeleton** | Predict-first, experiment-choosing, keep-if-better rules made a weak model strong on FIND (RESULTS_LOG R5). The network's insides become one more box |
| Kepler's "hypotheses as executable code, judged by frozen tests" | **Keep, as the rule for every role** | It is the right contract between a language model and the evidence. What failed around it: 59 programs from a cheap model over three rounds (one with refinement, one on minimal pairs from channel 22) and 8 channel-derived rules, all rejected, with a library of static board features only (kepler/FINDINGS §4; kepler/DIARY). The missing parts were reasoning strength, a refine-against-failures cycle, and dynamic primitives |
| Catalog and playground | **Keep, as a view of the Glass Model** | The trace view (1b) grows out of the playground |
| The literature log (Aether, core survey) | **Keep, wire in** | As the Librarian role's store: known mechanisms become hypotheses; novelty is checked before claims |
| The reader/judge single-leap loop as the organising centre | **Scrap** | Succeeds only where the mechanism's kind is already familiar to the reader; on dense or unfamiliar networks it copies or recalls (E1, E4, E7, E10) |
| DeepSeek V4.1 Flash as the reasoning engine | **Scrap for reasoning roles** | Weak weight reader on the record; never compared with a stronger model (R3) |
| The size ladder | **Scrap** | Difficulty is structure, not size (Gap 6) |
| Literal-counting and copy rules | **Scrap** | Replaced by compression against a shared library and proof compression (3.2) |
| Per-study fixed vocabularies | **Scrap** | Replaced by a growing library (3.3 E) |
| Bespoke pipelines per sprint | **Scrap as a pattern** | Each sprint's useful code becomes a verb or a recogniser; the rest stays as its sprint's record |
| Templated LM tasks with analyst feature rows as the construction's input | **Replace** | The construction's input becomes the actual input (the text, the board, the frames), so the network's unnamed variables can appear (1b) |

**Why not scrap everything?** Because the record contains the hardest-won parts of any interpretability system:
measurements calibrated against known answers, controls that caught real errors (the shuffled control that exposed
disguised copies), an honest memorisation detector, and a blind decompiler. What failed is the organising idea, not
the parts.

### 4b. Alternatives considered for the new centre, and why the Glass Model

| Alternative centre | What it would be | Why not as the centre (and where it fits instead) |
|---|---|---|
| **A learned decompiler first** | Train one model on very many synthetic networks to map weights to programs, then apply it | Ground truth exists only for networks we can make, and real networks use mechanisms we do not know how to make, which is the point of discovery. A trained reader cannot propose what is absent from its training distribution. It fits as an amortiser that grows from verified explanations (3.6), the way DreamCoder's recognition model is trained on its own solved tasks |
| **Behavioural science only** | Treat networks as organisms: map behaviour, distil laws and formulas (Kepler's chess rules, planets' formula) | Answers *what* it knows, never *how*, and cannot tell a law from a heuristic that matches it (Vafa et al. 2025). It fits as cartography (stage A) and as the source of top-down hypotheses |
| **Interpretable by design** | Train transparent networks (sparse, modular, program-like) that match the black box, and study those | Changes the object: a twin can match outputs through a different mechanism (the literature log's "interpretable yet unfaithful" warning). It fits as a hypothesis generator whose variables are then aligned with the original (3.3 B) |
| **Verification first** | Start from formal methods and prove properties of the network | Proofs need a statement to prove, and the statement is the explanation, which has to be found first. It fits as the last rung of the guarantee ladder (Gap 3) |
| **A better single reader** | Keep the constructor, swap in a frontier model (the R3 test) | Worth running once as a measurement (Q7), and it would probably raise the ceiling. But one-leap authorship stays the wrong shape: the decisive month result needed behaviour models, rival tests and replications that no single submission contains |

The Glass Model design takes each alternative's strength as a stage, and puts at the centre the one thing none of
them has: an accumulating, verified model of the network, extended by a loop that invents vocabulary.

---

## 5. The research questions the next build must answer, with what would count

Each question is stated so that it has a yes/no answer and a decision, in the repo's own routine.

| # | Question | What counts as yes | If yes / if no |
|---|---|---|---|
| Q1 | Can the box find a network's own variables with no supplied vocabulary? | On networks with planted variables that are published nowhere (3.7), recovery of the planted variables well above the shuffled control; on real networks, variables that pass interchange at their sites | Yes: build B as the front of the loop. No: the bottleneck is decomposition quality; study why on the planted set |
| Q2 | Can it write definitions for variables it found (concept invention)? | Synthesised programs predict held-out values of the variable and pass intervention, on planted *novel* concepts | Yes: library learning is possible. No: the library's base language is too weak; enrich it (3.5) |
| Q3 | Does the library compress across networks? | Explanation length of the n-th network of a family falls as the library grows, at constant faithfulness; recognisers find learned motifs in fresh networks | Yes: universality becomes a capability. No: networks of that family don't share structure at this level; report it |
| Q4 | Can a composed algorithm be verified at intermediate states, not only at the answer? | Interchange at every step of the trace above controls, on known-answer networks, then on real ones | Yes: the trace view is a claim. No: find which steps fail and why |
| Q5 | Can typed guarantees go beyond finite tables? | Non-vacuous regional or compact bounds on some real class (for example ReLU regions of an RNN; one game phase) | Yes: "provable" extends. No: say which claims stay statistical, and how strong their sampling is |
| Q6 | Do explanations transfer? | A fresh model given only the explanation predicts the network better than given only the task; then people (teachability) | Yes: the words are true. No: the explanation is a description, not an understanding |
| Q7 | Which roles need which model? | The same tasks run with a frontier model and a cheap one, role by role | Decides where capability is the binding limit (the R3 question, finally asked) |
| Q8 | How does a language model read weights, and can the skill be trained? | A mechanistic account of an open reader model reading small networks, and a trained reader that beats its untrained self on held-out networks with verified rewards | Yes: reading becomes an instrument. No: the skill is local arithmetic plus priors (as the reading audits suggest); then rely on tools |
| Q9 | Does the residual curve bend? | On a flagship, each round of the loop adds verified concepts and the structured residual falls, beyond what the classical library reaches | This is the box's headline number |
| Q10 | Do alignments compose? If each link of the hierarchy holds within its measured tolerance, is the end-to-end error bounded by their combination, or do small errors compound (the "compounding structureless noise" that DIAGNOSIS_ABOVE_32 cites from Gross et al.)? | On known-answer networks, the end-to-end error of the composed Glass Model stays within a stated function of the per-link errors | Yes: a hierarchy's guarantee can be assembled from its links, which is what makes the trace view trustworthy. No: per-link certificates must be complemented by end-to-end tests at every level, and the ledger must track compounding |

---

## 6. Flagship targets, and what "decoded" would mean for each

Each flagship pulls specific capabilities. "Decoded" is defined in advance so that partial results are measured, not
argued.

### Flagship 1. A chess network's strategies (Leela t1, with LD2 and 744706 for replication)

- **Where the record leaves it:**
  - 61 classical concepts explain 78% of LD2's value; quiet positions 81%, tactical ones 36%;
  - two causal channels (22, 15) track the residual;
  - 59 concept programs from the cheap scientist, all rejected; the one accepted feature was a calculation
    (capture-resolution) feature, and the agent concluded the residual "is mostly not a simple static feature";
  - one counter-textbook behaviour rule (trade queens when lost against strong play) verified in play;
  - no rule shown to be an internal switch (kepler/FINDINGS §4–5).
- **Decoded means:**
  - an L3 algorithm whose trace on a position reproduces the network's value and policy through named concepts and
    an explicit look-ahead component;
  - the structured residual driven well below the classical library's 20–22%;
  - each new concept defined as a board program, aligned internally, and teachable (a fresh model and then human
    players predict the network's choices better with it).
- **Capabilities it pulls:** transcoder-scale decomposition of a transformer over squares; a chess library that grows
  (attacks, pins, overloads, forcing lines); search-procedure recognisers (look-ahead, Jenner et al.); attention
  query/key handled; teachability tests.
- **What a "strategy" is, for the box.** A human strategy is a plan: a goal and a method that unfold over several
  moves (a minority attack, a king walk). In the network it can live in three places, and the box must look in all
  three: the value function (what it considers good, and how those terms interact by phase), the internal
  representation of future positions (look-ahead), and the policy's recurring multi-move patterns in its own play.
  So the chess library needs temporal primitives (plans as programs over move sequences), not only predicates on one
  board. Kepler's verified rule (trade queens when lost against strong play) is a one-move tendency; the box's target
  is the plan it belongs to, and the place in the network where it is chosen.
- **The engine is a network plus search.** Kepler studied the raw network (no search); at play time lc0 searches,
  using the network's value and policy. Decode the network first, then how search amplifies its preferences (Kepler's
  own next step was play-outs with search, `docs/kepler/STATUS.md`).
- **Known-answer rungs:** McGrath's concepts must be found as variables without being supplied; Jenner's look-ahead
  must be found as a step of the algorithm.

### Flagship 2. How a fluid model is cheap

- **Where the record leaves it:** the law read with coefficients and regimes; a learned nonlocal Biot–Savart-like
  kernel; advection assembled in blocks 3–6; a resolution limit above wavenumber 8; velocity reconstructed from smoke
  (kepler/FINDINGS §1–3).
- **Decoded means:** the model's **numerical scheme** written as an algorithm: how it represents the field, how it
  approximates the nonlocal solve (kernel reach, any multi-scale structure), what it does with scales it cannot
  resolve (its learned closure, if any), and where its error comes from. That algorithm must reproduce the model's
  predictions, including its failures out of distribution, and be priced against a solver at matched accuracy.
- **Capabilities it pulls:** operator-level decomposition of convolutional blocks (kernels, Green's functions,
  multigrid and spectral recognisers); comparison with reference numerical schemes; known-answer models trained on
  simulators whose scheme is known.
- **The target has to be rebuilt first.** Kepler's model emulates its 64×64 pseudo-spectral solver at the same 64×64
  grid, over 4 solver substeps per frame (`kepler/targets/fluid/sim.py`; `scripts/kepler/fluid_build.py`,
  FRAME_DT 0.05 against dt 0.0125). It replaces four solver steps on the solver's own grid: there are no
  unresolved scales for it to model and no long step to integrate, so there is little for it to be cheap *about*.
  And the law it was trained on was sealed in the simulator, so the best possible finding was "it implements the
  law". To ask how a network does physics cheaply, train it where
  cheapness is forced:
  - on coarse-grained frames of a much finer simulation, so it must learn an effective closure for the scales it
    cannot see;
  - and with steps many times the solver's step, so it must learn a long-step integrator.
  That is how learned weather and climate emulators get cheap. The fine simulation is the ground truth for checking
  whatever the box finds.
- **Why this one can be a real discovery:** a learned closure for unresolved scales, and a learned long-step
  integrator, are scientific objects in their own right, comparable with known closure models, and nobody would
  write the network's version by hand.

### Flagship 3. What gives a language model the ability to lie

- **Where the record leaves it:** the standard black-box detector reads a lying *mode* set by the instruction, carried
  by chat-template tokens; a real-lie trace exists inside each model but does not transfer across families
  (liar/DISCOVERY). The sprint's planned mechanism step was never run, because it was gated on the detector passing
  its test first, and single mean-difference directions closed only 1–12% of the gap against a 50% bar
  (liar/FINDINGS, LT1 mechanism, LT2 M2–M3). So the lying mode is not one direction: the next step is the routing
  and gating question below, not another direction. The literature log adds that fine-tuned detectors fail across lie types, and transfer tracks
  surface similarity (`main:docs/literature/aether/alignment-anthropic-com-2026-lie-detectors.md`).
- **Decoded means:**
  - L1: the model's internal representation of what is true, and of what it is about to assert;
  - L2/L3: the computation that reads the first and produces a divergent second, how the lying mode gates it, and
    whether instructed, sycophantic and strategic lies share that machinery or only its output;
  - a fingerprint defined by the mechanism (belief and assertion diverge, through these parts), tested across lie
    types and model families.
- **Capabilities it pulls:** decompositions and attribution graphs at 7–8B; model diffing (honest vs deceptive
  fine-tunes; base vs instruction-tuned); developmental checkpoints where available; a library for beliefs, states and
  speech acts.
- **What a lie is, for the box, and the order of questions that follows from it.** A lie is an assertion that
  differs from what the model itself represents as true. That splits the work into questions the box can answer in
  order:
  1. **Belief.** Where does the model represent whether a statement is true? Validate causally: moving that variable
     must change how the model *uses* the fact afterwards, not only a probe's reading.
  2. **Assertion.** Where does it represent what it is about to say, before it says it?
  3. **The divergence.** A lie is belief ≠ assertion. That difference, located and measured, is the candidate
     fingerprint, defined by mechanism rather than by a classifier's training set.
  4. **The gate.** How does the lying mode the liar sprint found (carried by the chat-template tokens after the
     instruction) change the path from belief to assertion: suppress the true answer, promote another, or invert
     something? Follow the path with attribution graphs under the lie and truth conditions.
  5. **Generality.** Do instructed, sycophantic, strategic and trained-in lies use the same parts, or only produce
     the same output? Same parts would make one fingerprint; different parts make a family, and the box reports the
     family.
  6. **Origin.** Does the base model already have the machinery (people lie in its training text), and what does
     instruction tuning add? Model diffing and data attribution answer this.
  7. **Control.** Switch the gate off, predicting the side effects first (does fiction or role-play break too?).

### Flagship 4. How a language model reads weights

- **Where the record leaves it:**
  - the cheap reader, working through Python, verified a blinded game policy and named the game when asked a
    specific question (EXPERIMENT_LOG E12); Shreyas's Nov 13 view that its failures are interface failures is still
    untested;
  - reading audits say reading is local arithmetic plus dataset priors;
  - Singh: from weights alone, edit effects are predicted barely above a no-weights control (Spearman 0.17 vs 0.11);
    given the true hidden state, 0.97 (`docs/literature/wsp/K1_STATE_OF_ART.md`). The missing activation is the
    bottleneck, which is one more reason to interpret inference, not weights alone.
- **Decoded means:** a mechanistic account, inside an open reader model, of what it computes when it reads a small
  network's weights (which internal variables track "periodic in the input index", "copies the previous token",
  "this unit is a comparator"), and a trained reader that is measurably better on held-out networks.
- **What the literature log already says about how they read** (`docs/literature/wsp/`):
  - a careful audit of Claude Opus 5.5 reading 125k-weight CNNs found the arithmetic accurate but the semantics
    guessed: permuted labels still got the standard class names, KMNIST was read as MNIST, satellite images as
    CIFAR-10. In the audit's words, the semantic step is "where the model stops reading and starts guessing"
    (`arxiv-2611.08070`);
  - function-preserving permutations and rescalings drop readers' accuracy, while a probe that *executes* the network
    stays at 100% (`arxiv-2611.05798`);
  - on trained networks, what weights give a reader is mostly a layout map (which inputs connect where), not the
    computation (`main:docs/literature/aether/arxiv-2611.21449.md`).
  So today's "reading" is local arithmetic plus priors, and execution beats it. The box's version gives the reader
  activations and interventions, and scores it like any other hypothesis source.
- **What it gives the box:** the Reader role becomes an instrument with known strengths, and the result answers
  Shreyas's question directly: what LLMs are doing when they appear to read weights, and how to do it deliberately.

### The calibration ladder under all four

Networks whose algorithm is known to the experimenters and published nowhere. The box's **discovery score** is its
recovery rate on these, blind. Calibrating only on published mechanisms measures recall.

**Rungs, from exact ground truth to realistic ground truth:**

| Rung | How the truth is known | Why it tests discovery | Existing seed in the repo |
|---|---|---|---|
| Planted by construction | Hand-set weights implementing an invented algorithm | Exact truth; nothing to recall | The atlas's planted RNN (5 mechanisms per matrix) and planted transformer |
| Compiled from programs | RASP/Tracr-style programs, then rotated into a random basis or compressed so they do not look compiled | Exact truth at the program level; the basis hides it | The Tracr targets (`tracr_*`) |
| Written by weight-space programming | A frontier model writes a network that implements an algorithm invented for the test, with its design notes sealed | New algorithms on demand, in realistic sizes (`arxiv-2611.18114`, `arxiv-2611.21449`) | None yet |
| Trained with a planted rule | Train on data where a hidden, invented rule decides the label in some regime; the network must compute that rule somewhere | The network is a real trained network; the planted rule is a known part of its competence | The memoriser (`modrand_p113`) is the negative version |
| Rebuilt flagships with known truth | A coarse-grained fluid model with its fine simulation kept; a small board game whose winning plan was designed in | The flagship's own question, with an answer key | Kepler's fluid simulator; Leela ports for the real step |
| Recall controls | Published mechanisms under a secret relabelling of symbols (the decompiler's device) | Separates reading from remembering | Decompiler challenge sets |

**Scoring, written before the networks exist (the decompiler's practice: the commit times record it):**
- planted variables found among L1, with their alignment passing;
- definitions equivalent to the planted ones on held-out inputs;
- the algorithm's trace matching the planted steps;
- no invented structure on the must-fail controls (shuffled, untrained, the memoriser, and decoys where a tempting
  known representation is present but unused, as the month circle was);
- the cost of getting there.

### What "provably" can mean for each flagship, and what would raise it

| Target | Strongest guarantee available now | What would raise it |
|---|---|---|
| Group-law networks (decompiler) | **Exhaustive**: the program equals the network on every input of the finite table | A compact proof that the mechanism, not only the function, is what the network computes |
| Fluid model | **Probabilistic** over a stated distribution of flows, regime by regime, plus adversarial search for flows where the scheme and the network part | Bounds over classes of fields for the operator parts (kernels are linear, so their error can be bounded analytically) |
| Chess network | **Probabilistic** over positions from stated sources, plus adversarial search; **exhaustive** inside small sub-domains (endgames with few pieces, where tablebases give the truth too) | Regional proofs per position class; tablebase-checked sub-domains growing outward |
| Language models (lying, reading) | **Probabilistic** over stated prompt distributions, with adversarial search and must-fail controls | Regional bounds on the mechanism's parts (a gate's effect bounded over a family of inputs) |

The rule in every row: a claim states which guarantee it has, and never borrows the language of a stronger one.

---

## 7. Shoulders to stand on

Keys in `backticks` are entries in the repo's literature folders on `main` (`docs/literature/core/`, `wsp/` and
`aether/`; most of the core and wsp entries are also on this branch). The others are standard references.

| Pillar of the box | What to stand on | What it gives |
|---|---|---|
| Faithfulness under intervention (3.2 item 1) | Causal abstraction and interchange interventions (`arxiv-2106.02997`, `arxiv-2301.04709`); distributed alignment search (`arxiv-2303.02536`); causal scrubbing (Redwood 2022); hypothesis tests for circuits (`arxiv-2410.13032`) | The judge's theory, and how to learn alignment subspaces instead of fixing one affine map |
| Proof compression and typed guarantees (Gap 3, 3.2 item 4) | Compact proofs (`arxiv-2406.11779`); the verifier grade on group networks (`arxiv-2410.07476`); formal verification of ReLU networks over regions (interval bound propagation, SMT-based verifiers) | A grade that cannot be talked around, and proofs over regions |
| Compression against a shared library (3.2 item 3, 3.3 E) | DreamCoder (Ellis et al., PLDI 2021): wake/sleep library learning with an MDL objective; Stitch (Bowers et al., POPL 2023): fast compression into abstractions; LILO (Grand et al., arXiv 2310.19791): LLM-guided search plus Stitch compression plus LLM-written documentation that names each abstraction; ARC's surprise accounting (EVIDENCE_LADDER) | Concept invention as compression, with the library's cost amortised. LILO is the neurosymbolic template for the abstractor and translator roles: the language model finds concepts enumerative search cannot reach, compression consolidates them, and the language model names them |
| Variables without a vocabulary (3.3 B) | Sparse dictionaries, transcoders, cross-layer transcoders and attribution graphs (`tcpub-2025-attribution-graphs-methods`); parameter decomposition (SPD, `arXiv 2506.20790`; already in `lab/spd.py`); crosscoders for comparing models (`tcpub-2024-crosscoders`) | Decomposition at scale, and model diffing |
| Definitions (3.3 C) | MIPS (`arxiv-2402.05110`); symbolic distillation (Cranmer et al. 2020); sparse regression on dynamics (SINDy, PDE-FIND); the decompiler's structure recognition | Programs and laws from parts of a network |
| Concept discovery and transfer to people (3.2 item 5, flagship 1) | Schut et al. 2025 (AlphaZero concepts, teachability filter, grandmaster study); McGrath et al. 2022; Jenner et al. 2024 (learned look-ahead) | The teachability test, and the known answers for chess |
| Verifiable words (3.4) | Simulation scoring of explanations (OpenAI's neuron explainer, in SYNTHESIS §1); the 200-word rebuild test (`arxiv-2611.18114`) | Making the language layer testable |
| Learned readers (3.6) | Neural decompiling of Tracr transformers (`arxiv-2410.00061`); Tracr (`arxiv-2301.05062`); graph metanetworks (`arxiv-2312.04501`); weight-space programming (`rikiyat-github-io-blog-weight-space-programming`, `arxiv-2611.21449`) | Training a reader on ground truth, and generating ground truth with new algorithms |
| Developmental and data lenses (Gap 11) | Mechanistic data attribution (`arxiv-2601.21996`); influence functions for LLMs (Anthropic 2023); singular learning theory at scale (`arxiv-2611.00640`); stage-wise model diffing (`tcpub-2024-model-diffing`) | When and from which data a mechanism forms |
| Universality, at the right level (3.6) | Shared dynamical motifs across tasks (Driscoll et al.); universal fixed-point topology across thousands of RNNs (Maheswaranathan et al. 2019); weak universality in group networks (Chughtai et al. 2023); component reuse across tasks (Merullo et al. 2024); all in `docs/glass/UNITS_PROGRAM.md` §2 | Where to look for shared units (function and topology), and where not to (exact wiring) |
| Laws vs heuristics (Gap 7) | Vafa et al. 2025 ("What has a foundation model found?"); Kang et al. 2024 | The warning: a good predictor need not implement the law; test regime by regime |
| Agents (3.4) | MAIA (`arxiv-2404.14394`); the Pitfalls paper (`arxiv-2603.20101`) | What agent loops do well, and how their evaluations fail (memorised circuits, lenient judges) |

The literature log already contains most of what the box needs to stand on. What it lacks is a path from a paper's
lesson into the build: today the per-paper "Pattern" notes are written and filed. In the box, the Librarian role reads
them as hypotheses and tests.

---

## 8. Risks, and how each is handled without retreating from the goal

| Risk | How it is handled |
|---|---|
| Invented concepts are just-so stories | Every concept must predict held-out values, pass intervention, and transfer to a fresh reader. The critic writes must-fail cases first. Exploratory concepts never become claims without the confirm mode |
| Decompositions at scale leave dark matter (cross-layer transcoders: about 11–22% reconstruction error, SYNTHESIS §3) | The residual ledger accounts for it explicitly; exactness is used where a claim needs it (parameter decomposition); "complete" means accounted for |
| Several explanations pass the same tests (non-identifiability) | Rivals are generated on purpose (theorist, critic) and separated by designed experiments. When two survive, both are reported, with the tests that would separate them. Compression and proof compression rank survivors |
| Language-model roles game the verifiers, as the cheap reader did | Verifiers are frozen in confirm mode, and roles never see confirm data. Every role's output is scored, including the critic's |
| Cost and time (SPD took hours at 227k) | Decompositions are trained once per model and reused; iteration happens on the calibration ladder; large runs only for questions already shaped by small ones |
| Drift: the project re-founded itself about every two days | The flagships and the residual curve are fixed reference points. Each new phase extends the Glass Model and its library instead of starting a new instrument |
| Overclaiming "provable" | Each claim carries its guarantee type, stated plainly. Exhaustive and proven claims are separated from statistical ones on every page |
| The network is not modular or hierarchical at any level, so there is no middle to find | Then the rate–distortion curve says so: no explanation dominates both the copy and the task rule at any level. That is a measured fact about the network, reported as such, and the exact account (parameter decomposition with labels) is what the box delivers |
| The Glass Model of a large network is itself too large to read | It is a structured store, read through questions and traces (3.9), not end to end. Its size is measured (item 3 of 3.2), so "too large to read" becomes a number to improve |
| Fragile, dense networks resist short accounts (AlgZoo above 32 parameters) | They get an exact account (parameter decomposition) with rule and lookup labels. That is honest and complete in the ledger sense. The box does not force a short program where the network has none |

---

## 9. The order of work (research order, not an implementation plan)

The order is set by which answers the later steps depend on. Each phase ends in a measured answer to questions from
section 5, and each extends the same Glass Model and library instead of starting a new instrument. It is Shreyas's
own principle (stated in conversation, Nov 17) applied at the scale of the whole project: decide the task first,
then build only what serves it. Here the tasks are the flagships, and the capabilities are built because a flagship
needs them.

1. **Build the yardstick for discovery first: the calibration ladder (3.7; designed in section 6).** Networks whose algorithm is known to the
   experimenters and published nowhere: planted, compiled, written by weight-space programming, or trained with a
   planted shortcut. Include rebuilt flagship targets where the truth is known by construction (a coarse-grained fluid
   model with a known fine simulation; a chess-like game with a planted strategy). Without this, every later claim of
   discovery is a claim of recall. *Answers nothing yet; makes Q1, Q2 and Q4 measurable.*
2. **Run the loop on the ladder, role by role.** Cartography → variables → definitions → composition, with the
   language-model roles of 3.4, the Helix skeleton (predict every experiment before running it), and the two modes.
   Run each reasoning role with a frontier model and a cheap one. *Answers Q1, Q2, Q4, Q7.*
3. **One flagship end to end, with its residual curve.** The fluid flagship (rebuilt as in section 6) is the fastest
   to iterate on: a small convolutional model, a simulator that gives ground truth, and a crisp discovery target (the
   learned closure and integrator) that physics can check without outside experts. Chess is the most prepared: three
   ported networks, a concept library, a measured 78% baseline, two causal channels already found. Either can go
   first. The criterion is the residual curve bending beyond the reference library (classical chess concepts;
   known physics terms and closure models). *Answers Q9, then Q6.*
4. **Grow and test the library across networks.** Seeds and sizes of one family, then a second family. *Answers Q3.*
5. **Typed guarantees beyond finite tables** on the classes where the loop has produced algorithms (ReLU regions,
   game phases, flow regimes). *Answers Q5.*
6. **The language-model flagships** (lying; weight reading), once the decomposition layer reaches 7–8B models and the
   library has primitives for beliefs, states and speech acts. *Answers Q8 and flagship 3.*

Two things run through every phase and are never deferred: the residual ledger, and the honesty rules at the confirm
boundary.

---

## 10. The first five experiments, and what each would decide

Each one is small (minutes to hours), tests a premise the whole design rests on, and has a decision for each outcome.
Pass lines come from controls, set before the runs, as `CLAUDE.md` requires. None has been run.

| # | Experiment | The premise it tests | If it holds / if it fails |
|---|---|---|---|
| X1 | **The stronger-reader test, finally.** The same dense targets that stalled (M8,5, lc_6_5), with a frontier model in the reasoning roles and DeepSeek Flash as the control, same budget | That capability was a binding limit (R3, Q7) | Holds: put frontier models in the theorist, formaliser and critic roles from the start. Fails: the bottleneck is the method, not the model, which strengthens the case for the new centre |
| X2 | **A first discovery score for today's tools.** The puzzle-maker writes ten small networks implementing algorithms invented for the test, keys sealed; a Claude session with today's instruments tries to recover them blind | That a calibration ladder of unpublished algorithms can be made, and how far today's system is from Q1–Q2 | Holds: the ladder exists and has a baseline to beat. Fails: weight-space programming cannot produce realistic test networks; fall back to planted-rule training |
| X3 | **Does the rebuilt fluid model have something to be cheap about?** Train the video model on coarse-grained frames of a 4× finer simulation, with a longer step; compare with the plain coarse solver at matched cost | That a coarse emulator learns a closure worth decoding (flagship 2) | Holds: the closure is the discovery target. Fails: it learns no better than the coarse solver; raise the resolution gap or the step until it does, or choose a different physics target |
| X4 | **One concept-invention cycle on the chess residual** (section 3.8): the observer, the formaliser and the simulation test on the positions of channel 22 | That a definition can be synthesised for a variable nobody named, on a real network (Q2) | Holds: the residual curve gets its first point. Fails: find where it breaks (the observer's description, the program, the alignment) before scaling the loop |
| X5 | **The Helix skeleton on the insides of a network.** Point `probe/` at a planted RNN, with interventions as the probes and a predict-first model of the internals | That the predict-first loop transfers from black boxes to network internals | Holds: it becomes the loop's skeleton. Fails: keep the roles, and replace the loop's selection rules with ones built for internal claims |

---

## 11. Glossary (plain words, with a one-line example where it helps)

| Term | Meaning |
|---|---|
| **Site** | A named place inside a network where activations can be read or overwritten. Example: `resid8@p`, the residual stream entering layer 8 at position p |
| **Interchange test** | Copy a variable's value from input B's run into input A's run, and check that the network's answer changes the way the explanation predicts. Example: copy the "gap" variable of (1, 4) into the run on (5, 2); the max-network should now output 2 |
| **Substitution test** | Replace a site's activation with what the explanation predicts there, run the rest of the network, and check that the answers survive |
| **Copy frontier** | How faithful the network's own weights are when rounded to 1, 2, 3 … bits each: the best a "copy" can do at each length. An explanation must beat it |
| **Task rule** | The answer the task's own definition gives (for 2nd argmax, the true second-largest). The bar an account must clear to say anything about *this* network |
| **SPD** | Stochastic parameter decomposition: split each weight matrix exactly into many rank-1 pieces, and learn which pieces each input needs |
| **Rule vs lookup** | A mechanism's description is a *rule* if it still predicts the mechanism on input values never seen while fitting, and a *lookup* if it only describes the cases it saw |
| **ε95 (noise tolerance)** | The noise level, relative to a neuron's spread, that flips 5% of a network's decisions. Small means fragile: approximate accounts cannot be faithful |
| **Residual** | What the current explanation does not account for, measured as a share of the network's loss, decisions or internal state |
| **Residual ledger** | The list of every unexplained part, with where it is, how big it is, and whether it is memorised, noise, or structured but unnamed |
| **Glass Model** | This document's name for the box's output: a five-level, executable, aligned model of a network (network → variables → operations → algorithm → knowledge), with a residual ledger and a guarantee on each claim |
| **Library learning** | Growing a vocabulary by turning definitions that recur into new named primitives, so later explanations get shorter (DreamCoder, Stitch, LILO) |
| **Rate–distortion curve of understanding** | Faithfulness plotted against description length (in library terms) for a network's explanations, against the copy frontier and the task rule |
| **Proof compression** | How much an explanation lets a verifier check the network faster than brute force, at a stated bound |
| **Typed guarantee** | The kind of evidence behind a claim: exhaustive, regional proof, compact proof, probabilistic with sample size, or adversarial search with budget |
| **Marr's levels** | Three levels at which any information-processing system can be explained: what problem it solves and why (computational), with which representations and algorithm (algorithmic), and in what physical parts (implementation) |
| **Weight-space programming** | A language model writing or reading a network's weights directly, as text or through code |
| **Calibration ladder** | Networks whose true algorithm is known to the experimenters but published nowhere, used to score discovery rather than recall |
| **Explore / confirm modes** | Exploration is cheap, wide and never a claim; confirmation is frozen, pre-registered and held out, and is the only route to a claim |
