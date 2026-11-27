# The finished system, in plain words

*The short version of `SPEC.md` (the full specification) and `GAP.md` (where we are and how to get there). Nov 26,
2026.*

## What it is

You give it a trained network, the behaviour you want explained, and some inputs. It gives you back a **blueprint**
of how the network produces that behaviour. A building's blueprint has one drawing of the whole and a specification
for every beam. This is the same thing for a network:
- a one-page overview of the network's strategy;
- the systems that carry the strategy out;
- every working part, each with a definition and proof that it is real;
- which parts feed which;
- a trace of any single decision;
- an honest count of what is explained and what is not.

## What it must be able to do

**Getting a network in**
1. **Take any network in our scope with just a plug-in.** You tell it how to run the network and what the inputs mean.
   You do not build a new pipeline for each network.

**What it tells you**

2. **Say what the network does,** including:
   - its mistakes;
   - what it knows beyond the textbook;
   - the shortcuts it takes compared with a standard method;
   - where it will break on unusual inputs.
3. **Show how it does it, from the inside.** Every part it names really exists in the network: change that part in
   the network, and the network changes exactly as the blueprint says it will. And the blueprint runs on its own:
   with all its definitions working together, it reproduces what the network does, step by step.
4. **Account for everything.** Nothing that matters is left out. Whatever is not explained is measured and pointed
   to.
5. **Say honestly what each part is:**
   - a rule;
   - a memorised table;
   - something unused;
   - something not yet understood.

   A test decides which, not the system's preference.
6. **Go from the big picture to the parts, and back.** It gives the strategy, then the systems, then the parts, and
   each level is explained by the one below.
7. **Trace any single decision:** which parts produced it, and the smallest change that would have changed it.
   **And fingerprint any behaviour,** such as a lie: the parts whose activity gives it away.
8. **Name things nobody has named.** When the network uses an idea we have no word for, the system invents the word
   and the definition, and proves it.
9. **Explain where the network's efficiency comes from.** When the network does a job a known method also does (a
   fluid model and the solver it imitates), say which steps it skips or approximates, and what that costs.

**How it guarantees what it says**

10. **Never be fooled.** A disguised copy of the network, or a fitted imitation, never counts as an explanation. A
    network with no structure gets "no structure found", not a story.
11. **Be usable by a person.** Someone holding the blueprint, or an AI that is not allowed to run code, predicts the
    network better than without it. If the network knows something people don't, the blueprint can teach it.
12. **Let you change the network on purpose:** switch a part off, or strengthen it, and know the effects beforehand.
13. **Say how sure it is, and be right about that.**

**Practical**

14. **Work at real sizes in all our domains:**
    - small algorithmic networks;
    - chess engines;
    - learned physics simulators;
    - small language models, for named behaviours;
    - 7–8B language models, with how far it got reported. That is where the lying and weight-reading questions sit,
      so those two get measured, partial accounts in this build.
15. **Do the research itself,** from start to finish, without a person choosing each step.

## What "fully interpretable" means

A network is fully interpretable for a behaviour when its blueprint meets five conditions:
- **Complete:** nothing that matters is missing, and the blueprint runs on its own.
- **Honest about types:** every part is labelled rule, table, unused or not yet understood, by a test.
- **True at every level:** every named part checks out inside the network.
- **Navigable:** short at the top, every detail below it.
- **Usable by a person.**

It does **not** have to be short. Real networks often run on hundreds of small rules of thumb rather than one neat
algorithm. A complete blueprint of such a network is long, but every entry in it is simple and checked, and the top
page still fits on a page.

"Provably" means every claim says what kind of evidence backs it, and the system's confidence has been checked on
networks where the answer is known.

## What a blueprint could look like

*This example is invented to show the shape. It is not a finding.*

- **Page 1:** "This chess network picks moves by:
  1. following forcing lines (checks and captures) three moves ahead;
  2. scoring the material each line wins;
  3. adjusting for king safety and pawn structure;
  4. recalling about 2,000 memorised opening patterns.

  This page alone predicts 85% of its choices."
- **The look-ahead system:** six parts. Part 31 is "the square my opponent's best reply lands on", defined from parts
  12 and 17. It checks out in 98% of intervention tests.
- **The trace for one move:** "It played Nxe5 because parts 31 and 44 fired. Switch off part 31 and it plays Bd3
  instead."
- **The ledger:** 91% rules, 6% memorised table, 3% not yet understood (located in block 7).

## Where we are

- **Strong:** checking whether a claimed part is really in the network.
- **Partly built:**
  - one core for every domain;
  - "what it does";
  - honest types;
  - copy-proofing;
  - stated confidence;
  - real sizes.
- **Never built:**
  - complete accounts of real networks, with the blueprint running on its own;
  - big-picture-to-parts;
  - tracing single decisions;
  - inventing new concepts;
  - usability by people;
  - control through edits;
  - explaining where efficiency comes from;
  - doing the research unaided.
- **What broke last time** is big-picture-to-parts (item 6) and inventing new concepts (item 8).
  - Rosetta tried to explain the chess network's verdicts ("we are winning") directly from board features, and could
    define 2 of the 16 that matter.
  - It never went backward through the layers to the ingredients.
  - It had no way to invent a new kind of ingredient. The one new kind it was given by hand (material won along the
    network's planned line) carried a sixth to a third of what the network knows beyond the textbook, though no single
    part of the network holds it: a pointer to the missing kind of ingredient, not a finished concept.

## What the next build must deliver

1. **The test bench:** sealed test networks with known answers. These include ones built around ideas outside the
   system's vocabulary (half of them ideas with no human name), ones that compute verdicts in stages, and ones that
   mix rules with memorised tables. Also disguised copies that must fail. This is the finish line.
   - It is sealed as a generator. The build practises on some networks; the final exam is a fresh set, scored once.
   - The pass marks are fixed before the build starts.
2. **The checker:** every test in one place.
3. **The map-maker:** finds the network's parts at every layer, and which parts feed which.
4. **The definer:** defines each part from the parts that feed it, using operations ("play out the exchange") as well
   as board features.
5. **The inventor:** adds new kinds of idea when the vocabulary runs out.
6. **The assembler:** builds the blueprint, the ledger and the traces.
7. **The reader interface:** makes the blueprint usable, and tests that people can use it.
8. **The editor:** changes the network through the blueprint, with predictions.
9. **Plug-ins** for each domain: chess, the rebuilt fluid model, GPT-2, and 7–8B models.
10. **The director:** runs the whole loop without help.

## The order

1. **The finish line:** the test bench and the checker. If the checker can't tell true answers from copies, stop and
   fix that first.
2. **The make-or-break:** the map-maker, the definer and the inventor, tested on the hardest planted networks. This is
   where the last build broke and the only genuinely uncertain part, so it goes second. If it fails, it fails at a
   named part.
3. **Complete blueprints** on every test network.
4. **Real networks with known answers:**
   - chess look-ahead;
   - the capture idea Rosetta found (it passed its checks at the margin);
   - GPT-2's month rollover;
   - modular multiplication.
5. **People and edits.**
6. **No hands:** the whole bench runs unaided.

Then the real questions: chess strategy, the fluid model, lying, and weight reading.

## What you need before starting

- **A strong AI model that can actually see a chess board.** The only one we have could barely use board rules at
  all. This is the most important missing piece.
- **GPU time.**
- **A budget per step.**
- **A separate session to write the sealed answers.**
- **Days, not hours.**

## The honest answer

**The three fixes alone do not get you there.** Those fixes are:
- going backward through the layers;
- a language of operations;
- an inventor that can see.

They repair what broke. "Fully interpretable" also needs the parts that were never built: complete accounts,
traces, usability, edits, and running unaided.

**Build all fifteen, and pass every gate on the fresh final exam at the pass marks fixed in advance, and yes, you
have the system,** for the networks in our scope: small algorithmic networks, chess engines, the rebuilt fluid model,
and GPT-2 once it passes its known answers. "Fully interpretable" is then defined, tested, and true of every
blueprint it gives you. Where it cannot finish, it says exactly where and how much is left.

**Step 2 of the build answers, early, whether this approach works at all.** It is the only genuinely uncertain step.
The later steps use methods with precedent in the field, but some have never been shown to this standard, so they can
fail too. If one does, it fails at a named part.

**What no build can promise is what you'll find.**
- A network that mostly memorises gets an honest blueprint that is mostly tables.
- Whether the chess engine knows a strategy no human has named is a fact about the engine. The system finds it if
  it's there.
- On 7–8B language models it may not finish within budget, and it will tell you how far it got.

**One risk remains even after the bench passes.** Real networks could hold structure that no test network
anticipated. The real networks with known answers, and the rehearsal network built for each question, shrink that
risk before the unknown targets are run. They cannot remove it.

**Why this attempt can't end up open-ended:**
- the finish line is built first and sealed;
- the pass marks are fixed before the build, so a partial pass counts as a fail;
- the uncertain part is tested second;
- every part has its own test.

It ends in one of two ways: everything passes, or one named part fails one named test, and you know exactly which
member gave way.
