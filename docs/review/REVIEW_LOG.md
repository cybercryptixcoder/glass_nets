# Glass Networks: research review toward the next build — running log

*Started Wed Nov 25, 2026, 23:24:46 ET (system clock). Reviewer: Claude (Opus 5.5), in a cloud session.
Repo read: `cybercryptixcoder/glass_nets`, branch `liar` (the furthest line; `kronecker` is its sibling; `main` holds Aether's literature log).
Push access to the repo was refused by the session's permission system, so this log lives in the workspace and is sent to Shreyas as files.*

## The brief (Shreyas, 23:24)

- Not bugs, not efficiency. What the system **is not** and **needs to be** to become a "magic box": hand it any network, get it back fully interpretable, provably, at the level of what the whole network was trained to do (chess strategies, how a fluid net cheats physics, what lets an LLM lie, how LLMs read weights).
- Look at every run, every target, what worked and what didn't, how it was done, and how the Claude agent used the tool as a researcher.
- Scrapping and rebuilding from the ground up is a valid answer if justified.
- Do not stop at "a fundamental barrier makes interpretability impossible". Do not mistake pessimism for realism.
- Reason and read the literature more than run experiments.
- Document everything: what was done, found, learned, reasoned, decided.

## Method (decided 23:25)

1. Read the history and intent docs first (PROJECT_REVIEW, WHAT_WE_ARE_BUILDING, NORTH_STAR, DIRECTIONS, LEARNINGS).
2. Then each instrument's results, in build order: constructor (rounds 1–2, LM track), units program, atlas, decompiler, Kepler, planets, Kronecker, liar.
3. Then how the agent used the tool (diaries, the CHANGES ledger, the experiment log).
4. Then the literature log on `main` (SYNTHESIS.md first).
5. Write findings into this log as they appear; synthesize into `NEXT_BUILD.md` at the end.

*What actually happened (added 00:06):* steps 1–4 ran as planned, but the synthesis started early (23:32) and was revised continuously rather than written at the end, because drafting exposed what still needed reading. Every revision is a dated line in the Log. Two claims audits (00:00 and 00:05) checked 58 numbers and quotations against their source files. Experiments: none run, by design (the brief asked for reasoning and literature over runs); the only tool use beyond reading was one web search (library learning).

## Log (oldest first; times from the system clock)

- 23:24:46: started; told Shreyas the plan.
- 23:25: tried to attach the repo with push access to keep the log on a branch; refused by the permission classifier. Decision: keep the log here and send it as files.
- 23:25: scheduled two safety-net check-ins (23:46, 00:08 extension window).
- 23:27: read `docs/PROJECT_REVIEW.md` (Nov 18 self-review), `docs/glass/LEARNINGS.md` (E1–E12), `docs/WHAT_WE_ARE_BUILDING.md` §1–3, `docs/DIRECTIONS.md` §1, §2c, §2f.
- 23:27: read `docs/glass/UNITS_FINDINGS.md` (summary), `docs/glass/DIAGNOSIS_ABOVE_32.md`, `docs/glass/ATLAS_RESULTS.md`, `docs/kepler/FINDINGS.md`, `docs/liar/DISCOVERY.md`, `docs/planets/DISCOVERY.md`, the Kronecker part of `docs/SPRINTS_NOV25_REPORT.md` (branch `kronecker`), `main:docs/literature/SYNTHESIS.md` §1–3, the tail of `main:docs/literature/aether/NOTES.md`; checked which sprint code imports the Glass engine.
- 23:29: read `lab/constructions/gpt2_month_rollover/DISCOVERY.md`, `docs/EXPERIMENT_LOG.md` E28–E30 (how the month finding was made), `docs/glass/COMPARISON_PITFALLS.md`, `docs/kepler/RESEARCH.md`, `docs/literature/wsp/K1_STATE_OF_ART.md`, `docs/decomp/RESULTS.md`, `docs/glass/PLAN_SCALING.md`, the start of `docs/glass/DIARY.md`; grepped for any stronger-reader run.
- 23:32: Shreyas added: weigh *how* interpretability is done and *what* is interpreted and probed for (and how that framing shapes the work), not only model size; "the information lies in inference"; use language models in many more parts of the system. Treated as part of the original brief.
- 23:33: wrote `NEXT_BUILD.md` §1 (the whole record in one table, levels B/V/M/A/K and guarantees S/E/P) and §2 (15 gaps in four groups).
- 23:34: sent update 1.
- 23:35: checked `lab/lm.py` (an LM TASK's input is one row of integer features per prompt, chosen by the analyst) and `lab/unitmap.py` (fixed unit library); wrote §1b on framing: what each instrument interprets, the common question "is my Q at S and used?", its three ceilings, and the alternative: interpret the inference as a process (an execution trace of the network's own algorithm).
- 23:36–23:39: wrote `NEXT_BUILD.md` §3 (the Glass Model; five measures; the residual-driven loop; language-model roles; hypothesis spaces; learning across networks; substrate and modes), §4 (keep/scrap), §5 (nine research questions), §6 (four flagships and the calibration ladder), §7 (literature mapped to pillars; every key checked to exist on `main`), §8 (risks), §0 (one-page answer). Marked the illustrative rows in the §3.1 table as hypothetical, keeping only Kepler's real fluid findings as facts. Corrected the flagship-4 wording (Shreyas's interface view is untested; E12 is the evidence).
- 23:40: read `docs/RESULTS_LOG.md` R4–R5 (Helix Probe on FIND and ARC-AGI-3), `docs/JOURNAL.md` Nov 7–8, `docs/FINDINGS.md` Goal 2.
- 23:40: read `main:docs/literature/SYNTHESIS.md` §8–11 (shortcuts, closest work and the gap, the 'white box' definition, open questions).
- 23:42: checked how Kepler's fluid target was built (`kepler/targets/fluid/sim.py`, `model.py`, `scripts/kepler/fluid_build.py`): a 64×64 pseudo-spectral solver (dt 0.0125) generates the frames; the model predicts the next frame at the same 64×64 grid, 4 solver substeps per frame (FRAME_DT 0.05).
- 23:43: self-check of NEXT_BUILD §0–§1 against `lab/constructions/INDEX.md` found an error of mine: I had written that the automated reader "never produced a result above 32 parameters" (taken from PROJECT_REVIEW §1, which was written about the AlgZoo ladder). INDEX shows the blinded cheap reader produced a complete construction of modmul_p113 (226k params, 97.3% behaviour; prior-only 0 of 4 stages) and of GPT-2 be-verb (prior-only 1 of 5). Corrected in four places in NEXT_BUILD.
- 23:44: second pass on NEXT_BUILD: added to §1b "the framing is certification, not discovery" (evidence: INTENT's library-of-certificates goal; the judge changed in 22 commits in three days, PROJECT_REVIEW §7); grounded the Glass Model's levels in Marr's three levels; added behavioural-model ladders and interpretable twins as hypothesis generators (with the unfaithful-sparse-circuits warning); added module discovery as the first step of composition (SYNTHESIS §9 names grouping pieces into mechanisms as open); added §3.8, how a person uses the box (questions in words, answered from the Glass Model or by running the loop); added two risks (no modular middle exists; the Glass Model too large to read).
- 23:45: added a sixth measure to NEXT_BUILD §3.2, **control** (use the explanation to edit the network with every effect predicted first), grounded in Goal 1's cue-breaking editors and Shreyas's Nov 7 tweakable-model vision; updated §0. Added to flagship 1 what a "strategy" is for the box (value terms by phase, internal look-ahead, recurring multi-move patterns; plans as programs over move sequences).
- 23:45: read `docs/glass/HANDOFF.md`. Added to NEXT_BUILD §4b (alternatives considered for the new centre: learned decompiler first; behavioural science only; interpretable by design; verification first; a better single reader) with why each fits as a stage rather than the centre; added the ops-friction point to Gap 13.
- 23:46: read the WSP audits' findings (`arxiv-2611.08070`, `arxiv-2611.05798`) and added to flagship 4 what the record already says about how LLMs read weights (accurate arithmetic, guessed semantics; execution beats reading; trained-net weights give a layout map). Added a drawing of today's workflow to §1b.
- 23:47: fairness fix on the A35→A37 point (a proper new version judged on fresh networks, still person-added vocabulary); fixed a paraphrase I had put in quotation marks (month route), corrected the decompiler's description (it also reads MLP activations over the finite table); added §3.8, one cycle of the loop worked through on the chess residual (illustrative, labelled as such); measure 1 now scored in and out of distribution, with declared, restricted alignment maps.
- 23:47: one web search on library learning (results: LILO arXiv 2310.19791, Stitch, DreamCoder, MIPS). Added Stitch and LILO to §7's library-learning row: LILO's LLM-guided search, compression, and LLM auto-documentation that names abstractions is the template for the abstractor and translator roles.
- 23:47: added to the end of NEXT_BUILD §2 a traceability table: each of the 15 gaps, where the design answers it, and how that is measured.
- 23:48: expanded the calibration ladder in NEXT_BUILD §6 into a research design: six rungs (planted, compiled and basis-hidden, written by weight-space programming, trained with a planted rule, rebuilt flagships with known truth, recall controls under secret relabelling), each with how its truth is known and its seed in the repo, plus scoring written before the networks exist and must-fail controls including decoys (a tempting representation present but unused).
- 23:48: added NEXT_BUILD §3.10: the public product becomes the library of learned concepts and mechanisms (programs with recognisers, examples, provenance, guarantees), with per-network Glass Models as its evidence; connects to Shreyas's Nov 17 AlphaFold-style portal idea.
- 23:49: readability pass on NEXT_BUILD: found paragraphs of 5–9 lines with a script, broke the six longest into bullets (the trace view, certification stance, the Helix skeleton, roles intro, how a person uses it, the verdict).
- 23:49: checked `probe/` (the Helix loop: about 1,200 lines, env adapters in `envs/`) and noted in §3.3 that the network's insides can become one more box for it. Added to the end of §1b a table of eleven general lessons about interpretability the record has already taught, each with its evidence file.
- 23:50: added to flagship 1 that the engine is network plus search (decode the network, then how search amplifies it; Kepler STATUS's own next step).
- 23:50: read-through of NEXT_BUILD §3.1–3.3 and §5–6: fixed an inconsistent image (the residual curve "bending upward" while the residual falls), a long line, and softened the fluid claim to what the setup supports ("little to be cheap about": one forward pass replaces four solver steps on the same grid).
- 23:51: added to flagship 3 a definition of a lie for the box (an assertion that differs from what the model itself represents as true) and the order of questions it implies: belief, assertion, divergence (a mechanism-defined fingerprint), the gate (the sprint's template-token lying mode), generality across lie types, origin (base vs instruction-tuned; data), control.
- 23:51: brought §0 in line with the later sections (the library as the public product; the fluid target must be rebuilt; the research order).
- 23:51: the 23:46 safety-net check-in arrived (read via notifications at 04:51 UTC); the review turn was still running, so no action beyond this note. Consistency pass on §2 Gaps 5–13: fixed a wrong cross-reference (residual ledger is §3.1) and a long line.
- 23:52: added to NEXT_BUILD §6 a table of what "provably" can mean per flagship now (exhaustive for group laws; probabilistic plus adversarial search elsewhere; exhaustive inside few-piece chess endgames, where tablebases give the truth) and what would raise each.
- 23:52: consistency pass on §4 and §9: added two "keep" rows (the Helix Probe loop as the skeleton; Kepler's hypotheses-as-code contract as the rule for every role); made §9's criterion name the reference library per flagship.
- 23:52: added to §1b a target-level framing point: most targets were networks trained on functions people wrote down first (argmax, groups, Tracr, templated behaviours), so only the implementation could be new; knowledge discoveries need networks that know something people do not (chess, physics emulators trained on finer truth, mathematical data with unknown rules). Known-function networks become the calibration ladder; knowing networks are the discovery targets.
- 23:53: added a thirteenth language-model role, the puzzle-maker (invents unpublished algorithms and writes networks that implement them, or plants rules in training data, for the calibration ladder; sealed keys; kept apart from interpreting roles); updated §0.
- 23:53: added to §3.3 C that definitions are programs where the domain has a formal language, and simulation-verified descriptions where it does not (much of natural language), both entering the library marked by kind.
- 23:53: read `docs/PAPER_DRAFT.md` (the month paper: one behaviour, the "small operation" level) and `docs/WORKING_METHOD.md` (crystallise, plan finite steps, four docs updated per event). Added to Gap 13 that the Glass Model, as a store current by construction, can carry the "repo is the handoff" rule, with prose files as views.
- 23:53: added to §0 "why the box is buildable: the argument from the record": each stage of the box mapped to the place where it already works (decompiler, atlas, Kepler law and strategy, month, Helix on FIND, LILO, compact proofs). What is missing is their composition and the vocabulary loop.
- 23:54: added NEXT_BUILD §10, the first five experiments (X1 stronger reader; X2 a first discovery score via puzzle-maker networks; X3 does a coarse-grained fluid emulator have something to be cheap about; X4 one concept-invention cycle on chess channel 22; X5 the Helix skeleton on network internals), each with the premise it tests and a decision per outcome. None run.
- 23:54: read UNITS_FINDINGS' closing recommendations; added ε95 (noise tolerance, seconds to compute) to cartography as the router between approximate and exact routes per regime.
- 23:55: accuracy fix in §0 item 4: "dense 112-parameter ones were impossible" overstated it; the atlas did certify dense AlgZoo networks up to 1,376 parameters by exact decomposition. Now: they never got a short account and yielded only to an exact decomposition.
- 23:55: read §0 end to end: fixed two broken line wraps, made the month row precise (5/5 stages in GPT-2 small; line-not-circle replicated in three models), updated the status line.
- 23:56: added to Gap 4 an operational answer to the field's open unit question (SYNTHESIS §3, §9: seed-dependent dictionaries, clean native neurons vs a blamed basis): a variable is the network's own if it is read by later computation, stable across runs and decompositions, and compact.
- 23:56: added a "Reasoning" section to this log (R1–R9): how the findings led to each part of the design.
- 23:56: added a picture of the Glass Model's five levels (invented chess labels) under the §3.1 table.
- 23:57: added to Gap 10 that the attention query/key gap is on the critical path: query/key decide routing (which information moves where at inference); the month route and the liar sprint's template-token lying mode are routing stories.
- 23:57: added an "informant" role for language-model targets (the target model's own reports, and introspection adapters, as hypotheses checked against internals; the liar sprint as the cautionary case).
- 23:57: accuracy fix in Gap 2: the information-vs-copy-frontier measure had two versions; the first failed its pre-registered test, the revised T2 separated all 30+ saved copies (S ≥ 0.91, PLAN_SCALING) but stayed a signal, not a gate. The draft now says so, and frames shared-library and proof compression as generalisations of it.
- 23:58: marked the noise-tolerance lesson in §1b as the repo's own "candidate" result (13 networks plus 15 training runs), not an established one.
- 23:59: measure 1 now names output-appropriate scores (distributions, fields, values) instead of only the judge's "same largest output".
- 23:59: claims audit, part 1: grepped 29 key numbers and quotations in NEXT_BUILD against their source files (Kepler 78%/36%/0 of 24/32%/5 of 38/0.974/+0.066; atlas 5 of 6, 1,376, 187 of 193; decompiler 8/12, A5; units 5 of 5, 80×, 7%; month 97.4%; modmul 97.3%; Helix 0.85–0.88 vs 0.38; 22 judge commits; Singh 0.17/0.97; liar failed-lie test; Kronecker 465,183,748; FRAME_DT 0.05 and dt 0.0125; Goal 2's related-word job; Goal 1's 2 of 12; DIARY 2,166 lines). All 29 found in their sources.
- 00:00: fixed a stale cross-reference in §8 (how a person uses the box is §3.9 after renumbering).
- 00:00: added Q10 to NEXT_BUILD §5: do alignments compose (is end-to-end error bounded by the per-link tolerances, or do errors compound)? This is the question a hierarchical explanation's trustworthiness rests on.
- 00:01: read `docs/glass/UNITS_PROGRAM.md` §2 (the units literature review). Added to §3.6 and §7 that motif recognisers must match by function and topology, not wiring (Chughtai; Maheswaranathan; Driscoll; Merullo).
- 00:02: read `kepler/core/scientist.py`, a scientist prompt and answer (`results/kepler/chess/pairs/scientist/channel22_0.json`), and `docs/kepler/DIARY.md` around K9. Correction to my draft: the scientist wrote 59 programs over three rounds (0/24; 0/15 with refinement from round-1 scores; 0/20 from channel-22 minimal pairs, at low effort with a brevity instruction after a token-budget failure), not 24; it did get unit-derived contrast pairs. The agent's conclusion: "The residual is mostly not a simple static feature"; the one accepted feature was a calculation feature. Updated Gap 5, §1, §3.4, §4, §6 flagship 1, and §3.8 (now acknowledges K9 tried a one-shot observer step on the same pairs, and says what the cycle adds).
- 00:03: read `docs/liar/FINDINGS.md` verdict table: the mechanism step (LT1-mech) was not run because it was gated on K2, which failed; single-direction steering closed 1–12% of the gap (bar 50%). Added to flagship 3's record: the lying mode is not one direction, so the next step is routing and gating.
- 00:03: read `docs/glass/REPORT_ROUND2.md` corrections (the three conditions any replacement of the copy rules must meet). Added to Gap 2 that shared-library compression and proof compression meet (a) and (b) by construction and should meet (c), which must still be shown on the saved copies before either becomes a gate.
- 00:03: read `docs/decomp/PLAN.md` intent ("if yes: point it at networks whose algorithm is unknown"); added to Gap 15 that instruments stopped at validation and the project moved on (decompiler → Kepler).
- 00:04: verified the "last three sprints did not use the Glass code" claim for Kronecker too (branch `kronecker`, `scripts/kronecker/*.py` import only numpy, lightgbm and their own modules; planets imports spock, rebound, scipy).
- 00:05: claims audit, part 2: 29 more numbers and quotations from §2–§10 checked against sources (atlas A37/A40/A42, QK fixes, SPD times; decompiler S4 and nested logs; Kepler's resolution limit, blocks 3–6, swap 0.86, plateau, 144 situations, 61 concepts, 38 candidates; liar template tokens and 0.875/0.864; planets 1,736; units 23–82%; E7's 90.8%; modmul re-certification; the DIARY and COMPARISON_PITFALLS quotations; `lab/lm.py`'s feature-row definition; the WSP audit quotations; neural decompiling 73%). Three first showed as missing; all three were line wraps in the source, and the quotations are exact. Narrowed one attribution (R7–R8, not R6–R8). Sent the extension request at 00:04.
- 00:06: read `docs/kepler/RESULTS.md` (confirms: 59 programs over three rounds; the queen-trade strategy "verified by play, not by a mechanism inside the network"). Added the decompiler's factor mode to §3.5 as a general recogniser for independent sub-computations (not specific to groups).
- 00:06: §0 now lists five assumptions instead of four: added "networks trained on functions we already wrote down are where to look" (F29, F33).- 00:08: tied §9's order of work to Shreyas's own principle (stated in conversation, not in a repo file): decide the task first, then build only what serves it; flagships are the tasks.
- 00:09: the 00:08 extension-window check-in arrived; the request had already gone out at 00:04, so no action. Made the rate–distortion curve concrete in §3.2 item 3 (axes, references, the understanding gain over the copy frontier).
- 00:11: checked every table in NEXT_BUILD for consistent columns (none broken); added §11, a glossary of 19 terms in plain words (site, interchange, substitution, copy frontier, task rule, SPD, rule vs lookup, ε95, residual, residual ledger, Glass Model, library learning, rate–distortion curve, proof compression, typed guarantee, Marr's levels, weight-space programming, calibration ladder, explore/confirm).
- 00:12: made a patch file (`review-next-build.patch`) that adds both documents under `docs/review/` on top of branch `liar`, so Shreyas can commit them himself (`git apply`), since this session could not push. The local clone was left clean (staged only to make the diff, then reset; nothing committed).
- 00:13: §0 item 3 now carries its evidence (inference-time evidence behind every result; Spearman 0.17 from weights vs 0.97 from the hidden state).
- 00:17: final checks of quotations (13 quoted phrases matched to their sources); added a reading guide at the top of NEXT_BUILD; regenerated the patch; delivering the final versions of the first hour. No reply to the extension request yet.
- 00:49: Shreyas replied (he had not seen the extension request): another full hour is available if it adds value; respond first with whether I want it and what I would do; and commit and push everything to GitHub under his account (cybercryptixcoder, official.shreyas28@gmail.com).
- 00:50: push access attached this time (the earlier refusal was at 23:25); cloned to `/home/claude/glass_nets`; branch `review-next-build` cut from `liar` (60ebd80c); this log and NEXT_BUILD committed under `docs/review/` and pushed. The patch file is no longer needed.
- 00:52: while scoping the next hour, read `results/kepler/chess/calculation_features.json` and found my chess claim needed qualifying: three hand-written calculation features (capture resolution, threats, mate in the sequence) together add only 1.2 points (77.4% → 78.5%), so the missing vocabulary is not shallow calculation either. Corrected Gap 5, flagship 1 and §3.8: the untested candidate is the network's own deeper look-ahead, readable only from inside. Also found that the per-position chess data (values, activations) lives outside the repo (`KDATA`, e.g. `scripts/kepler/chess_dark.py` loads `value_split.npz`), so no free re-analysis of the residual is possible from the repo alone.
- 00:55: second run started (up to 1.5 hours; minimum 1 hour; extensions may be requested from 01:55 to 02:25). Shreyas's addition: the next version may be an agentic, many-iteration setup on a Flash-grade model, as Helix Probe and Helix Evolve were; or a toolkit handed to a strong agent; decide with evidence.
- 00:56: read memory on Helix Evolve; cloned `cybercryptixcoder/helix_evolve` read-only; read its README (results table, how it works), `docs/research/RESEARCH_CASE.md`, `docs/VERSIONS.md` open questions, `docs/LEAN_RESULTS.md` (experiments 1–4 and "so far"); in this repo read `docs/V3_JOB.md`, `versions/v1|v2/README.md`, `docs/handoff/03_ANALYSIS.md` §0 and §6.
- 00:58: Shreyas approved a separate fresh agent for the independent critique, 10-minute updates, and full freedom to rework from scratch.
- 00:59: wrote NEXT_BUILD §3.11 (the engine question, with the evidence table from all three Helix lines, the four conditions under which a cheap model at volume is strong, a sub-problem-by-sub-problem verdict, the three-layer answer, why interpretability suits Helix, and what would show it wrong); added Q11 and experiment X6; updated §0 and the keep table.
- between 00:59 and 01:02 (not clocked at the moment of launch): launched a separate, fresh agent for the independent critique (read-only, writes its report to `/home/claude/review/CRITIQUE.md`); it has not seen this review's reasoning.
- 01:02: wrote `docs/review/PREREG_FIRST_EXPERIMENTS.md`: X1–X6 in the repository's routine (build step served, question and decisions, frozen measurement, must-pass and must-fail cases, controls, runs, cost; pass lines from controls). Added a pointer entry to `docs/glass/DIARY.md`, as CLAUDE.md asks for every step to be documented there.
- 01:03: literature check on the four flagships and the volume engine (six web searches: neural closures and their interpretation; LLM-SR; lying mechanisms; Leela look-ahead; LLM-guided program evolution for interpretable surrogates; how LLMs read weights). Added to NEXT_BUILD §6 a table "where each flagship's open seat is", and a volume-engine row to §7.
- 01:04: added NEXT_BUILD §3.12: one run of the agentic box end to end (inner Helix volume loop with an exact scorer; middle director cycle with the residual curve; outer loop across versions with the discovery score), with a diagram, and why the inner loop optimising its score is safe here (scorer written and frozen first, must-fail controls inside it, claims only through confirm mode).
- 01:05: wrote `docs/review/BRIEF.md`, the two-page version (where it stands; the five assumptions; the box; the engine; keep/scrap; flagships and their open seats; the six experiments). The critique is still running; the brief and NEXT_BUILD will be revised by whatever survives it.
- 01:05: read Helix Evolve's VERSIONS.md sections on lean (stores, Scout), v9 (design map), v8 (register of hills) and v7 (side lines). Added to NEXT_BUILD §3.11 a table mapping each Helix Evolve part to its job in the box (two loops → inner engine; stores → confirmed claims and rival ledger; register of hills → rival explanations grouped by behaviour, a working answer to non-identifiability; design map → hypothesis-space map and library growth; side lines → keeping rivals alive; Scout → Librarian; golden version → the outer loop's rule).
- 01:06: added NEXT_BUILD §3.13, the flywheel: search → verify → distil into a trained reader → repeat (the AlphaZero pattern for interpretation), with interpretability self-play on the calibration ladder as an unlimited, verifiable curriculum (a puzzle-maker builds networks with hidden algorithms; an interpreter decodes them); why its signal comes from outside the reader's prior; the risk (puzzle networks unlike trained ones) and its guards. Added it as phase 7 of §9.
- 01:06: checked `lab/`, `kepler/core`, `kepler/targets/chess` and `lab/nets.py` for what the toolkit already has; added NEXT_BUILD §3.14, an inventory of each verb or part: what exists in the repo and what is missing (the biggest gaps: the Glass Model store, decompositions at Leela and 7–8B scale, recognisers beyond groups, one scorer library, unpublished-algorithm ladder networks).
- Messages to Shreyas, for the record: update 1 at 23:34; update 2 with both files at 23:44; update 3 with files at 23:52; update 4 with files at 00:01; the extension request at 00:04 (each time taken from the clock reading just before sending).

## Findings (numbered, added as they appear)

### From the history docs (23:27)

- **F1. The project has re-founded itself about every two days** (Helix Probe Nov 3–5 → Goal 1 weight-editing Nov 5–7 → Goal 2 map Nov 7 → survey Nov 11–12 → constructor Nov 13–14 → GPT-2 month Nov 17 → Glass rounds Nov 17–18 → units/atlas Nov 19 → decompiler Nov 21 → Kepler Nov 24 → planets/Kronecker/liar Nov 24–25). Each phase built a new instrument and a new judge rather than extending one. Evidence: PROJECT_REVIEW §2, §7.6.
- **F2. The one discovery that held up on a hard target (GPT-2 month line) was made by a strong agent (Claude) working by hand with the tools, not by the autonomous reader.** The cheap reader (DeepSeek V4.1 Flash) never explained anything above 32 parameters. *(Overstated: corrected by F30 at 23:43.)* PROJECT_REVIEW §1, §2, §6.
- **F3. The constructor's central unsolved problem was "explanation vs disguised copy".** Truth tests (behaviour, interchange, substitution) can't separate them, because a copy is perfectly true. Five literal-counting rules were each beaten by rewriting numbers (decimals, rescaled, integers over a scale, tables + indices). E1, E10, E11; PROJECT_REVIEW §6.
- **F4. The reader optimises whatever score it is shown** (Goal 2 Nov 7; Glass rounds: E4 curve-fitting the scoreboard with threshold tables). The system's design put an optimizer against a hand-written rule. PROJECT_REVIEW §7.
- **F5. The reader's default is "write the task's textbook algorithm, then patch toward the network"** (E7), and its success came only by compressing a working copy (E6). It never *derived* structure; it either recalled or copied.
- **F6. The original direction list (Nov 11) already named the reader-interface question (K1), a trained reader (K2), and 'what reading weights actually is' (R2).** None was built as a capability; they stayed as experiments. These are exactly Shreyas's "how do LLMs read weights" question.


### From the instrument results (23:27)

- **F7. Fragility was measured, and it changed the plan correctly but narrowly.** AlgZoo nets above 32 params flip 5% of decisions at ~1% noise; units can replace at most ~7% before the account falls below the task rule (UNITS_FINDINGS F2–F3, U17). Response: the atlas keeps exact weights and explains *when* rank-1 pieces fire. This solved "faithful" by giving up "compact program". It did not solve "explain the algorithm": the atlas output is a list of mechanisms with triggers, not a composed computation.
- **F8. Every instrument explains in a vocabulary the analyst supplies.** Atlas triggers are trees over task features (a, b, answer, ranks, time steps; then "angles k·v mod p" added in A37 after A35 failed: the answer's own vocabulary was added to pass). Kepler: 61 classical chess concepts; a 38-term physics library for the fluid law; 144 situations × 42 move types for chess rules. Decompiler: a library of structures (circles, groups, Smith normal form; closed forms k·t, k·t², k·log). **When the network's concept is not in the library, the system can only report "unexplained residual".** Chess: 78% explained by classical concepts; the 22% residual was characterised (where: sharp/tactical positions; which channels: 22, 15) but never named; 24 LLM-proposed concepts and 8 channel-derived rules all rejected. This is the single most important gap for "discovery".
- **F9. The atlas's explanations do not compose upward.** Its unit is a rank-1 piece of one matrix, its explanation is "when it fires" + "which output it moves". There is no level above mechanisms (no module, no subroutine, no algorithm, no strategy), and in RNNs many trusted triggers are time gates ("after step 2"). ATLAS_RESULTS 'Limits'. So the atlas gives a faithful parts list, not understanding of what the network was trained to do.
- **F10. The headline modular-arithmetic results recovered a published answer from a vocabulary built to contain it** (A37 angle vocabulary, A40/A42 two bases offered). The SHOWCASE says so. It is a strong validation of honesty machinery (rule vs lookup, memorizer flagged), not evidence the system can find an unknown representation.
- **F11. The decompiler is the closest thing to the 'magic box' standard (blind, no task name, outputs a verified program, recovers secret relabelings), and it works because it knows a mathematical theory of what to look for (representation theory of groups).** Its reach is exactly the reach of that theory: group laws and near-group laws (a^b mod p via nested discrete logs). Lesson: blind decompilation succeeds when the analyst has a *hypothesis space with structure* (here, group representations), not a flat feature list.
- **F12. Kepler's two kinds of success were both 'outside-in'.** Fluid: the law was read from the model's input→output dynamics by sparse regression over a term library (SINDy on the model as a simulator), plus probes and one velocity swap. Chess: the verified strategy (trade queens when lost vs strong opposition) came from mining the network's *choices* against human choices; the cause tests showed the rules 'are not a switch inside the network'. Neither result decoded how the network computes its answer. The question Shreyas asks ("how does the fluid net run so cheaply; what is it doing underneath") was answered as "it implements the true law, with a resolution limit", i.e., by confirming known physics, not by reading the network's own approximation scheme.
- **F13. The last three sprints (planets, Kronecker, liar) did not use the Glass engine at all.** `scripts/planets` (29 files), `scripts/liar` (20 files), Kronecker scripts: standalone pipelines; only Kepler reuses `lab/luminal.py`. Planets replaced the network with a fitted physics formula (distillation of behaviour); Kronecker compared ML accuracy against recursive known theorems (no network opened); liar tested a black-box detector and localised the 'lying mode' to chat-template tokens by attention knockout. **Each target became a bespoke project; the 'tool' was the agent plus the documentation discipline, not the code.**
- **F14. Pattern in the drift:** when reading internals got hard, the work moved to questions answerable from behaviour (is the black box needed? does a formula match it? does a detector read what it claims?). These are valuable and publishable, but they are evaluations *of* networks, not interpretations *of* them. The system has no capability that makes the inside-out route cheaper than the outside-in route, so a rational agent takes outside-in.
- **F15. The literature effort is huge (Aether: 793 full-paper entries on `main`; 193-entry core survey) but its notes are dominated by process bookkeeping (batch arithmetic, strip checks) and the per-paper 'Pattern' lessons are not wired into the build.** The core SYNTHESIS (Nov 12) already lists the field's hurdles (unit choice, decodable ≠ used, output-match ≠ mechanism, compensation, unstable evaluation, attention QK, dark matter, little uplift over black-box, scaling). The system has since run into hurdles 1, 3, 6, 7 and 9 head-on.

### How the tool was actually used (23:29)

- **F16. The stronger-reader experiment (R3) was never run.** Every automated reader run used DeepSeek V4.1 Flash, which the project's own literature file records as a weak weight reader (0/8 on Sparks shapes; 0–11% on EEG). Every result on a hard target that held up was produced by a Claude session agent acting as the researcher *(wording narrowed by F30 at 23:43)* (month rollover: E29 "made by the session agent with the tool's measurements and judge"; units, atlas, decompiler, Kepler, sprints: all agent-built). PLAN_SCALING; UNITS_FINDINGS D5; DIAGNOSIS R3. **So 'the system' in practice = a frontier agent + bespoke scripts + a docs discipline. The reader/judge loop the code is organised around was never the engine of discovery.**
- **F17. How the one hard-target discovery happened (month rollover):** (1) fit a ladder of simple *behavioural* models of the network's decision margin (constant, circular distance, linear order, additive, full table), (2) the winning behavioural model named a candidate variable (one scalar per month), (3) a hand-written construction placed that variable at depth 8 and the judge's interchange/partial/edit tests confirmed it, (4) a rival representation from the literature (the circle, Engels et al.) was causally tested and found unused, (5) replication in two more models, (6) steering along an unfitted direction. The method was **behaviour model → candidate variable → causal localisation → rival test → replication**, run by a strong agent. None of it is automated as a pipeline; only one piece (E31) became a reader built-in.
- **F18. Difficulty is not parameter count.** The 226k-parameter modmul net is cleanly explainable (fidelity 97.4%, 7/7 claims); 112-parameter AlgZoo nets are not (PLAN_SCALING findings 2–3). What matters is whether the network's computation has margin/structure (tolerance, symmetry) or is a dense precise tangle. The 'ladder of whole models by size' strategy (DIARY Background: "the path is a ladder of whole models") climbed the wrong axis and spent days on adversarially messy tiny nets.
- **F19. The project decided early that 'proven accurate isn't possible beyond tiny nets' and replaced proof with a measured certificate** (DIARY Background). Shreyas's brief asks for 'provably'. No layer of the system produces a proof object or a bound, even where one is cheap (finite input domains, which the decompiler verifies exhaustively but does not call a proof).
- **F20. The analysis tools the reader lacked were identified but not built** (R9: find the low-dimensional subspace the rest of the network uses, by substitution not variance; fit it against a library of generic features; hand candidates to the reader). R6–R8 showed the cheap reader 'can't find decision-relevant low-dimensional quantities alone'. This is the missing *candidate-variable generator*: the step the strong agent did by hand in the month finding.

### On framing (23:35, after Shreyas's addition)

- **F21. Every instrument asks the same kind of question: "is quantity Q (chosen by the analyst) located at site S, and is it used?"** Localisation plus necessity/sufficiency. Ceilings: the analyst's prior caps what can be found; the unit of evidence is a decision, not a computation; one templated behaviour at a time. In LM tasks the construction's input is literally the analyst's feature row (`lab/lm.py`: "inputs.npy (X: one row of integer features per prompt, the construction's input)"), so anything else the network uses is invisible by construction.
- **F22. The construction's DESCRIPTION (the human-language layer) is never tested** (COMPARISON_PITFALLS: "The description isn't tested by the judge"; PROJECT_REVIEW §3: "the description and the program can say different things"). The words a person reads are the one part with no verification.
- **F23. The one hard-target success (month) has the shape of an execution trace** (bind, reduce to a line, compare, decode, with locations). That suggests the product: a verified, human-readable trace of the network's own algorithm for any input, plus the program whose traces they are.

### Re-reading the Helix era (23:40)

- **F24. The project's strongest evidence about *method* comes from its first build and was then dropped.** Helix Probe (experimenter → updater → theorist; the model predicts every experiment before the box answers; fixes must explain the past, new theories must predict the future, on trial) with DeepSeek Flash, thinking off, scored 0.85–0.88 on FIND numeric and 0.92–0.97 on strings, against 0.38 and 0.40–0.46 for the same model in FIND's own protocol, and 0.20 / 0.01 for published GPT-4 (RESULTS_LOG R5; matched-budget advantage smaller, +0.10 numeric at the control's inputs). The constructor replaced that structure with one long conversation and a submit-every-6-steps rule, and the cheap reader then failed above 32 parameters. **Lesson: a predict-first, experiment-choosing loop with explicit model-selection rules made a weak model strong on black boxes; the box should bring that skeleton back, pointed at the network's insides.** (ARC-AGI-3: 0 of 35 levels, so the loop's limit is long-horizon stateful planning, not description.)
- **F25. Goal 2 found that the model is not organised by our behaviour labels.** Late layers do one shared "turn this word into a related word" job across translation, capitals, word forms and young-animal names; layers expected to handle arithmetic affected gender words (FINDINGS.md, Goal 2, discovery 2). That is direct evidence for §1b: interpreting one templated behaviour at a time cuts across the model's own operations. The box should find the model's shared operations first, and explain behaviours as uses of them.

### Weights versus inference (23:40)

- **F26. The project's founding frame is weights-first; its decisive evidence came from inference.** The survey's definition of a white box is "an executable model M, derived from N's weights" (SYNTHESIS §10), inherited from weight-space programming; the reader reads weights, the decompiler reads embedding geometry, SPD decomposes weights. But every result that held up rests on inference-time evidence: the month line (behaviour models, then interchange on activations), Kepler's fluid law (the model's implied dynamics), its velocity (probes and a swap), the atlas's triggers (importance over inputs), the liar sprint (probe answers, attention knockouts). The reading literature says the same: weights alone predict edit effects at Spearman 0.17 (0.11 with no weights), the hidden state at 0.97 (K1_STATE_OF_ART, Singh). **So: inference first for discovery; weights second, for generalising a found mechanism to all inputs and for proofs.** This is Shreyas's "the information lies in inference".
- **F27. The survey already named the core gap and the project answered it with the weakest component.** SYNTHESIS §9: causal abstraction's "hypotheses come from researchers, never derived from weights". The constructor's answer was the cheap reader, which recalled textbooks or copied. The hypothesis-derivation problem is still the unsolved centre (NEXT_BUILD Gap 4).
- **F28. The survey's strongest warning stands:** Anthropic's interference-weights work found a 1-layer model's weights not sparse as a program (about 2.4% of 331M virtual weights needed for 90% of helpfulness), and blamed the basis (SYNTHESIS §9). A whole-model sparse program at the weight level may not exist for real models; the explanation has to be hierarchical over a learned basis, with the residual accounted for.

### What the fluid target can and cannot answer (23:42)

- **F29. The fluid target cannot pose Shreyas's question.** The video model emulates the solver at the solver's own resolution (64×64) over 4 substeps, so there is nothing for it to be cheap *about*: no unresolved scales to model, no long step to integrate. Its law was sealed in the simulator, so the best possible outcome was "it implements the law" (which is what was found, with a resolution limit outside its training range). **To ask how a network does physics cheaply, the network must be trained where cheapness is forced: on coarse-grained frames of a much finer simulation (it must learn an effective closure for the scales it cannot see), and/or with steps many times the solver's step (it must learn a long-step integrator).** That is how learned weather and climate emulators get cheap, and it makes the network's closure and integrator the discovery targets, checkable against known closure models and the fine simulation. This is a framing failure in the sense Shreyas asked about: the target was built so that the answer was already inside it.

### Correction (23:43)

- **F30 (corrects F2/F16's wording).** The cheap reader succeeded on clean networks with familiar mechanism kinds: toys, Tracr programs, the hand-written Breakout policy, modmul_p113 (226k parameters, complete), GPT-2 be-verb (complete), AlgZoo ≤ 32 parameters. It failed on dense AlgZoo above 32, GPT-2 countdown and month rollover (1 of 5). Prior-only controls failed where it succeeded, so it did read those networks. **The pattern supports the vocabulary thesis (F8) rather than a size thesis: the reader succeeds when the network uses a kind of mechanism already in its prior (Fourier/discrete log, agreement, simple logic), and fails when it must find something it has no name for.**

### Second pass (23:44)

- **F31. The system's stance is certification, not discovery.** Its vocabulary (judge, certificate, catalog of certified entries) and its effort (most engineering on the measuring side) are those of quality assurance: decide whether to accept something produced elsewhere. The producing was left to a reader that could only recall or copy. Kepler is the one place the stance turned to discovery, and it gave the most interesting results.

### Operations (23:45)

- **F32. Plumbing shaped the science.** Colab had no GitHub credential, so results reached the repo only when an agent relayed them by hand; local jobs died when sessions paused; SPD runs took hours while agents waited (glass/HANDOFF; ATLAS §6; DIARY "waiting" entries). Small, fast questions were the affordable ones.

### Target-level framing (23:52)

- **F33. Most targets were networks trained on functions people had already written down** (AlgZoo tasks, modular arithmetic, groups, Tracr programs, templated GPT-2 behaviours). On them, only the implementation can be new (nested discrete logs is the best example). Knowledge discoveries need networks that learned something people do not know: from nature's data or from self-play beyond human level. The project reached such targets only in Kepler's chess work and (as an evaluation, not an interpretation) in planets and Kronecker.

### Routing (23:57)

- **F34. The atlas's open query/key limit is the routing half of inference.** Queries and keys decide which information moves to which position. Both hard LM results so far are routing stories (month: month1 → year token → final token; liar: the lying mode carried by template tokens). A box that interprets inference has to explain routing, so this limit is central, not a corner case.

### Kepler's scientist, read closely (00:02)

- **F35. The cheap scientist was given the network's own contrasts and still failed, and the agent diagnosed why.** *(Qualified at 00:52: shallow hand-written calculation features also close only 1.2 points, so the remaining candidate is the network's internal look-ahead, read from inside.)* 59 programs over three rounds (including refinement with scores and minimal pairs from causal channel 22) all failed the frozen tests; the last round ran at low effort with a brevity instruction because long answers exhausted the token budget. The agent concluded "the residual is mostly not a simple static feature", and the only accepted feature was a calculation feature (capture resolution). Lesson for the box: concept invention on chess needs dynamic primitives (calculation, search) in its library and strong reasoning in its definition roles; static board predicates proposed one shot at a time do not reach the residual.

### The engine question (00:59, second run)

- **F36. Helix Evolve shows what a Flash-grade model does at volume with an exact scorer.** Run C (v10.2, 2,564 programs) beat AlphaEvolve's first result, the best human construction (0.941362) and several published LLM systems on the second autocorrelation inequality, from a constant function, on a 2-core VM (`helix_evolve/README.md`). Its lean experiments: unguided mutations set 75–98% of gains; notes did not help on average; planner programs exploited a floating-point boundary in the checker until checking was made exact (`docs/LEAN_RESULTS.md`).
- **F37. Helix Probe's FIND advantage was mostly discipline, not reasoning.** The gap over the same model in FIND's protocol is "mostly behavioural" (the loop always experiments); at equal inputs +0.10; the theorist showed no measurable value; ARC-AGI-3 0 of 35 (no goal hypothesis) (`docs/handoff/03_ANALYSIS.md`).
- **F38. The rule: a cheap model at volume is strong exactly when the searched object is a small program, the score is exact and loophole-free, progress is incremental, and the volume is real.** Every failure in the record breaks one of those (Goal 2 and the copy rules: gameable scores; ARC: no goal signal; the constructor: one leap to a large artifact; Kepler's scientist: 59 programs, about one fortieth of Helix's volume).
- **F39. Definition synthesis for a variable found inside a network meets all four conditions,** so the chess-residual conclusion ("an LLM guessing does not find new chess knowledge") has not been tested at the volume where Helix works. Experiment X6 tests it.
- **F40. Helix's research question and Glass's vocabulary problem are the same problem.** Helix asks how a discovery system gets past its model's prior (`helix_evolve/docs/research/RESEARCH_CASE.md`); Glass's scientist proposed classical chess ideas because that is its prior. In interpretability the network's own variables are an exact signal from outside the prior, which makes interpretability an unusually good home for Helix's engine.

### Open seats (01:03, second run)

- **F41. Each flagship has a named open seat, and in three of four the seat is exactly the inside-out step the box adds.** Chess: Jenner et al. state as open how look-ahead is implemented, whether lines are compared, and how it combines with heuristics across positions. Fluid: closures trained as closures are well studied; reading the implicit closure of an end-to-end coarse emulator is not. Lying: a shared sycophancy/lying/instructed-lying head circuit is reported (arXiv 2604.19117); the gate, the no-instruction case and a transferable mechanism-defined fingerprint remain. Weight reading: no work opens the reader model; written networks are 99% zeros (arXiv 2611.16981), which confirms X2's readability check.
- **F42. LLM-guided program evolution for interpretability exists only against outputs** (LLM-SR for equations; LLMGEN for genomic surrogates; programmatic features; cognitive models). Scoring programs against a network's internal variable and requiring interchange where it lives is the inside-out version, and X6 tests it.

## Reasoning: how the findings led to the design (written 23:56)

- **R1. Who the engine is.** F2, F16, F30: everything that held up on hard targets was done by a strong agent by hand; the cheap reader succeeded only where the mechanism's kind was already in its prior. So the box must be designed around reasoning that is plentiful, explicit and verified (language-model roles in series, NEXT_BUILD §3.4), not around one author and one judge.
- **R2. Where discovery came from.** F8, F10, F11, F17: every success had a structured hypothesis space supplied from outside (Fourier, circle vs line, group theory, a physics library, a chess library); the decompiler shows that a *rich* structured space allows blind recovery. So the box needs many structured hypothesis spaces (§3.5) and a way to grow the vocabulary itself (library learning, §3.3 E). That is the difference between confirming and discovering.
- **R3. What understanding is made of.** F9, F12, F17, F23, F25: parts lists and behaviour laws are each one level; the one hard-target success had the shape of an execution trace through levels; models are organised by shared operations, not our labels. So the product is a hierarchy aligned level to level (Glass Model, §3.1), read as traces of the network's own algorithm (§1b).
- **R4. Which axis.** F18: difficulty tracked structure (tolerance, symmetry), not size. So route by measured tolerance (ε95) and come in top-down from competence as well as bottom-up from decompositions (Gap 6, §3.3 A).
- **R5. What makes an explanation not a copy.** F3 and the literature (compact proofs; MDL library learning): rules about the text of a program lose to disguises; measures that make copies expensive by construction do not (shared-library compression, proof compression; §3.2).
- **R6. The loop.** F24: the project's best method evidence is Helix Probe's predict-first, experiment-choosing loop. So reuse its skeleton on network internals (§3.3, X5).
- **R7. What to point it at.** F29, F33: most targets were functions people wrote down, and the fluid target could not pose the cheapness question. So calibrate on known-function networks (including unpublished planted algorithms) and aim discovery at networks that know more than we do, rebuilt where needed (§6).
- **R8. Shreyas's 23:32 addition.** F21, F22, F26: the framing ("is my Q at S?"), the untested words, weights-first vs inference. So §1b, the simulation/transfer test for every sentence, and "inference first for discovery, weights second for reach and proof".
- **R9. Why not scrap everything.** The judge's calibrated tests, the controls, the memoriser test, the decompiler and Kepler's law extraction each solved a hard problem honestly; what failed was the organising idea. So rebuild the centre, keep the parts (§4).

## Decisions

- **D1 (23:25).** Log and drafts live in `/home/claude/review/` and go to Shreyas as files, because push access was refused.
- **D2 (23:33).** Organise the diagnosis as gaps (missing capabilities or stances), never as bugs; every gap carries evidence, why it blocks, what is needed.
- **D3 (23:35).** Treat Shreyas's 23:32 addition (framing; inference; LLMs throughout) as part of the brief: a dedicated §1b, plus a role table in §3.4.
- **D4 (23:37).** Verdict: rebuild the centre (reader/judge single-leap loop, size ladder, fixed vocabularies, copy rules), keep the instruments (judge tests, controls, decompiler, SPD/atlas labels, Kepler's law extraction and teachability, ledger at the confirm boundary).
- **D5 (23:37).** Propose the Glass Model (hierarchical, executable, aligned, residual ledger, typed guarantees) as the product; the residual curve and the discovery score (blind recovery of unpublished planted algorithms) as the headline measures.
- **D6 (23:38).** Ground the explanation-vs-copy problem in two theories already in the repo's literature log: compression against a shared library (DreamCoder-style MDL) and proof compression (compact proofs). Reason: both make a copy expensive by construction, instead of by a rule about its text.

- **D7 (23:45).** Add control (predicted edits) as a sixth measure of understanding. Reason: the project's own Goal 1 showed that editing without understanding produces fake success, so precise, predicted editing is evidence of understanding that transfer and fidelity do not give.

- **D8 (23:53).** End the document with five small, decision-bearing first experiments rather than a build plan: Shreyas's routine is intent → measurement → small experiment, and these test the premises the design rests on (capability, ladder feasibility, fluid target, concept invention, loop transfer).

- **D9 (00:59).** Answer the engine question with three layers, built in order: toolkit first (every verb and score as code); a Helix-style volume engine on a Flash-grade model inside the sub-problems with exact scores (definitions, laws and closures, experiment design, abstraction); a strong agent as director for composition, framing and claims. Reason: F36–F40.
- **D10 (00:59).** Commits from now on carry no Claude co-author lines: Shreyas's stated rule for Helix work is to commit under his own name with no Claude attribution, and Glass Networks is a Helix spinoff. The branch's first two commits carry the lines; offer to rewrite them.

- **D11 (01:06).** Add the flywheel (search, verify, distil, self-play) as the last phase: it is the part whose ceiling does not depend on how many good researchers exist, which is what the brief asks for; it waits until the inner loop works.

## Next (as of 00:17)
- Final versions of the first hour delivered (NEXT_BUILD.md, REVIEW_LOG.md, review-next-build.patch).
- If the extension is granted: tighten NEXT_BUILD (remove repetition between §0, §1b, §2 and §3 while keeping every evidence pointer), read the decompiler's FACTOR.md and the rest of REPORT_ROUND2 in full, and re-send.
- Note on times: decision times D1–D8 are the nearest clock reading taken before each decision was written down.

## Sources read or cited (generated 00:12 from the backticked paths in this log and in NEXT_BUILD)

- `docs/DIRECTIONS.md`
- `docs/EXPERIMENT_LOG.md`
- `docs/FINDINGS.md`
- `docs/JOURNAL.md`
- `docs/PAPER_DRAFT.md`
- `docs/PROJECT_REVIEW.md`
- `docs/RESULTS_LOG.md`
- `docs/SPRINTS_NOV25_REPORT.md`
- `docs/WHAT_WE_ARE_BUILDING.md`
- `docs/WORKING_METHOD.md`
- `docs/decomp/PLAN.md`
- `docs/decomp/RESULTS.md`
- `docs/glass/ATLAS_RESULTS.md`
- `docs/glass/COMPARISON_PITFALLS.md`
- `docs/glass/DIAGNOSIS_ABOVE_32.md`
- `docs/glass/DIARY.md`
- `docs/glass/HANDOFF.md`
- `docs/glass/LEARNINGS.md`
- `docs/glass/PLAN_SCALING.md`
- `docs/glass/REPORT_ROUND2.md`
- `docs/glass/UNITS_FINDINGS.md`
- `docs/glass/UNITS_PROGRAM.md`
- `docs/kepler/DIARY.md`
- `docs/kepler/FINDINGS.md`
- `docs/kepler/RESEARCH.md`
- `docs/kepler/RESULTS.md`
- `docs/kepler/STATUS.md`
- `docs/liar/DISCOVERY.md`
- `docs/liar/FINDINGS.md`
- `docs/literature/core/`
- `docs/literature/wsp/`
- `docs/literature/wsp/K1_STATE_OF_ART.md`
- `docs/planets/DISCOVERY.md`
- `kepler/core/law.py`
- `kepler/core/probe.py`
- `kepler/core/scientist.py`
- `kepler/targets/fluid/sim.py`
- `lab/atlas.py`
- `lab/constructions/INDEX.md`
- `lab/constructions/gpt2_month_rollover/DISCOVERY.md`
- `lab/decomp/`
- `lab/decomp/factor.py`
- `lab/judge.py`
- `lab/lm.py`
- `lab/lm_tasks/`
- `lab/luminal.py`
- `lab/spd.py`
- `lab/unitmap.py`
- `main:docs/literature/SYNTHESIS.md`
- `main:docs/literature/aether/NOTES.md`
- `main:docs/literature/aether/alignment-anthropic-com-2026-lie-detectors.md`
- `main:docs/literature/aether/arxiv-2410.07476.md`
- `main:docs/literature/aether/arxiv-2601.21996.md`
- `main:docs/literature/aether/arxiv-2611.18114.md`
- `main:docs/literature/aether/arxiv-2611.21449.md`
- `results/kepler/chess/pairs/scientist/channel22_0.json`
- `scripts/kepler/fluid_build.py`
- `scripts/kronecker/*.py`
- `scripts/liar`
- `scripts/planets`
- Plus, read in full or in part without a path in backticks: `docs/PROJECT_REVIEW.md`, `docs/glass/LEARNINGS.md`, `docs/glass/DIAGNOSIS_ABOVE_32.md`, `docs/glass/PLAN_SCALING.md`, `docs/glass/ATLAS.md`, `docs/glass/ATLAS_RESULTS.md`, `docs/glass/INTENT.md`, `docs/glass/EVIDENCE_LADDER.md`, `docs/glass/UNITS_FINDINGS.md`, `docs/glass/UNITS_PROGRAM.md`, `docs/glass/COMPARISON_PITFALLS.md`, `docs/glass/HANDOFF.md`, `docs/glass/M43_VS_ARC.md`, `docs/glass/REPORT_ROUND2.md`, `docs/glass/DIARY.md` (opening and headings), `docs/decomp/DECOMPILER.md`, `docs/decomp/RESULTS.md`, `docs/decomp/PLAN.md`, `docs/kepler/INTENT.md`, `docs/kepler/STATUS.md`, `docs/kepler/FINDINGS.md`, `docs/kepler/RESEARCH.md`, `docs/kepler/RESULTS.md`, `docs/kepler/DIARY.md` (K9 section), `docs/planets/DISCOVERY.md`, `docs/planets/FINDINGS.md`, `docs/planets/STATUS.md`, `docs/liar/DISCOVERY.md`, `docs/liar/FINDINGS.md`, `docs/liar/STATUS.md`, `docs/SPRINTS_NOV25_REPORT.md` (branch `kronecker`), `docs/DIRECTIONS.md`, `docs/WHAT_WE_ARE_BUILDING.md`, `docs/EXPERIMENT_LOG.md`, `docs/RESULTS_LOG.md`, `docs/RESULTS_LM.md`, `docs/JOURNAL.md`, `docs/FINDINGS.md`, `docs/PAPER_DRAFT.md`, `docs/WORKING_METHOD.md`, `README.md`, `CLAUDE.md`, `lab/README.md`, `lab/constructions/INDEX.md`, the docstrings of every module in `lab/`, `lab/decomp/` and `kepler/core/`, and on `main`: `docs/literature/SYNTHESIS.md`, `docs/literature/aether/INDEX.md`, `docs/literature/aether/NOTES.md` (latest runs), and about ten Aether, core and wsp entries (the ones named in NEXT_BUILD §2, §6 and §7).
