# The Hidden Methodological Layer: Uncovering Threats to Validity in LLM-Assisted Empirical Software Engineering

- **Key:** arxiv-2611.20033
- **Authors / organisation:** Victória Gomes, Rodrigo Spinola — Virginia Commonwealth University; Fabio Palomba — University of Salerno; Robert Feldt — Chalmers University of Technology; Thomas Zimmermann — University of California; David Lo — Singapore Management University
- **Date:** 2026-11-15
- **Link:** https://arxiv.org/abs/2611.20033
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

The authors start from a change in how empirical software engineering is actually done: researchers now hand
classification, screening, extraction, summarisation and even judging to language models, and the methodological
literature has been written for humans or for deterministic tools. Their concern is not that the models are bad at
these jobs but that the decisions they embody — the wording of a prompt, the model chosen, the number of
refinement turns, the reuse of a generated label as if it were a measurement — are rarely written down as
decisions at all. They call the resulting set of partially visible influences the *hidden methodological layer*
and set out to measure how it shows up in published work: whether its risks get reported, what the reported risks
are, and where they land in the classical validity scheme. The larger aim is a reporting standard: prompts, model
configurations, interaction protocols, validation procedures and downstream reuse should be part of the method
section, not implementation detail.

## What they studied

No model is trained or run. The object of analysis is a corpus of fifty primary studies published 2020–2025 in
twelve venues (ICSE, FSE, EASE, MSR, ESEM, CAIN, SANER, EMSE, IST, JSS, TSE, TOSEM), and the unit of analysis is
deliberately **the LLM-assisted task, not the paper** — sixty-nine tasks in total, coded earlier in a companion
review and re-used here. Method mix by task: mining repositories 34, experiments and case studies 21, secondary
studies 11, surveys 3. Task types: classification 17, screening 14, evaluation 11, synthetic data generation 8,
content transformation 5, pattern analysis 5, extraction 5. Model usages: sixty-two reported across the fifty
papers, ChatGPT 38, Llama 8, Claude 4, DeepSeek 4, with two papers not naming a model.

## How they did it

One new data item was extracted per task: the threats to validity the authors *themselves* raised in connection
with LLM use, taken verbatim so the coding stays traceable. Threats inferred by the reviewers from the study
design were excluded on principle — the paper is measuring reporting, not risk. Task-level coding matters because
a study can use a model for three jobs and discuss the risk of one; paper-level coding would credit it with having
addressed all three. Reporting was then tabulated against a four-way scheme for how the model participates:
augmentation (human still owns the task), automation (the model does most of it), decision support (the model
recommends, a human decides), and evaluation (the model judges). The thirty-three threat excerpts from the tasks
that reported any were thematically coded by one author and reviewed by a second across two refinement rounds
until consensus, producing lower-level themes and five higher-level ones. Finally each theme was mapped to the
validity dimensions appropriate to the *study type* — construct, internal, external and conclusion for
experiments and repository mining; search, selection, extraction and synthesis bias for secondary studies;
sampling, non-response, instrument and response bias for surveys — with one threat allowed more than one
dimension where the reported consequence justified it.

## What counts as an explanation here

Not a mechanism and not a model: a **taxonomy plus a layered conceptual picture**. The taxonomy has five
higher-level families — threats in constructing empirical evidence (classification errors, screening errors,
unreliable generated artifacts, extraction errors), threats from prompt and workflow contingency (prompt-dependent
interpretation, workflow instability), threats from model-based evaluation (evaluator bias), threats from
downstream propagation, and threats from model contingency and domain fit. The conceptual model is three stacked
layers: the hidden layer holding prompts, configurations, outputs, iterations and the model properties that make
them consequential (opacity, stochasticity, context dependence); the visible empirical workflow; and a validity
layer whose effects show up as cross-dimensional, cascading, context-sensitive or cumulative. Understanding, on
this account, is knowing where in the workflow a model-mediated decision sits and what it can contaminate from
there — explicitly partial, and offered as a lens rather than a prediction.

## How they judged success

Counts of reported versus unreported tasks overall and within each participation category, the thematic taxonomy
with code frequencies and the paper identifiers behind each theme, a threat-by-validity-dimension matrix with
cell counts and a per-threat study count, and — this is the part worth noting — the paper's own validity section,
which concedes the taxonomy and the cross-dimensional propagation claims are analytical synthesis rather than
controlled demonstration. Coding reliability is handled by two-coder consensus, not by an agreement statistic;
there is no scoring of threat severity, no attempt to check whether the threats reported were the *right* ones,
and no independent re-extraction.

## What they found

Forty-three and a half percent of the sixty-nine tasks (thirty) explicitly reported at least one LLM-related
threat; the remaining fifty-six and a half percent (thirty-nine) reported none. Reporting tracks the role the
model plays: automation 22 of 42 (52.4%), evaluation 5 of 13 (38.5%), augmentation 3 of 13 (23.1%), and the single
decision-support task reported nothing. The authors read the automation row both ways — most likely to report,
and also the place where twenty tasks carried substantial model autonomy with no validity discussion at all. The
thirty-three excerpts concentrate in evidence construction (19): classification and labelling 10, screening 4,
generated-artifact unreliability 4, extraction 1. Then prompt and workflow contingency 5 (prompt-dependent
interpretation 3, workflow instability 2), evaluation bias 4, downstream distortion 3, model dependence and
domain mismatch 2. Mapped to dimensions: construct validity takes 16 mappings, internal 8, conclusion 6, external
1, selection bias 1, extraction bias 1, with none at all in search bias or synthesis bias; twenty-eight
study-level reports sit behind the matrix. The paper's own reading of that skew is that reporting follows
*visibility*, not risk — misclassification is inspectable and gets written up, prompt assumptions and model drift
are not. It notes the same shape among the three survey studies, none of which reported an LLM-related threat
despite using models to generate persona responses and to thematically analyse participant text.

## Limits and open problems they name

Underreporting cannot be distinguished from non-existence: a task with no reported threat may have had none, or
may have had one nobody wrote down, and the corpus offers no way to separate the two. The corpus is fifty papers
from twelve venues, dominated by proprietary conversational models, and the task codes were inherited from the
companion review rather than re-derived here. Coding and mapping were consensus between two authors with no
quantified agreement. The cross-dimensional and cascading claims are described as theoretically grounded
relationships supported by the analysed evidence, explicitly not demonstrated causal mechanisms. And the
recommendation the paper closes on is itself an open question: it says a replication study that applies stronger
controls would be needed to find out whether any of these threats actually changes the empirical findings.

## What we take from it

Three transfers, in order. (1) **The threat table is a reporting instrument, so its unit has to be the task, not the
paper** — sixty-nine tasks from fifty papers is the whole reason the numbers mean anything, since a study that
flags the risk of one model use and not another reads as "addressed" at paper level. Our own write-ups have the
same structure: one methods section covering extraction, fitting, intervention and read-out, so the risk of each
stage gets a sentence or none. Code the risks per stage and the blanks become visible. (2) **Report rates measure
visibility, not exposure, and the paper's own data is the counter-argument to reading them as safety** — the
category with the most model autonomy is also the one where a fifth of tasks said nothing, while the category
where a human still holds the pen has the lowest rate of all, which is exactly the case where a model's influence
on interpretation is least likely to be noticed. Our probes that "no longer need checking because a human is in
the loop" fall in that cell. (3) **A threat that is only visible as a wrong output will be the only threat
reported** — ten of thirty-three excerpts are labelling errors and none is at search or synthesis, which is the
signature of an audit that stops where inspection is easy; the counterpart discipline is to ask of each stage
which of its risks would leave no trace in the output. The paper also supplies a compact map of the eight model
properties that make the layer exist at all (opacity, probabilistic outputs, input sensitivity, hidden
configuration, autonomous influence, training-data dependence, contextual variability, output plausibility),
which is a serviceable checklist for the conditions an interpretability claim inherits when the model is a
component of the measurement rather than its object.

## Tags

unit: whole-model
form: description
method: other
eval: benchmark
goal: evaluation-of-methods
