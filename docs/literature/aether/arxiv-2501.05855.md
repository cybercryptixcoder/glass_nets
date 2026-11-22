# ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability

- **Key:** arxiv-2501.05855
- **Authors / organisation:** Antonin Poché, Alon Jacovi, Agustin Martin Picard, Victor Boutin, Fanny Jourdan; IRT Saint Exupéry, IRIT (Université Paul-Sabatier), Google Research, CerCo/CNRS and ANITI (Université de Toulouse). Six authors are listed in the abs-page author string but only five bylines with printed affiliations appear in the HTML, so the sixth's affiliation is not claimed here.
- **Date:** 2025-01 (v1 2025-01-10; v4 2025-06-04, the HTML read)
- **Link:** https://arxiv.org/abs/2501.05855
- **Kind:** paper (ACL 2025)
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Concept-based explanations are scored almost entirely inside the concept space — how faithful the concepts are to the model's embeddings, how sparse or concise they are — and hardly ever on whether a reader given those concepts can actually anticipate what the model will do. The authors argue this is the wrong end of the problem: an explanation's job is to make the system's behaviour predictable to someone else, so the right measurement is *simulatability*, the accuracy with which a third party reproduces the explained model's answers from the explanation alone. Honest simulatability studies need many human participants per comparison, which is unaffordable for a benchmark spanning dozens of setups, so they propose substituting a language model for the human and then auditing that substitution. The larger goal is a scalable, method-agnostic benchmark for concept-XAI that prices the concept space, the concept interpretation and the concept-importance score together.

## What they studied

Classification models split at the penultimate layer into a representation part and a classifier head: DistilBERT, a version of DistilBERT with a ReLU inserted before the studied latent (so that non-negative factorisations apply), T5, a ReLU version of T5, and instruction-prompted Llama-3-8B (five models in total, with the unmodified Llama and the unmodified encoders using four of the five extraction methods). Four text-classification datasets: a 10-class subset of BIOS, IMDB, Rotten Tomatoes and the emotion subset of TweetEval. The unit of analysis is a *concept* — a coordinate of a learned linear projection of the embedding vector, i.e. a direction in latent space — together with a scalar importance for each concept per sample and per class. The extraction methods compared are ICA, NMF, PCA, SAE and SVD; the interpretation methods are maximally activating words (with the least-activating words added when negative activations exist) and alignment of each concept to labels proposed by GPT o1 with representative sentences. Simulators are GPT-4o-mini, Gemini-1.5 Flash and Gemini-1.5 Pro.

## How they did it

Every concept explanation is factored into three parts they formalise: a projection from embedding space to a concept space plus a decoder back, an importance score per concept (gradient-times-input on the concept-to-output half), and an interpretation (the words or label shown to the reader). They then construct an artificial reader. In the initial phase the simulator is told the task and, optionally, a global explanation — class-wise concept importances plus the interpretations of the important concepts. In the learning phase it is shown a sample of examples *with* the model's predictions and, optionally, local explanations. In the evaluation phase it gets fresh examples with no predictions and no explanations, and must guess the model's labels; its agreement with the model is the simulatability score. Explanations are withheld from the evaluation phase because presenting them there would leak the answer. Concepts with negligible normalised importance are dropped from the prompt, and the remaining importances are shown only as four coarse buckets rather than numbers. Because a score is only comparable within one setting (dataset × model × seed × extraction × interpretation × prompt type × simulator), they turn each setting into a vote, rank methods by pairwise wins under Copeland's ranked-choice rule, and test the mean pairwise accuracy differences with a t-test at the 0.05 level. The main GPT-4o-mini sweep covers 23,360 settings over 7 seeds and 2,000-token prompts, at an API cost they report as about seven dollars; the Gemini comparison reuses a representative subset of 960 prompts. An anonymity variant replaces class names with opaque tokens so the simulator cannot shortcut the task by recognising the output vocabulary.

## What counts as an explanation here

A concept space (a dictionary of linear directions learned unsupervisedly from penultimate-layer activations, with a decoder), a per-sample and per-class importance score over those directions, and a short textual interpretation of each direction. The explanation is partial by construction: only the concepts above an importance threshold are shown, and the importance is communicated as one of four signs rather than a value. Its predictive content is precisely what is being measured — the score *is* the explanation's ability to let a different agent predict the explained model's outputs.

## How they judged success

Simulatability accuracy on held-out evaluation-phase samples, converted to a method ranking by Copeland's method, with statistical significance from pairwise paired tests across settings. Baselines are prompts carrying the same task description and examples but no explanation, and a no-learning-phase variant, so a method has to beat "explain nothing". Accuracy is reported both with real class names and with anonymised ones. Concept-space-only metrics are deliberately not used, and the paper positions simulatability as subsuming faithfulness and plausibility because it is end-to-end.

## What they found

Concept explanations do help a machine reader predict the model, and the ranking of extraction methods is stable across simulators and across settings. NMF ranks first overall, followed by SAE and ICA in the top three, then a gap, then PCA and SVD, with SVD typically ranked last — and PCA and SVD fall *below* the no-explanation baseline. In the model-fixed and dataset-fixed subsets the order is preserved: NMF first whenever it applies, PCA and SVD in the bottom three. Most pairwise differences clear the t-test, though the ordering among SAE, ICA, the baseline, PCA and SVD is not individually significant everywhere, and SAE-versus-ICA is not significant for all simulators. For interpretation, the cheap word-list method beats the expensive GPT-o1 alignment method in most settings, significantly, and both beat the no-explanation baseline. Most striking for methodology: with real class names the prompt-type comparisons show no significant difference between having an explanation and not, and the authors read this as the simulator shortcutting the task; only under anonymised class names does the explanation produce a clear gain. Local explanations add nothing once a global explanation is present. Their recommendation is NMF where non-negative embeddings are achievable, SAE otherwise, and the word-list interpretation as a constant-cost baseline.

## Limits and open problems they name

The penultimate layer as the place to look for concepts is imported from the vision literature and assumed to carry over to text. The SAE they used predates the newer architectural variants, so SAE's ranking is probably understated. And crucially, they have no proof that a simulator-based ranking would coincide with one built from human participants — it is an approximation licensed by earlier correlational work, not a validated substitute. Three further caveats are visible in the design rather than named as limitations: the simulatability target is the model's label, so the score mixes "understands the model" with "can imitate it from surface cues"; the model itself is only ever presented to the simulator through 20 labelled examples per setting; and the anonymity result implies that the non-anonymised results, which carry the main ranking, are partly a measure of the simulator recognising class names.

## What we take from it

Three things. First, a *measure* we should copy almost directly: simulatability turns our white-box account into an artefact whose quality is "does an independent agent, given only the account, predict the network's outputs" — a criterion that is stricter than any faithfulness metric and that can be run at scale, with the anonymity trick as the control that tells us the reader is using the account rather than the surface. Second, a *warning* that generalises far beyond concept-XAI: a metric computed inside the representation can rank a method above the do-nothing baseline while the end-to-end reader shows no gain, and PCA and SVD ranking *below* no-explanation-at-all is the cleanest possible demonstration that in-space quality is not a proxy for usefulness. Third, a *framing* worth borrowing for our comparisons — across-settings scores are converted to votes and aggregated by ranked-choice before any method is called better, which is a principled way to summarise a benchmark that has no single common scale.

## Tags

unit: direction | concept
form: dictionary, direction
method: decomposition (ICA/NMF/PCA/SVD/SAE), other (LLM-as-simulator evaluation)
eval: prediction/simulation, human-rating (approximated by LLM), benchmark
goal: evaluation-of-methods
