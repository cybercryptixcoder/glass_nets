# AI safety via debate

- **Key:** arxiv-1805.00899
- **Authors / organisation:** Geoffrey Irving, Paul Christiano, Dario Amodei — all OpenAI (affiliation stated once for the byline; Irving marked corresponding author with an openai.com address)
- **Date:** 2018-05-02 (v1); this entry read from v2, 2018-10-22
- **Link:** https://arxiv.org/abs/1805.00899
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why
The authors take as given that aligning a capable system means supervising it with human judgements, and ask what to do when the task is past the point where a person can judge a single answer. Their move is to keep the human as judge but change what the human sees: instead of one answer, two advocates arguing, so the information the judge must absorb is compressed into a contest rather than an artifact. The proposal is a zero-sum self-play game between two agents who state a claim, then alternate short statements, with a human awarding the win. Everything around it — a complexity-theoretic argument, one small image experiment, a long list of worries — is there to make the proposal precise enough to be tested. It is a programmatic paper: the architecture is proposed, not built.

## What they studied
No trained model is offered as a result; the unit of analysis is a *protocol* (who speaks, in what order, what the judge sees, what payoff each side maximises). The one experiment uses an MNIST classifier as a fixed judge that sees only a handful of pixels, with two players — one designated honest, one designated liar — selecting which pixels the judge is shown. A second, human experiment is specified but only reported informally. The intended target system is a natural-language agent pair, which the paper says it has not trained.

## How they did it
The game is defined first: question and context are shown to both players; both declare answers, which may coincide; they alternate statements up to a fixed turn limit; the judge decides a winner; the payoff is zero-sum, so each player maximises win probability. The claimed justification is that a single path through the space of arguments is evidence about the whole tree of arguments, in the way one game between strong Go players is evidence about the whole game tree. In the MNIST arm, a convolutional judge is trained in advance to classify digits from a random sparse mask of 4 or 6 nonzero pixels and then frozen; the players never see it train, because joint training would supply the truth the protocol is meant to elicit without it. Players are run by Monte Carlo tree search with 10,000 rollouts per move and the PUCT selection rule, which stands in for agents with no generalisation error. One player is fixed honest and one is fixed to lie about the label; in the precommit condition the liar must fix the lie before moving, and in the no-precommit condition any judge error counts as a liar win. Human play is arranged through a prototype web page where debaters may type, draw rectangles, and reveal exactly one pixel over the whole game.

## What counts as an explanation here
A transcript: a short adversarial dialogue that ends at a point where the judge feels able to decide. It is deliberately not a summary of all reasons — the paper is explicit that the reason for an answer is the whole tree and the transcript is one path through it — and it is not presented as faithful in any mechanical sense. What it is supposed to buy is a *guarantee-like* condition: that at equilibrium a false claim is harder to defend than to attack, so the winner's claim is true even when the judge cannot verify it. The complexity-theory section turns that into a claim about what judge power suffices, replacing the human with a polynomial-time algorithm and asking which questions become decidable.

## How they judged success
For the machine arm: judge accuracy on the MNIST test set when deciding from the sparse mask (the baseline is the same frozen judge on randomly chosen masks), plus the fraction of games in which the honest player wins. Because one side is designated honest, an honest win rate above the judge's unaided competence is the operational form of "honesty is the best strategy". The complexity argument is offered as an analogy rather than a measured result. For the human arm, no metric is reported at all — it is described as played informally, with the conclusion stated in words.

## What they found
The frozen judge classified from 6 random nonzero pixels at 59.4% and from 4 at 48.2%. Under debate with precommit the same judge reached 88.9% and 85.2% respectively, and the honest player won 90.4% (6 pixels) and 86.7% (4 pixels) of games; the authors add that without the mean-over-3-seeds adjustment those honest rates are 87.7% and 83.8%. In the no-precommit condition the table reports only the liar's win rate (67.4% at 6 pixels, 59.4% at 4) and the judge's accuracy, so there is no honest-win cell to compare directly, though the prose states that precommit is decisive and that the liar does much better when the lie can adapt across the game tree. The authors also report that the successful lies concentrate on particular digit pairs (5 argued as 8, 4 argued as 9) and attribute that to the nonzero-pixel restriction, since an honest debater could in principle show the black pixels where an 8 would be white. In the human cat-versus-dog game they say lying was very hard as soon as players understood the rules, and give no counts.

## Limits and open problems they name
The paper is candid that the natural-language case is untested and that the MNIST result stands proxy for only one property (a judge with less information than the debaters). Section 5 is a long list of ways the idea could fail: judges may accept what they already believe; debates may be longer than a person can track; whether humans can check single steps containing unfamiliar terms is an empirical question; debate may cost capacity relative to direct training; the neighbourhood around equilibrium may not be honest even where equilibrium is; self-play may cycle between honest and lying phases; and stochastic simulation breaks the interval-branching protocol because a liar can choose noise that is locally plausible and globally improbable. They also flag that a single-pixel reveal budget is not serious for natural language, that they detect imbalance between the two sides only in part, and that debate does nothing about robustness, distributional shift or safe exploration.

## What we take from it
First, the *protocol* framing is the part that transfers: what a read-out may claim depends on who is allowed to see what, in what order, with what payoff, and the folder's other entries on debate (`arxiv-2211.03540`, `arxiv-2210.10860`) can now be placed against the design they were testing. Second, the precommit condition is a reusable control we would otherwise miss: forcing both sides to stake a claim before evidence is produced removed a large share of the liar's advantage, which is the same shape as asking a surrogate model to commit to a prediction before its explanation is generated. Third, "one path is evidence about the whole tree" is exactly the claim our own accounts make when we show a circuit or a feature and assert it explains the behaviour, and the paper's own caveats are a checklist for when that inference is and is not licensed. Fourth, the honest-win-rate-against-a-designated-liar construction is a cheap, designed adversary — a way to price an account without needing a naturally occurring deceptive model.

## Tags
unit: whole-model
form: description
method: other
eval: downstream-task
goal: safety/auditing
