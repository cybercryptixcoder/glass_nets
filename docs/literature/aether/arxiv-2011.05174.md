# Safety Verification of Neural Network Controlled Systems

- **Key:** arxiv-2011.05174
- **Authors / organisation:** Arthur Clavière, Eric Asselin, Christophe Garion, Claire Pagetti (Collins Aerospace France; ISAE-SUPAERO; ONERA; ANITI)
- **Date:** 2020-11-10 (v1)
- **Link:** https://arxiv.org/abs/2011.05174
- **Kind:** paper
- **Read from:** full text
- **Relevance to us:** 2 (useful)

## What they wanted, and why

Certification standards for airborne software assume that each item's requirements can be written down completely and that the implementation can then be shown to meet them. Neither assumption survives a learned controller: the only specification a trained network usually comes with is a set of examples, which says nothing about the rest of its input space, and training offers no correctness guarantee to check against. So the authors abandon item-level refinement and argue for a *system-level* claim instead: build one model of the plant and the controller together, compute a sound enclosure of everything the closed loop can reach, and show that this enclosure never touches the failure region. The larger goal is a safety argument that a certification authority could accept for a learned component inside a safety-critical aircraft function, and the demonstration target is the neural re-implementation of the ACAS Xu collision-avoidance logic.

## What they studied

A closed loop made of a continuous-time plant described by an ordinary differential equation and a discrete-time controller that runs every T seconds. The controller is deliberately not a bare network: it is a selection map from the previous command to one member of a finite family of ReLU networks, wrapped in a deterministic preprocessing stage (state to network input) and a deterministic postprocessing stage (network output to one of a finite set of actuation commands). The state under analysis is the pair (plant state, currently held command), and the reachable set is a set of such pairs over the horizon. For the ACAS Xu instance: five networks with six hidden layers of fifty units for the t_sep = 0 slice, a five-dimensional relative-kinematics state, five discrete turn-rate commands, a 1 s control period, an 8000 ft sensor range, a 500 ft collision cylinder, and a 20 s horizon.

## How they did it

The method composes two existing over-approximation instruments along the loop's own seams, one per kind of dynamics.

The plant part is validated simulation: over each control interval the command is constant, so the ODE is integrated with a rigorous enclosure (a Löhner-style two-step scheme — a Banach-fixed-point enclosure for the whole interval plus an interval integration with its local truncation error) producing a box that contains every trajectory from a box of initial states, and a tighter box for the state at the far end of the interval.

The controller part is abstract interpretation: interval arithmetic on the preprocessing function, a network abstract transformer (here ReluVal's symbolic interval propagation) on the selected network, and interval arithmetic again on the postprocessing, whose output is not a box of commands but the *finite set* of commands still reachable.

The two are stitched at the sampling instants: the plant's end-of-interval box becomes the next control step's input box, and the command set produced by the controller becomes the labels of the next step's states. This motivates the central data structure — a *symbolic state*, an (l-dimensional box, single command) pair, and a *symbolic set*, a finite collection of those. The command component is what makes the representation tighter than a box over the product space: because the command can only take finitely many values, a joint box over (state, command) would have to admit every combination, while the tagged collection only admits the pairs that actually occur. The overall procedure walks control steps j = 0..q, emitting one symbolic set for the reachable states inside each interval [jT,(j+1)T) and one for the reachable states at each sampling instant, and unions them into the final enclosure. Soundness of the union is proved by a two-part induction over steps and intervals, with the interval part leaning on continuity of the state across the sampling instant.

Three refinements are then layered on. Precision: replace the single interval enclosure per control step with M sub-enclosures obtained by M successive validated simulations, since one box over a whole period admits arcs the plant never takes. Cost: the count of symbolic states can multiply by the number of commands at each step, so once a set exceeds a threshold Γ the two nearest states *with the same command* are joined into their bounding box, repeatedly, using a Euclidean distance between box centres; the threshold must exceed the number of commands or the loop cannot terminate. Termination: a state entirely inside the target set is not propagated further, which needs the caveat that termination is only detectable at sampling instants.

The implementation is Python over DynIBEX for validated simulation, ReluVal for the networks, and interval arithmetic for the outer stages. The ACAS Xu initial set is meshed empirically into 198,764 boxes (629 arcs of 80 ft around the sensor circle by 316 headings of 0.01 rad), with M = 10 and Γ = 5, and a failed box is bisected along its three position/heading dimensions up to depth two.

## What counts as an explanation here

A certified enclosure, not a description. The object produced is a finite union of labelled boxes that provably contains every reachable state of the closed loop over the horizon, plus a Boolean saying the enclosure misses the failure set and the loop terminated. It is an over-approximation by construction and is explicitly incomplete: failing to produce a safe verdict means "not proved", never "unsafe". Nothing here is decomposed into network components — no neuron, feature or direction is named — so the account is predictive of the *system* and silent about the model's internals. The one place internal structure appears is incidental: the enclosure is a function of the partition, so the partition's non-uniform difficulty is the only thing the account says about the controller's behaviour.

## How they judged success

Two measured quantities over the meshed initial set: coverage — the percentage of initial states (weighted over split depth by the reciprocal of the split factor) for which safety until termination was proved — and elapsed analysis time, both reported per region of the initial-position space, with the spatial resolution being 500 ft arcs. There is no comparison against a competing verifier, no ablation of the components, and no randomness to average over, since the procedure is deterministic given the mesh. Correctness rests on the soundness theorem and on the two component tools being sound, not on a benchmark.

## What they found

The full run took about twelve days of wall clock on two 12-core Xeon hosts and certified 90.3% of the meshed initial states, so the headline is a near-miss: there is no complete proof of safety for this controller, and the authors say so plainly and argue that the partial result is still worth having.

The partial result is more informative than its size suggests, because it is spatial. The uncovered and slow regions are the approaches from the left and from the right (negative x with positive y, and positive x with positive y): coverage there falls to roughly 75% while elsewhere it runs from 85% to 100%, and the analysis time in those regions is on the order of five times ten to the fourth seconds against a thousand or less elsewhere. The authors read this as the system-level statement that the hardest situations are lateral approaches rather than head-on ones, and observe that the coverage and timing maps are roughly symmetric about the zero-heading axis, which is what the geometry of the encounter would predict, and which they treat as evidence that the analysis is behaving sensibly rather than as a result in itself.

Two uses for a negative verdict are proposed. The per-region map could drive a runtime monitor that hands control elsewhere when the current state lies in an uncertified region; and the map is feedstock for retraining, since the regions it flags are the ones where the controller's behaviour is least constrained by the analysis.

## Limits and open problems they name

The obvious limit is the missing 9.7%, and it is structural rather than incidental. Partitioning is the only source of precision at the level of the initial set and it is chosen empirically and uniformly; the authors name a better partitioning strategy as future work and point at mesh generation in computational fluid dynamics for inspiration. Splitting is plain bisection along fixed dimensions, and they propose instead splitting along the variable that most influences the system. For the uncertified regions they want a falsification search, which would convert "not proved" into either a witness or a tighter enclosure. Errors in the network inputs and in the sampling, and the alternative assumption of a varying intruder velocity, are out of scope. The termination detector can miss the true termination instant because it only looks at sampling instants, and the paper argues that this is tolerable when the target set behaves like an attractor — an argument, not a guarantee. Everything after the first command switch is conditional on the plant model: constant velocities, level flight, straight-and-level intruder.

## What we take from it

**Tag the discrete factor instead of boxing the product space.** The symbolic state is a box plus a single value of the discrete control variable, and that one design choice buys more precision than any amount of box tightening, because the discrete factor takes finitely many values and a joint box must admit the cross product. Warning attached: the gain is also what creates the blow-up, and the mitigation the authors use — join nearest states *within the same tag* — preserves soundness and destroys the precision it was protecting at a rate controlled by an arbitrary threshold. For our account the analogue is exact: whenever the thing we are modelling has a discrete branch (a mode, a selected head, a chosen expert), we keep the branch index as a tag on the region rather than smearing it into the numeric enclosure, and we report the tag count and the join threshold as parameters because they are the whole precision/cost story.

**Different dynamics, different instruments, joined at the seam where each is exact.** The plant is handled by validated simulation and the controller by abstract interpretation with no attempt to unify them; the join is the sampling instant, where the first instrument's output box is exactly the right input type for the second, and the second's output is exactly the right label type for the first. This is the compositional version of run 618's stacking rule and it says something the single-instrument papers cannot: a soundness proof for a composite is an induction over the *interface*, so the interface type (here, a box at a sampling instant, shared by both sides) is the design decision that determines whether the composition can be proved at all. *Take:* for a composite or multi-stage account, we design the intermediate object so that both the producer and the consumer can consume it natively, and we write the soundness argument as an induction over that object.

**A partial certified map is a first-class deliverable, and its value is in the geometry of what is missing.** Coverage-with-a-map and a single scalar coverage are different artefacts: 90.3% as a number invites "nearly done", while the same figure drawn over the initial set says which approaches are hard, at what cost, and with what symmetry, and the symmetry check doubles as a sanity test of the analysis itself. Three downstream uses are concrete and each would be unthinkable from the scalar — runtime monitoring, targeted retraining, and a claim about where the controller is least constrained. *Take:* whenever our account leaves part of its domain undecided, we publish the decided set as a set (not a fraction), we report cost per region as well as verdict per region, and we include at least one structural prediction the partition should satisfy (a symmetry, a monotonicity) and say whether the results show it.

**Unspecified thresholds are where the account's honesty lives.** M, Γ and the split depth are all stated, all set by trial, and all explicitly trade precision for time; the join heuristic is a heuristic and nothing in the soundness theorem depends on it being good. This is the same discipline run 620 asked for, and it is worth noting that here the parameter that matters most is not the one that sounds most important: Γ = 5 equals the number of commands, which is the minimum value that lets the join loop terminate at all, so the run was done at the edge of feasibility rather than at a tuned optimum. *Take:* we list the knobs, give the setting used, and distinguish — as this paper does not quite — between knobs whose value is forced by the algorithm's correctness (Γ > P) and knobs whose value is chosen by taste.

## Tags

unit: whole-model
form: causal-abstraction
method: ablation
eval: prediction/simulation
goal: safety/auditing
