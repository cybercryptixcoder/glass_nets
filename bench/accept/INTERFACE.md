# The interface: what the box hands the scorer (bench-accept/v1)

*Written by the bench-writer (`docs/spec/BENCH_BRIEF.md` §6), Nov 27, 2026. This file is an executable contract: its
Python form is `bench/accept/interface.py` (typed formats, the two protocols, and the format check the scorer itself
applies), and the network loader is `bench/accept/netlib.py`. Where this file and the code disagree, the code binds and
the disagreement is a bug to log in `bench/accept/CHANGES.md`. Definitions use the frozen vocabulary v2
(`bench/accept/VOCABULARY.md`, commit f0b3b680), whose encoding this interface adopts unchanged. The scorer uses its
own frozen copy, `bench/accept/vendor/vocab/`.*

Check a blueprint's format the way the scorer does:

```
python3 bench/accept/interface.py <task folder> <blueprint.json>
```

---

## 1. The task folder (what the bench hands the box, one per network)

```
<draw>/<network id>/
  task.json          the task (below)
  network/           the weights, in the format task.json names; netlib.load_network(task folder) runs them
  inputs.json        token tasks: {"tokens": [[...], ...]}      (or inputs.npz: {"x": [N, n]}, chess: {"fens"})
  train_items.json   where the training data is known: {"items": [{"id": "train:0", "input": ...}, ...]}
```

`task.json` (type `TaskSpec` in `interface.py`):

| field | meaning |
|---|---|
| `interface` | `"bench-accept/v1"` |
| `id`, `draw`, `tier` | the network's id, its draw, its tier (S0 to S7) |
| `domain` | the vocabulary domain definitions are written in: `tok`, `seq`, `chess`, `fluid`, `weights` |
| `network` | `{"format", "path", "n_params", "multiply_adds"}`; formats: `tracr_np`, `rnn_relu`, `transformer`, `mlp_relu` (more are added as tiers need them, each with a loader in `netlib.py` and a row in `CHANGES.md`) |
| `behaviour` | `{"description", "outputs": [{"name", "kind", "values", "positions"}], "metric", "decision"}`: which outputs, which metric (`accuracy` for categorical outputs, `r2` for numbers, `kl` for distributions), and how a decision is read (`argmax`, `sign`, or a `threshold`) |
| `inputs` | `{"path", "n"}`: the inputs the behaviour is explained on. The scorer's held-out inputs are separate and sealed |
| `train_items` | `{"path", "n"}` where the training items are known (R5's lookup keys must be among them), else `null` |
| `sites` | the named intervention sites: `{"name", "shape": [positions, width], "order", "stream", "value_offset", "labels"}`. Sites with the same `stream` share one residual basis. `value_offset` is the position of value index 0 (1 where position 0 is a BOS position). `labels` names the basis directions only where the bench publishes them |
| `params` | the named parameters `{"name", "shape"}` a parameter-component location may decompose |

The network object (`interface.Network`): `run(ctx, edits=None, param_edits=None, return_sites=False)` returns
`{"outputs", "raw", "sites"}`. `edits` maps a site name to a function from that site's activation `[N, *shape]` to a
new activation of the same shape; `param_edits` replaces named parameters for one run.

## 2. The blueprint (what the box hands the scorer, one per network)

One JSON file, `blueprint.json` (type `Blueprint`), with paths inside it relative to its own folder.

| field | meaning |
|---|---|
| `interface`, `task`, `domain` | `"bench-accept/v1"`, the task id, the task's domain |
| `parts` | every part (below) |
| `output` | the id of the part whose value is the network's output |
| `library` | invented primitives, `$name` -> entry (§4) |
| `overview` | `{"text", "predictor"}`: the one-page overview (at most 4,000 characters) and its callable top-level predictor, an S-expression over `@` parts (R6) |
| `systems` | `[{"id", "role", "parts", "fed_by", "feeds"}]`: groups of parts with their roles (R6) |
| `wiring` | optional `[{"from", "to", "strength"}]`; the parts' `upstream` lists are binding |
| `ledger` | `{"shares": {"rule", "lookup", "irrelevant", "unresolved"}, "unresolved_by_site": {site: share}}`; the scorer recomputes both |
| `rivals` | `[{"path", "separating_experiment"}]`: alternative blueprints that pass the same tests (R12) |
| `cost_map` | for networks with a reference method (R15): `{"reference", "steps": [{"step", "parts", "flops", "error"}], "skipped"}` |
| `rendering` | the reader rendering's `index.md`, relative to the blueprint (§7) |
| `usage` | what the run used: `{"gpu_hours", "model_tokens"}` (R13) |

### A part (type `Part`)

| field | meaning |
|---|---|
| `id` | unique within the blueprint |
| `type` | `rule`, `lookup`, `irrelevant` or `unresolved` |
| `vtype` | the vocabulary type of its value (`SOp`, `Seq`, `Num`, `Sq`, `Pos`, `Field`, `Mat`, `Vec`, ...) |
| `location` | `{"site", "positions", "map"}`: required for every part (R3) |
| `encoding` | how the value sits in the location: `numerical` (`scale`, `offset`; dimension 1), `categorical` (`values`, one basis direction per value, value = the largest coordinate's), or `vector` |
| `definition` | a rule's definition: one S-expression of the vocabulary (`VOCABULARY.md` §1) |
| `table` | a lookup's table: `{"entries": [[key, value], ...], "keys_are": "inputs" or "train_items", "default"}`; a key is `interface.canonical_key` of an input, or a train-item id |
| `upstream` | exactly the parts its definition names with `@id` |
| `choice` | set (to an output's name) when the part computes the network's own choice (§5) |
| `claims` | `[{"claim", "value", "p", "evidence"}]` for calibration (R12); `claim` is one of `type`, `rule_holds`, `iia_pass`, `substitution_pass`, `irrelevant_pass`, `lookup_pass`; `evidence` is one of `exhaustive`, `proven_bound`, `statistical`, `adversarial_search` |
| `evidence` | claim name -> evidence kind, for every claim the blueprint makes about the part |

**Positions.** `"all"`: one value per position (value index i is read at position `value_offset + i`); `[p]`: one value
per input, read at position p; `"flat"`: one value per input, read from the flattened site.

**The allowed map classes (R3).** Each is linear in the activations or the weights. Anything else is rejected, because
unrestricted maps make any network "implement" any algorithm.
- `{"class": "subspace", "basis": [[...]]}`: an orthonormal basis `[d, k]` of declared dimension k (a direction is
  k = 1), or `{"class": "subspace", "labels": [...]}` where the site publishes labels;
- `{"class": "dictionary_feature", "dictionary": "<file>.npz", "feature": i}`: a feature of a declared sparse
  dictionary, with its encoder and decoder supplied (`W_enc [d, m]`, `b_enc [m]`, `W_dec [m, d]`, `b_dec [d]`);
- `{"class": "param_component", "param": name, "decomposition": "<file>.npz", "component": i}`: a parameter component
  of a declared decomposition (`components [C, *param shape]`, summing to the parameter).

**How the scorer reads and writes a part.** Reading: the coordinates of the site's activation in the location, then
the encoding (numerical: `scale * coordinate + offset`; categorical: the value of the largest coordinate). Writing
(interchange, substitution): the location's coordinates are replaced and the orthogonal complement is left as it was;
for a dictionary feature, the activation moves along the feature's decoder row.

## 3. Definitions

A definition is an expression in the grammar of `VOCABULARY.md` §1, in its encoding (`(op arg ...)`, `@id` for a part,
`$name` for an invented primitive), checked by the vocabulary's own `admissible`: in the grammar, well typed, of an
allowed root type, at most L symbols (each part reference and each invented primitive counts as one) and at most 5
numeric constants after every invented primitive it uses is expanded. Arbitrary code is never a definition.

## 4. Invented primitives (R8)

Entries of `library`, each with `"provenance": {"run", "evidence"}`, which the scorer records:
- a **named expression**: `{"kind": "expression", "definition": text}`, a closed definition (no part references)
  within L and the constants budget, used later as the one symbol `$name`;
- a **new operation**: `{"kind": "operation", "source": python, "args": [types], "ret": type}`: one Python function
  using only arithmetic, control flow and `np.<function>` calls, with no imports, no input or output, no string
  literals and no access to the network, its weights or its activations, of at most **200 AST nodes** (fixed here,
  before S2 is written: `interface.MAX_OPERATION_NODES`). Its numeric literals are its constants.

Length counts each primitive as one symbol; constants are counted after expansion, so invention adds structure, never
fitted numbers.

## 5. The network's own choice

A definition refers to the network's own choice (the square its chosen move goes to, its planned line) only through a
part of the blueprint that computes it, marked with `choice`. When the scorer tests a part on the network (R3), every
part marked `choice` takes the network's actual choice; when the blueprint runs alone (R4), it takes the blueprint's
own value.

## 6. What the scorer computes from the blueprint alone

The scorer evaluates the blueprint itself; the box supplies no code for these.
- **Run alone (R4).** Every rule is its definition, evaluated on the inputs in dependency order; every lookup is its
  table (on inputs not in the table, its `default`, else a resampled value charged to the ledger); every unresolved
  or irrelevant part is resampled by the scorer from the network's values on other inputs. The evaluation runs in a
  separate process that is given the blueprint and the inputs and never the weights. The `output` part's values are
  the blueprint's predictions; every located part's values are compared with the network's, level by level.
- **Unusual inputs (R2).** The bench's unusual inputs go only to the run-alone evaluation, before the network is run
  on them; the blueprint's outputs there are its predictions of where the network fails.
- **Interventions (R3).** For each located part, on pairs of inputs, the scorer writes the part's value from the
  source input into the base input's run and compares the network's output with the blueprint's prediction
  (interchange accuracy), and writes the blueprint's own value of the part into the network (substitution: the share
  of the part's effect restored). Controls: random subspaces of the same dimension at the same site, the untrained
  and weight-shuffled network, and shuffled activity; the line is family-wise.
- **Kept parts suffice (R4).** At every site with a located part, the scorer keeps the span of the parts located
  there (and, on a shared stream, at earlier sites of that stream) and resamples the rest from the bench's resampling
  sets; it reports recovery input by input (its 5th percentile and mean) and the unresolved share by site.
- **The overview (R6).** The `overview.predictor` is evaluated like a definition, and compared with the sealed top
  level and with the task-rule baseline.

## 7. Entry points the box supplies (the `Box` protocol)

The start configuration (`StartConfig`) is one JSON file: `{"entry": "module:Class", "config": {...}, "budget":
{"gpu_hours": {"base", "per_param"}, "model_tokens": {"base", "per_param"}}}`. The scorer constructs `Class(config)`
once and uses the same entry point and configuration for every network and for certification.

| method | requirement | what it returns |
|---|---|---|
| `explain(task_dir, out_dir)` | all | writes `out_dir/blueprint.json` (with its rendering and rivals) and returns its path |
| `estimate(blueprint_path, question)` | R9 | `{"id", "estimate", "ops"}`: an estimate of a quantity over a distribution, and the floating-point operations it counted. Called in a process without the network's weights |
| `trace(blueprint_path, ctx, decisions)` | R7 | per decision: `{"id", "contributions": {part: value}, "flip", "p"}`; a flip is `{"kind": "input", "changes": [[element, new value], ...]}` or `{"kind": "part", "part", "value"}`. Flip size is the number of changed input elements (squares, tokens), or the L2 distance for continuous inputs; the bench's own search checks minimality among its candidates |
| `edit(blueprint_path, ctx, requests)` | R11 | per capability edit (`{"id", "capability", "kind"}`): `{"id", "parts", "actions": [{"part", "op", "factor", "value"}], "predicted": {"decisions", "p"}}`; the scorer applies the actions at the parts' locations and measures |
| `fingerprint(blueprint_path, behaviour)` | R7 | a detector `{"parts", "weights", "bias", "threshold"}` over the parts' values, which the scorer runs on activations |
| `verify(task_dir, blueprint_path)` | Gate 1 | `"certified"` or `"rejected"`: the builder's checker on one account. It must not print or write the account. The scorer feeds it the sealed true blueprints and the S5a accounts and prints only counts per tier: "certified X of N true blueprints, rejected Y of M accounts" |

Claims for calibration (R12) are the parts' `claims` plus the `p` of every flip and edit prediction. The reader
rendering (R10) is a folder of plain markdown: `index.md`, which the bounded reader opens first, and one file per part
or system, which it opens on demand within its budget. The scorer's reader is the only model call the scorer makes.

## 8. The reference true blueprint

The reference is a public S0 model: Tracr's own library program `make_frac_prevs` on `tokens == "x"`, compiled by
Tracr, with its residual basis as compiled and its labels published. Its task folder and full true blueprint are in
`bench/accept/reference/frac_prevs/` (built by `bench/accept/reference/make_reference.py`). The scorer passes it at the
ceiling: every statistic is 1.000 (run alone, every level, interchange, substitution, kept parts), with interchange
above its family-wise random-subspace line (0.301). Check it with:

```
python3 bench/accept/scorer/score.py --task bench/accept/reference/frac_prevs \
    --blueprint bench/accept/reference/frac_prevs/true_blueprint.json
```

Its parts (long label and value lists shortened here with "..."; each part also carries `claims` and `evidence`):

```
{"id": "tokens", "type": "rule", "vtype": "SOp", "location": {"site": "resid_0", "positions": "all", "map": {"class": "subspace", "labels": ["tokens:a", "tokens:b", "tokens:x"]}}, "encoding": {"kind": "categorical", "values": ["a", "b", "x"], "null_below": 0.5}, "definition": "tokens", "upstream": []}
{"id": "indices", "type": "rule", "vtype": "SOp", "location": {"site": "resid_0", "positions": "all", "map": {"class": "subspace", "labels": ["indices:0", "indices:1", "indices:2", "..."]}}, "encoding": {"kind": "categorical", "values": [0, 1, 2, "..."], "null_below": 0.5}, "definition": "indices", "upstream": []}
{"id": "is_x", "type": "rule", "vtype": "SOp", "location": {"site": "resid_2", "positions": "all", "map": {"class": "subspace", "labels": ["is_x"]}}, "encoding": {"kind": "numerical", "scale": 1.0, "offset": 0.0}, "definition": "(map (eqc \"x\") @tokens)", "upstream": ["tokens"]}
{"id": "frac_prevs", "type": "rule", "vtype": "SOp", "location": {"site": "resid_3", "positions": "all", "map": {"class": "subspace", "labels": ["frac_prevs"]}}, "encoding": {"kind": "numerical", "scale": 1.0, "offset": 0.0}, "definition": "(aggregate (select @indices @indices LEQ) @is_x)", "upstream": ["indices", "is_x"]}
```

with `"output": "frac_prevs"`, an overview whose predictor is `@frac_prevs`, and two systems (marking the x tokens;
averaging the marks over the prefix). In the sealed S0 networks the residual basis is an arbitrary orthonormal one and
no labels are published, so the same parts are located by `basis` maps instead of `labels`.
