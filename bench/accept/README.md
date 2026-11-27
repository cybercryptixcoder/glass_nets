# The acceptance suite: how to call the scorer

*Written by the bench-writer (`docs/spec/BENCH_BRIEF.md`), Nov 27, 2026. The interface the box must meet is
`bench/accept/INTERFACE.md` (code: `bench/accept/interface.py`). Changes to the interface and to the scorer's code are
logged in `bench/accept/CHANGES.md`.*

## What is here

| path | what it is |
|---|---|
| `INTERFACE.md`, `interface.py` | the contract: the task folder, the blueprint, the box's entry points, the format check |
| `netlib.py` | the network loader: `load_network(task folder)` runs a network and intervenes at its named sites |
| `scorer/` | the development copy of the scorer (the scorer of record is kept privately until certification) |
| `vendor/` | the scorer's frozen copies: `vocab/` (vocabulary v2 at f0b3b680) and `lab_nets.py` (lab/nets.py at b629f1e3) |
| `draws/<draw id>/<network id>/` | development networks: task folders the box explains |
| `sealed/` | gitignored: the development draws' sealed answers, which the scorer reads and the builder never opens |

## Calling the scorer

```
# score one blueprint against one development network (prints pass or fail, the score and the line per requirement)
python3 bench/accept/scorer/score.py --task bench/accept/draws/<draw>/<net> --blueprint <out>/blueprint.json

# score every network of a draw that has a blueprint in <out dir>/<net>/blueprint.json
python3 bench/accept/scorer/score.py --draw <draw> --blueprints <out dir>

# Gate 1's verifier hook: feeds the builder's checker (Box.verify) the sealed true blueprints and S5a accounts
python3 bench/accept/scorer/score.py --hook --draw <draw> --start <start config.json>
```

The scorer reads sealed answers from `bench/accept/sealed/<draw>/`. On development draws it prints, per network and per
requirement, pass or fail with the box's score and the line, and, for Gate 2's diagnostics, how many sealed
intermediate variables were recovered and which of the box's own parts matched one. On the certification draw it
prints only pass or fail, per requirement and per tier. It never prints a sealed definition, a family or a mechanism.

## Draws

Draw requests go in `docs/build/STATUS.md` as "Draw request: <gate>, <tiers>". A ready draw is announced there as
"Draw <id> ready: <tiers>". Every formal gate attempt needs a draw that no earlier formal attempt used.

## The reader model (R10)

Fixed by the bench for the whole build: the frontier model whose key Shreyas supplies, at a version recorded here.
**Pending: no frontier-model key exists yet**, so the usability lines are marked pending and are computed before
Gate 5.

## Checksums of large weights

None yet: every development network so far is under 50 MB and committed in its task folder.
