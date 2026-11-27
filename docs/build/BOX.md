# The box, in plain words (Nov 27, end of session 2)

## What it is

A program that takes a trained or compiled neural network and returns a blueprint of it: the network's own variables,
where each one sits inside the network, what each one computes, how they are wired, which claims were checked by
experiments on the network, and a list of what it could not explain.

## What it does, step by step

1. **Map.** Find the network's own variables. On a compiled transformer: the directions each layer writes into the
   residual stream (on the bench, first recovering a frame, because the network comes in a scrambled basis with no
   labels). On a trained network: the directions the output actually depends on (the top directions of the output's
   gradient). On Leela: dictionary features ranked by how much the evaluation depends on them.
2. **Define.** Write each variable as a short expression in the frozen vocabulary, from the inputs or from variables
   already defined, by search. A definition must reproduce the variable on inputs the search did not fit.
3. **Invent.** When nothing in the vocabulary fits, search small new operations (for example 1/(x + 1)) and record
   them, with where they came from.
4. **Assemble.** Put the parts together, with their wiring read from the weights, into one blueprint whose top part
   is the network's output.
5. **Certify.** Run the blueprint alone and compare it with the network; then swap each variable between two inputs
   inside the network and check the network's output changes the way the blueprint predicts. A part passes only if it
   beats lines from controls: random directions and networks with shuffled weights.
6. **Ledger.** Whatever is not defined stays in the blueprint as unresolved, with how far the best attempt got.
7. **Reader test.** Give the blueprint to a language model and ask it to predict the network's outputs on new inputs,
   with controls (no blueprint, scrambled blueprint, another network's blueprint).

## What it showed (session 2)

- On the bench's development networks (compiled transformers in a scrambled basis, no labels, never seen before):
  16 of 30 explained exactly and certified by the box's own checker. On the bench's public reference network, the
  bench's own scorer passes the box's blind blueprint on every requirement.
- On a network trained (not compiled) to compute relu(A - B) * C over real chess positions: the box found that the
  network keeps A - B as a single variable and C as another, and that the output is relu of the first times the
  second. Certified by swap tests.
- On Leela (a real chess engine network): the evaluation is a readout of four internal variables (certified), and each
  of those is a readout of four variables one level down (11 of 16 links certified). What those variables mean on the
  board is still unresolved.
- The reader test: with the blueprint, the reader predicted every output of the five public programs; without it, or
  with the wrong one, it mostly failed.

## How to run it

    python3 -m box.bench_s0 <bench task folder> <out folder>     # a bench S0 network; BOX_CERTIFY=1 adds step 5
    python3 -m box.run tracr <program>                          # a public Tracr program
    python3 -m box.learned_box                                  # the trained network with a known answer
    python3 -m box.run leela                                    # Leela LD2, top level

## What it cannot do yet

Split a layer that writes several variables at once (6 of the 30 bench networks); reach definitions that need longer
compositions; run on the bench's other tiers; score itself against the bench's sealed answers; test copies (R9); use a
frontier reader; work on the fluid and weights domains.
