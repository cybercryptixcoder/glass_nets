"""The box: one command from a network to a certified, readable blueprint.

    python3 -m box.run tracr <program>      a compiled transformer, blind (frac_prevs, hist, reverse, sort_unique,
                                            pair_balance)
    python3 -m box.run staged               the staged chess network (a builder's rehearsal: ingredients, verdict)
    python3 -m box.run leela                Leela's LD2, its evaluation at block 10 (a real network)

Each writes results/build/blueprints/<name>.md (the blueprint a reader sees) and .json, and prints the verdict.
The stages are the same everywhere (SPEC.md §1): map the network's variables, define each one from the frozen
vocabulary and from earlier parts (inventing primitives where the vocabulary falls short), assemble the hierarchy
with its wiring and an unresolved ledger, and certify it by interventions against controls.
"""
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def tracr(program):
    from box import report as REP
    from box.test_checker import programs
    from box.tracr_box import run_box
    prog, voc = programs()[program]
    net, bp, rep, variables, defined = run_box(program, prog, voc, verbose=True)
    REP.save(program, bp, rep, os.path.join(ROOT, "results", "build", "blueprints"))
    return rep["verdict"]


def staged():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "box", "rehearse_staged_check.py")], capture_output=True, text=True, cwd=ROOT)
    print(out.stdout[-3000:])
    return "certified" if "PASS rehearsal blueprint" in out.stdout else "rejected"


def leela():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "box", "leela_box.py")], capture_output=True, text=True, cwd=ROOT)
    print(out.stdout[-4000:])
    return "see results/build/leela_box_session2.json"


if __name__ == "__main__":
    kind = sys.argv[1]
    v = {"tracr": lambda: tracr(sys.argv[2]), "staged": staged, "leela": leela}[kind]()
    print("verdict:", v)
