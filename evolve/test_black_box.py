"""Known cases for the sealed evaluator, written before it judges any evolved program (the glass_nets routine:
the measurement is proven on cases that must pass, must fail, and controls). Run: python3 evolve/test_black_box.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import black_box as B  # noqa: E402

TRUE_PLANTED = '''
import numpy as np
def explain(net, budget_seconds):
    u = np.zeros(net.hidden); u[0], u[1] = 1.0, -1.0
    return {"parts": [{"id": "s", "type": "rule", "site": f"h{net.T}", "definition": "(sum x)", "upstream": [],
                       "basis": (u / np.sqrt(2)).tolist(), "scale": float(np.sqrt(2))},
                      {"id": "decision", "type": "rule", "site": "readout", "definition": "(le @s 0)", "upstream": ["s"]}],
            "output": "decision"}
'''
WRONG_PLANTED = TRUE_PLANTED.replace('"(sum x)"', '"(sum (abs x))"')
EMPTY = '''
def explain(net, budget_seconds):
    return {"parts": [], "output": "none"}
'''
BEHAVIOUR_ONLY = '''
def explain(net, budget_seconds):
    return {"parts": [{"id": "decision", "type": "rule", "site": "readout", "definition": "(argmax (mod (rank x) 2))",
                       "upstream": []}], "output": "decision"}
'''
DECORATIVE = '''
import numpy as np
import boxkit as bk
def explain(net, budget_seconds):
    X = net.inputs(1000, seed=3); out, S = net.run(X)
    site = f"h{net.T}"
    v = bk.evaluate("(argmax (mod (rank x) 2))", X)
    u, s = bk.fit_location(S, site, v, np.arange(1000))
    return {"parts": [{"id": "a", "type": "rule", "site": site, "definition": "(argmax (mod (rank x) 2))",
                       "upstream": [], "basis": u.tolist(), "scale": s},
                      {"id": "decision", "type": "rule", "site": "readout", "definition": "(add @a 0)", "upstream": ["a"]}],
            "output": "decision"}
'''
FORBIDDEN = '''
import os
def explain(net, budget_seconds):
    return {"parts": []}
'''
SNEAKY = '''
import numpy as np
def explain(net, budget_seconds):
    np.load("/etc/passwd")
    return {"parts": []}
'''
LIST_REMOVE = EMPTY.replace("return", "xs = [1, 2]; xs.remove(1)\n    return")
CASES = [  # (name, code, network, low, high)
    ("planted, true account: must pass", TRUE_PLANTED, "planted_sum_4_4", 0.95, 1.0),
    ("planted, wrong variable: must fail", WRONG_PLANTED, "planted_sum_4_4", 0.0, 0.3),
    ("planted, empty blueprint", EMPTY, "planted_sum_4_4", 0.0, 0.0),
    ("AlgZoo M4,3, empty blueprint", EMPTY, "zoo_2nd_argmax_4_3", 0.0, 0.0),
    ("AlgZoo M4,3, behaviour only (no located part): rejected by form", BEHAVIOUR_ONLY, "zoo_2nd_argmax_4_3", 0.0, 0.0),
    ("AlgZoo M4,3, the answer claimed as a variable at the last state: must score low", DECORATIVE, "zoo_2nd_argmax_4_3", 0.0, 0.1),
    ("random network, empty blueprint (no story): full marks", EMPTY, "random_4_3", 1.0, 1.0),
    ("forbidden import: not run", FORBIDDEN, "zoo_2nd_argmax_4_3", 0.0, 0.0),
    ("file access through numpy: not run", SNEAKY, "zoo_2nd_argmax_4_3", 0.0, 0.0),
    ("ordinary list.remove: runs (a first ban rejected it)", LIST_REMOVE, "random_4_3", 1.0, 1.0),
]


def main():
    ok_all = True
    for name, code, net, lo, hi in CASES:
        r = B.evaluate(code, nets=[net])
        sc = r["metrics"].get(f"{net}.score", r["score"]) if r["valid"] else 0.0
        if not r["valid"]:
            sc = 1.0 if net.startswith(("random", "shuffled")) else 0.0
        ok = lo - 1e-9 <= sc <= hi + 1e-9 and (r["valid"] or "not run" in name)
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {sc:.3f} (want {lo} to {hi})")
        if not ok:
            print("      " + "\n      ".join(r["feedback"][:6]))
    print("all pass" if ok_all else "SOME FAIL")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
