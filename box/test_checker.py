"""Known cases for the checker v0 (ledger G1), on public Tracr programs (S0's kind). Written before the checker was
first run. Must pass: the true blueprints of frac_prevs, hist, reverse and sort_unique. Must fail: a wrong definition,
a wrong location, a random location, a rule mislabelled as a lookup, an account given as code, a definition outside
the grammar, a sixth constant, a cycle. Run: python3 box/test_checker.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box.adapters.tracr_net import TracrNet  # noqa: E402
from box.blueprint import Blueprint, Part  # noqa: E402
from box.verify import Checker  # noqa: E402

RESULTS = []


def first_site(net, labels, inputs):
    """The first site at which the subspace carries anything (where Tracr writes the variable)."""
    _, sites, _ = net.run(inputs, return_sites=True)
    B = net.subspace(labels)
    for s in net.sites:
        if np.abs(sites[s] @ B).max() > 1e-6:
            return s
    return net.sites[-1]


def programs():
    from tracr.compiler import lib
    from tracr.rasp import rasp
    return {
        "frac_prevs": (lib.make_frac_prevs(rasp.numerical(rasp.Map(lambda t: t == "x", rasp.tokens)).named("is_x")).named("frac"),
                       ["a", "b", "x"]),
        "hist": (lib.make_hist().named("hist"), ["a", "b", "c"]),
        "reverse": (lib.make_reverse(rasp.tokens).named("reverse"), ["a", "b", "c"]),
        "sort_unique": (lib.make_sort_unique(rasp.tokens, rasp.tokens).named("sort"), [1, 2, 3, 4, 5]),
        "pair_balance": (lib.make_pair_balance(rasp.tokens, "(", ")").named("pair_balance"), ["(", ")", "a"]),
    }


def inputs_for(name, voc, n=60, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        L = int(rng.integers(2, 6))
        s = list(rng.choice(voc, size=L, replace=(name != "sort_unique")))
        out.append([int(x) if isinstance(voc[0], int) else str(x) for x in s])
    return out


def true_blueprint(name, net, inputs):
    lab = lambda n: next(l.split(":")[0] for l in net.labels if l.split(":")[0].startswith(n))  # noqa: E731
    def part(pid, sop, definition, upstream, encoding="categorical"):
        labels = net.labels_of(lab(sop))
        return Part(pid, "rule", "SOp", first_site(net, labels, inputs), labels=labels, encoding=encoding,
                    definition=definition, upstream=upstream)
    if name == "frac_prevs":
        parts = [part("is_x", "is_x", '(map (eqc "x") tokens)', [], "numerical"),
                 part("frac", "frac", "(aggregate (select indices indices LEQ) @is_x)", ["is_x"], "numerical")]
        return Blueprint("tok", parts, "frac", name="frac_prevs (true)")
    if name == "hist":
        return Blueprint("tok", [part("hist", "hist", "(selector_width (select tokens tokens EQ))", [])], "hist", name="hist (true)")
    if name == "reverse":
        parts = [part("length", "length", "(selector_width (select tokens tokens TRUE))", []),
                 part("opp_idx", "opp_idx_", "(sequence_map minus @length indices)", ["length"]),
                 part("opp_m1", "opp_idx-1", "(map dec @opp_idx)", ["opp_idx"]),
                 part("reverse", "reverse", "(aggregate_cat (select indices @opp_m1 EQ) tokens)", ["opp_m1"])]
        return Blueprint("tok", parts, "reverse", name="reverse (true)")
    if name == "pair_balance":
        parts = [part("bo", "bools_open", '(map (eqc "(") tokens)', [], "numerical"),
                 part("opens", "opens", "(aggregate (select indices indices LEQ) @bo)", ["bo"], "numerical"),
                 part("bc", "bools_close", '(map (eqc ")") tokens)', [], "numerical"),
                 part("closes", "closes", "(aggregate (select indices indices LEQ) @bc)", ["bc"], "numerical"),
                 part("balance", "pair_balance", "(sequence_map minus @opens @closes)", ["opens", "closes"], "numerical")]
        return Blueprint("tok", parts, "balance", name="pair_balance (true)")
    parts = [part("target_pos", "target_pos", "(selector_width (select tokens tokens LT))", []),
             part("sort", "sort", "(aggregate_cat (select @target_pos indices EQ) tokens)", ["target_pos"])]
    return Blueprint("tok", parts, "sort", name="sort_unique (true)")


def variants(name, bp, net, rng):
    """Must-fail accounts derived from the true blueprint (fixed before the checker was run)."""
    import copy
    out = []
    b = copy.deepcopy(bp)
    rule = [p for p in b.parts][-1]
    wrong = {"frac_prevs": "(aggregate (select indices indices LT) @is_x)",
             "hist": "(selector_width (select tokens tokens NEQ))",
             "reverse": "(aggregate_cat (select indices @opp_m1 LT) tokens)",
             "sort_unique": "(aggregate_cat (select @target_pos indices LT) tokens)",
             "pair_balance": "(sequence_map plus @opens @closes)"}[name]
    rule.definition = wrong
    b.name = f"{name}: wrong definition"; out.append(b)
    b = copy.deepcopy(bp)
    p0 = b.parts[0]
    p0.labels = ["one"]; p0.encoding = "numerical"
    b.name = f"{name}: wrong location (the constant direction)"; out.append(b)
    b = copy.deepcopy(bp)
    p0 = b.parts[0]
    k = len(p0.labels)
    Q, _ = np.linalg.qr(rng.normal(size=(net.d, k)))
    p0.labels, p0.basis = None, Q
    b.name = f"{name}: random location"; out.append(b)
    b = copy.deepcopy(bp)
    p0 = b.parts[0]
    p0.type, p0.table = "lookup", {}
    b.name = f"{name}: a rule mislabelled as a lookup"; out.append(b)
    out.append(("def account(tokens):\n    return net(tokens)\n", f"{name}: an account given as code"))
    b = copy.deepcopy(bp)
    b.parts[-1].definition = "(lambda t: t)"
    b.name = f"{name}: a definition outside the grammar"; out.append(b)
    b = copy.deepcopy(bp)
    b.parts[-1].definition = '(map (eqc "a") (map (eqc "b") (map (eqc "c") (map (eqc "x") (map (eqc 1) (map (eqc 2) tokens))))))'
    b.name = f"{name}: six constants"; out.append(b)
    b = copy.deepcopy(bp)
    b.parts[0].upstream = [b.parts[-1].id]
    b.name = f"{name}: a cycle"; out.append(b)
    return out


def main():
    rng = np.random.default_rng(1)
    for name, (prog, voc) in programs().items():
        net = TracrNet(prog, voc, 5)
        X = inputs_for(name, voc)
        chk = Checker(net, X, seed=0, n_pairs=120, control_reps=100)
        bp = true_blueprint(name, net, X)
        own = chk.run_alone(bp)
        # the line is the true blueprint's own run-alone score minus its bootstrap spread over inputs
        agree = np.array([chk.base[i] == chk.values(bp, [i])[bp.output][0] if not net.numerical_output else
                          np.allclose(chk.base[i], chk.values(bp, [i])[bp.output][0], atol=1e-3) for i in range(len(X))], float)
        spread = float(np.std([agree[rng.integers(0, len(agree), len(agree))].mean() for _ in range(200)]))
        line = own - spread
        r = chk.certify(bp, line)
        RESULTS.append((r["name"], "certified", r["verdict"], r))
        for v in variants(name, bp, net, rng):
            if isinstance(v, tuple):
                r = chk.certify(v[0], line); r["name"] = v[1]
            else:
                r = chk.certify(v, line)
            RESULTS.append((r["name"], "rejected", r["verdict"], r))
    bad = 0
    for nm, want, got, r in RESULTS:
        ok = want == got
        bad += not ok
        why = next((c for c in r["checks"] if not c[1]), None)
        print(f"{'PASS' if ok else 'FAIL'} {nm}: expected {want}, got {got}" + (f" ({why[0]}: {why[2]})" if why else
              f" (IIA line {r.get('iia_line', float('nan')):.3f})"))
    print(f"\n{len(RESULTS) - bad} of {len(RESULTS)} known cases as expected")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
