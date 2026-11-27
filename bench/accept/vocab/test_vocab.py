"""Known-case tests for the starting vocabulary. Run: python3 bench/accept/vocab/test_vocab.py
Cases that must pass, cases that must fail, and a cross-check of S0-style circuits against Tracr's compiled models."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vocab  # noqa: E402

FAILS = []


def check(cond, what):
    print(("PASS " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


def chain(dom, ctx, parts_src):
    """Evaluate a blueprint-style chain: [(id, type, definition)], each part may use earlier parts by @id."""
    vals, types = {}, {}
    for pid, typ, src in parts_src:
        ok, why, t = vocab.admissible(src, dom, types)
        check(ok and t == typ, f"{dom.name}: part {pid} admissible ({why}, type {t})")
        vals[pid] = vocab.evaluate(src, dom, ctx, vals, types)
        types[pid] = typ
    return vals


def test_grammar():
    d = vocab.get("chess")
    check(vocab.admissible("(count (and own_P (not defended)))", d)[0], "chess: a 5-symbol rule is admissible")
    check(vocab.length(vocab.parse("(count (and own_P (not defended)))")) == 5, "length counts symbols")
    check(not vocab.admissible("(count (and own_P (not (ge (add (mul mob_us 2) 3) 4))))", d)[0], "chess: length 10 > L rejected")
    many = "(add (add (add (mul (ge mob_us 1) 2) 3) 4) 5)"
    check(vocab.constants(vocab.parse(many)) == 5, "five constants counted")
    lib = {"six": {"kind": "expression", "definition": "(add (add (add (add (add mob_us 1) 2) 3) 4) 5)"}}
    check(not vocab.admissible("(add $six 6)", d, library=lib)[0], "constants counted after expanding an invented expression")
    check(vocab.admissible("(add $six mob_us)", vocab.get("chess"), library={"six": {"kind": "expression",
          "definition": "(add mob_us 1)"}})[0], "an invented expression counts as one symbol")
    check(not vocab.admissible("(count banana)", d)[0], "unknown symbol rejected")
    check(not vocab.admissible("(count (count own_P))", d)[0], "type error rejected")
    check(not vocab.admissible("(and own_P", d)[0], "syntax error rejected")
    check(vocab.admissible("(and @move_to (not defended))", d, {"move_to": "Sq"})[0],
          "the network's own choice through a part reference (@id) is admissible")
    check(not vocab.admissible("(and @move_to (not defended))", d)[0], "an undeclared part reference is rejected")


def test_operations():
    good = "def see(a, b):\n    s = a - b\n    return np.maximum(s, 0) * 2\n"
    ok, why, n, lits = vocab.check_operation(good)
    check(ok and lits == 2, f"an arithmetic numpy operation is allowed ({why}, {n} nodes, {lits} literals)")
    f = vocab.compile_operation(good)
    check(np.allclose(f(np.array([3.0, 1.0]), np.array([1.0, 2.0])), [4.0, 0.0]), "a checked operation runs")
    bad = {
        "import": "import os\ndef f(a):\n    return a\n",
        "file read": "def f(a):\n    return open('x').read()\n",
        "np.load": "def f(a):\n    return np.load('w.npy')\n",
        "network call": "def f(a):\n    return net(a)\n",
        "dunder escape": "def f(a):\n    return a.__class__\n",
        "print": "def f(a):\n    print(a)\n    return a\n",
        "two functions": "def f(a):\n    return a\ndef g(a):\n    return a\n",
        "too big": "def f(a):\n" + "".join(f"    a = a + {i}\n" for i in range(60)) + "    return a\n",
        "string literal": "def f(a):\n    return 'x'\n",
    }
    for k, src in bad.items():
        check(not vocab.check_operation(src)[0], f"invented operation rejected: {k}")


def test_chess():
    d = vocab.get("chess")
    fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1",           # white rook attacks an undefended queen
            "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"]         # back-rank mate available (Rd8#)
    ctx = vocab.featurize(fens)
    v = vocab.evaluate("(count own_P)", d, ctx)
    check(list(v) == [8, 8, 0, 3], f"own pawns counted: {list(v)}")
    v = vocab.evaluate("(count hanging_opp)", d, ctx)
    check(v[2] == 1, f"the undefended attacked queen is hanging: {list(v)}")
    v = vocab.evaluate("(max check_from_us)", d, ctx)
    check(v[3] == 1 and v[0] == 0, f"a checking square exists in the back-rank position: {list(v)}")
    v = vocab.evaluate("(count (and own_K (king_any own_P)))", d, ctx)
    check(v[3] == 1, f"spatial operation: the king next to its own pawns: {list(v)}")
    v = vocab.evaluate("(sum (north own_P))", d, ctx)
    check(v[0] == 8, f"north shifts toward the opponent: {list(v)}")
    v = vocab.evaluate("n_legal", d, ctx)
    check(v[0] == 20, f"20 legal moves at the start: {list(v)}")


def test_seq():
    d = vocab.get("seq")
    x = np.array([[3, 9, 1, 7], [5, 2, 8, 6]], float)
    ctx = {"x": x}
    vals = chain(d, ctx, [("r", "Seq", "(rank x)"), ("second", "Num", "(argmax (eq @r (bcast (add n -2))))")])
    check(list(vals["second"]) == [3, 3], f"second-largest position (AlgZoo's task) as two parts: {list(vals['second'])}")
    v = vocab.evaluate("(sum (gt x (bcast (mean x))))", d, ctx)
    check(list(v) == [2, 2], f"count above the mean: {list(v)}")
    v = vocab.evaluate("(at (cummax x) 2)", d, ctx)
    check(list(v) == [9, 8], f"prefix maximum: {list(v)}")


def test_tok_against_tracr():
    d = vocab.get("tok")
    seqs = [["a", "x", "b", "x"], ["x", "x", "a"], ["b", "a", "c", "a", "b"]]
    ctx = {"tokens": seqs}
    # S0-style circuits as chains of parts (each part one RASP step), checked against Python ground truth
    v = chain(d, ctx, [("length", "SOp", "(selector_width (select tokens tokens TRUE))")])
    check(v["length"] == [[len(s)] * len(s) for s in seqs], "length")
    v = chain(d, ctx, [("hist", "SOp", "(selector_width (select tokens tokens EQ))")])
    check(v["hist"] == [[s.count(t) for t in s] for s in seqs], "histogram")
    v = chain(d, ctx, [("is_x", "SOp", '(map (eqc "x") tokens)'),
                       ("frac", "SOp", "(aggregate (select indices indices LEQ) @is_x)")])
    truth = [[sum(t == "x" for t in s[:i + 1]) / (i + 1) for i in range(len(s))] for s in seqs]
    check(all(np.allclose(a, b) for a, b in zip(v["frac"], truth)), "fraction of previous x tokens")
    v = chain(d, ctx, [("length", "SOp", "(selector_width (select tokens tokens TRUE))"),
                       ("opp", "SOp", "(map dec (sequence_map minus @length indices))"),
                       ("rev", "SOp", "(aggregate_cat (select indices @opp EQ) tokens)")])
    check(v["rev"] == [s[::-1] for s in seqs], "reverse")
    try:
        from tracr.compiler import compiling, lib
        from tracr.rasp import rasp
    except ImportError:
        check(False, "tracr importable for the S0 cross-check")
        return
    progs = {"length": lib.make_length(), "hist": lib.make_hist(),
             "frac": lib.make_frac_prevs(rasp.numerical(rasp.Map(lambda t: t == "x", rasp.tokens))),
             "rev": lib.make_reverse(rasp.tokens)}
    ours = {"length": [("length", "SOp", "(selector_width (select tokens tokens TRUE))")],
            "hist": [("hist", "SOp", "(selector_width (select tokens tokens EQ))")],
            "frac": [("is_x", "SOp", '(map (eqc "x") tokens)'), ("frac", "SOp", "(aggregate (select indices indices LEQ) @is_x)")],
            "rev": [("length", "SOp", "(selector_width (select tokens tokens TRUE))"),
                    ("opp", "SOp", "(map dec (sequence_map minus @length indices))"),
                    ("rev", "SOp", "(aggregate_cat (select indices @opp EQ) tokens)")]}
    for name, prog in progs.items():
        model = compiling.compile_rasp_to_model(prog, vocab={"a", "b", "c", "x"}, max_seq_len=6, compiler_bos="BOS")
        got = [model.apply(["BOS"] + s).decoded[1:] for s in seqs]
        mine = chain(d, ctx, ours[name])[ours[name][-1][0]]
        same = all(np.allclose(np.asarray(g, float), np.asarray(m, float)) if name == "frac" else list(g) == list(m)
                   for g, m in zip(got, mine))
        check(same, f"S0 cross-check: the vocabulary's {name} equals Tracr's compiled model on every test sequence")


def test_fluid_weights():
    d = vocab.get("fluid")
    n, H = 2, 16
    yy, xx = np.meshgrid(np.arange(H), np.arange(H), indexing="ij")
    u = np.stack([np.sin(2 * np.pi * yy / H)] * n); v = np.zeros_like(u)
    ctx = {"u": u, "v": v, "p": np.zeros_like(u), "q": np.stack([np.cos(2 * np.pi * xx / H)] * n),
           "u_prev": u, "v_prev": v, "q_prev": np.zeros_like(u)}
    div = vocab.evaluate("(rms (div u v))", d, ctx)
    check(np.allclose(div, 0, atol=1e-9), "fluid: a shear flow is divergence-free")
    vort = vocab.evaluate("(rms (curl u v))", d, ctx)
    check(np.all(vort > 0.1), "fluid: a shear flow has vorticity")
    adv = vocab.evaluate("(rms (adv u v q))", d, ctx)
    check(np.all(adv > 0.05), "fluid: advection of q by u")
    w = vocab.get("weights")
    rng = np.random.default_rng(0)
    nets = [{"W": [np.diag([3.0, 2.0, 0.1]), rng.normal(size=(2, 3))], "b": [np.zeros(3), np.zeros(2)]},
            {"W": [np.eye(3) * 2, np.eye(3)], "b": [np.ones(3), np.zeros(3)]}]
    ctxw = {"nets": nets}
    s = vocab.evaluate("(spec Wprod)", w, ctxw)
    check(np.isclose(s[1], 2.0), f"weights: top singular value of the end-to-end map: {s}")
    r = vocab.evaluate("(rank W0 0.5)", w, ctxw)
    check(r[0] == 2 and r[1] == 3, f"weights: rank above a threshold: {r}")
    check(vocab.evaluate("nlayers", w, ctxw).tolist() == [2, 2], "weights: layer count")


def test_counts():
    for name in vocab.DOMAINS:
        d = vocab.get(name)
        c = vocab.count(d)
        print(f"     {name}: L={d.L}, {c['total']:,} expression skeletons of length <= L")
        check(c["total"] > 0, f"{name}: counted")


if __name__ == "__main__":
    for t in (test_grammar, test_operations, test_chess, test_seq, test_tok_against_tracr, test_fluid_weights, test_counts):
        print(f"== {t.__name__}")
        t()
    print(f"\n{len(FAILS)} failures")
    sys.exit(1 if FAILS else 0)
