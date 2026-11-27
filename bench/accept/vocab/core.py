"""The starting vocabulary's expression language, shared by every domain (bench/accept/VOCABULARY.md).

A definition is an S-expression over a domain's base features and operations:
    (and (own_P) (not (attacked)))        -> written without the inner parentheses: (and own_P (not attacked))
    (count (ge mob_us 3))
Symbols:
    name        a base feature (an atom) or an operation, by the domain's tables
    3, -0.5     a numeric constant (counts as one symbol and one constant)
    "x"         a token constant, for token sequences (counts as one symbol and one constant)
    @id         a reference to another part of the same blueprint, typed by that part's declared type
    $name       an invented primitive from the box's library (one symbol; expanded when constants are counted)

Length is the number of symbols in the tree. The constants budget (SPEC.md §6) is 5 numeric or token constants per
definition, counted after every $name is expanded. Values are evaluated over a batch: every value carries the inputs
as its first axis.
"""
import ast
import math
import re
from dataclasses import dataclass, field

MAX_CONSTANTS = 5           # SPEC.md §6; only Shreyas changes it
MAX_OPERATION_NODES = 200   # AST nodes in an invented operation (BENCH_BRIEF.md §6 default; INTERFACE.md may fix it)
CONST = "C"                 # the type of a constant literal


@dataclass
class Op:
    name: str
    args: tuple             # argument types
    ret: str                # return type
    fn: object              # fn(*values) -> value
    doc: str = ""


@dataclass
class Atom:
    name: str
    type: str
    fn: object              # fn(ctx) -> value
    doc: str = ""


@dataclass
class Domain:
    name: str
    types: dict             # type name -> meaning
    atoms: dict = field(default_factory=dict)
    ops: dict = field(default_factory=dict)     # name -> list of Op overloads
    roots: tuple = ()       # types a definition may have
    L: int = 0              # maximum definition length
    const_values: str = "number"   # "number" or "token"

    def atom(self, name, typ, fn, doc=""):
        self.atoms[name] = Atom(name, typ, fn, doc)

    def op(self, name, args, ret, fn, doc=""):
        self.ops.setdefault(name, []).append(Op(name, tuple(args), ret, fn, doc))


# ---------------------------------------------------------------------------------------------------- parsing
_TOKEN = re.compile(r'\s*(\(|\)|"[^"]*"|[^\s()]+)')


def parse(text):
    """S-expression text -> tree: a str (symbol), a float/int (number), ("str", s) for a token constant, or a list."""
    toks = _TOKEN.findall(text)
    pos = 0

    def read():
        nonlocal pos
        if pos >= len(toks):
            raise SyntaxError("unexpected end of definition")
        t = toks[pos]; pos += 1
        if t == "(":
            out = []
            while True:
                if pos >= len(toks):
                    raise SyntaxError("missing )")
                if toks[pos] == ")":
                    pos += 1
                    return out
                out.append(read())
        if t == ")":
            raise SyntaxError("unexpected )")
        if t.startswith('"'):
            return ("str", t[1:-1])
        try:
            v = float(t)
            return int(v) if v.is_integer() and "." not in t and "e" not in t.lower() else v
        except ValueError:
            return t
    tree = read()
    if pos != len(toks):
        raise SyntaxError("text after the end of the definition")
    return tree


def unparse(tree):
    if isinstance(tree, list):
        return "(" + " ".join(unparse(t) for t in tree) + ")"
    if isinstance(tree, tuple):
        return '"' + tree[1] + '"'
    return str(tree)


def is_const(t):
    return isinstance(t, (int, float)) or (isinstance(t, tuple) and t[0] == "str")


def length(tree):
    """Symbols in the tree; an invented primitive $name counts as one."""
    if isinstance(tree, list):
        return sum(length(t) for t in tree)
    return 1


def constants(tree, library=None):
    """Constants after expanding every $name through the library (name -> definition text)."""
    library = library or {}
    if isinstance(tree, list):
        return sum(constants(t, library) for t in tree)
    if is_const(tree):
        return 1
    if isinstance(tree, str) and tree.startswith("$"):
        entry = library.get(tree[1:])
        if entry is None:
            raise KeyError(f"unknown invented primitive {tree}")
        if entry.get("kind") == "expression":
            return constants(parse(entry["definition"]), library)
        return entry.get("constants", 0)     # a new operation declares (and is checked for) its literal count
    return 0


# ---------------------------------------------------------------------------------------------------- typing
def _type_of(tree, dom, part_types, library):
    if is_const(tree):
        return CONST
    if isinstance(tree, str):
        if tree.startswith("@"):
            if tree[1:] not in part_types:
                raise TypeError(f"unknown part {tree}")
            return part_types[tree[1:]]
        if tree.startswith("$"):
            e = library.get(tree[1:])
            if e is None:
                raise TypeError(f"unknown invented primitive {tree}")
            if e.get("kind") == "expression":
                return _type_of(parse(e["definition"]), dom, part_types, library)
            if e.get("args"):
                raise TypeError(f"{tree} takes arguments")
            return e["ret"]
        if tree in dom.atoms:
            return dom.atoms[tree].type
        raise TypeError(f"unknown symbol {tree!r} in domain {dom.name}")
    if not tree:
        raise TypeError("empty application")
    head, args = tree[0], tree[1:]
    at = [_type_of(a, dom, part_types, library) for a in args]
    if isinstance(head, str) and head.startswith("$"):
        e = library.get(head[1:])
        if e is None or e.get("kind") != "operation":
            raise TypeError(f"{head} is not an invented operation")
        if tuple(e["args"]) != tuple(at):
            raise TypeError(f"{head} expects {e['args']}, got {at}")
        return e["ret"]
    for o in dom.ops.get(head, []):
        if o.args == tuple(at):
            return o.ret
    raise TypeError(f"no overload of {head!r} for argument types {at} in domain {dom.name}")


def typecheck(text_or_tree, dom, part_types=None, library=None):
    tree = parse(text_or_tree) if isinstance(text_or_tree, str) else text_or_tree
    return _type_of(tree, dom, part_types or {}, library or {})


def admissible(text, dom, part_types=None, library=None):
    """(ok, reason, type): in the grammar, well typed, root type allowed, within L and the constants budget."""
    try:
        tree = parse(text)
        t = typecheck(tree, dom, part_types, library)
    except (SyntaxError, TypeError, KeyError) as e:
        return False, f"not in the grammar: {e}", None
    if dom.roots and t not in dom.roots:
        return False, f"root type {t} not allowed (allowed: {dom.roots})", t
    n, c = length(tree), constants(tree, library)
    if n > dom.L:
        return False, f"length {n} > L={dom.L}", t
    if c > MAX_CONSTANTS:
        return False, f"{c} constants > {MAX_CONSTANTS}", t
    return True, "ok", t


# ---------------------------------------------------------------------------------------------------- evaluation
def evaluate(text_or_tree, dom, ctx, parts=None, part_types=None, library=None, ops_code=None):
    """Value of a definition on a batch. ctx: the domain's input batch; parts: id -> value (same batch);
    part_types: id -> type; library: invented primitives; ops_code: name -> callable for invented operations
    (already checked by check_operation)."""
    tree = parse(text_or_tree) if isinstance(text_or_tree, str) else text_or_tree
    parts, part_types, library, ops_code = parts or {}, part_types or {}, library or {}, ops_code or {}

    def ev(t):
        if is_const(t):
            return t[1] if isinstance(t, tuple) else t
        if isinstance(t, str):
            if t.startswith("@"):
                return parts[t[1:]]
            if t.startswith("$"):
                e = library[t[1:]]
                if e.get("kind") == "expression":
                    return ev(parse(e["definition"]))
                return ops_code[t[1:]]()
            return dom.atoms[t].fn(ctx)
        head, args = t[0], t[1:]
        vals = [ev(a) for a in args]
        if isinstance(head, str) and head.startswith("$"):
            return ops_code[head[1:]](*vals)
        types = tuple(_type_of(a, dom, part_types, library) for a in args)
        for o in dom.ops[head]:
            if o.args == types:
                return o.fn(*vals)
        raise TypeError(f"no overload of {head} for {types}")
    return ev(tree)


# ---------------------------------------------------------------------------------------------------- counting
def count(dom, L=None, max_consts=MAX_CONSTANTS):
    """Number of well-typed expression skeletons of length <= L with a root type in dom.roots, by length.
    A constant is one slot (its value is fitted by whoever searches). Part references and invented primitives are
    not counted: the counts are over the starting vocabulary alone. Argument orders of symmetric operations are
    counted separately, so the counts are upper bounds on distinct expressions."""
    L = L or dom.L
    types = set(dom.types) | {CONST}
    # N[t][n][c] = number of expressions of type t, length n, constants c
    N = {t: [[0] * (max_consts + 1) for _ in range(L + 1)] for t in types}
    for a in dom.atoms.values():
        N[a.type][1][0] += 1
    N[CONST][1][1] += 1
    ops = [o for ov in dom.ops.values() for o in ov]
    for n in range(2, L + 1):
        for o in ops:
            m = len(o.args)
            if m == 0 or n - 1 < m:
                continue
            # distribute n-1 symbols and up to max_consts constants over m arguments
            acc = {(0, 0): 1}
            for i, ta in enumerate(o.args):
                nxt = {}
                rest = m - i - 1
                for (used, cu), ways in acc.items():
                    for k in range(1, n - 1 - used - rest + 1):
                        row = N[ta][k]
                        for c in range(0, max_consts - cu + 1):
                            if row[c]:
                                key = (used + k, cu + c)
                                nxt[key] = nxt.get(key, 0) + ways * row[c]
                acc = nxt
            for (used, cu), ways in acc.items():
                if used == n - 1:
                    N[o.ret][n][cu] += ways
    by_len = [sum(sum(N[t][n]) for t in (dom.roots or types) if t != CONST) for n in range(L + 1)]
    return {"by_length": by_len[1:], "total": sum(by_len)}


# ---------------------------------------------------------------------------------------------------- invented operations
_ALLOWED_NODES = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign,
                  ast.Name, ast.Load, ast.Store, ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
                  ast.IfExp, ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Call, ast.Attribute,
                  ast.Subscript, ast.Slice, ast.Tuple, ast.List, ast.ListComp, ast.GeneratorExp,
                  ast.comprehension, ast.keyword, ast.Pass, ast.Expr,
                  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not,
                  ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.BitAnd, ast.BitOr,
                  ast.Invert, ast.MatMult)
_ALLOWED_CALLS = {"range", "len", "min", "max", "abs", "sum", "float", "int", "bool", "round", "zip", "enumerate",
                  "sorted", "any", "all"}
_FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
                    "delattr", "input", "print", "breakpoint", "memoryview", "net", "network", "model", "weights",
                    "activations", "torch", "os", "sys", "subprocess", "socket", "pickle"}


def check_operation(src, max_nodes=MAX_OPERATION_NODES):
    """An invented operation must be one function of plain arithmetic and numpy (np.*) calls: no imports, no
    input or output, no access to the network, its weights or its activations, at most max_nodes AST nodes.
    Returns (ok, reason, n_nodes, n_literals). Its numeric literals count against the constants budget."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"syntax: {e}", 0, 0
    nodes = list(ast.walk(tree))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(tree.body) != 1 or len(funcs) != 1:
        return False, "must be exactly one function definition", len(nodes), 0
    if len(nodes) > max_nodes:
        return False, f"{len(nodes)} AST nodes > {max_nodes}", len(nodes), 0
    lits = 0
    for n in nodes:
        if not isinstance(n, _ALLOWED_NODES):
            return False, f"node {type(n).__name__} not allowed", len(nodes), 0
        if isinstance(n, ast.Name) and n.id in _FORBIDDEN_NAMES:
            return False, f"name {n.id} not allowed", len(nodes), 0
        if isinstance(n, ast.Attribute):
            if not (isinstance(n.value, ast.Name) and n.value.id == "np") or n.attr.startswith("_") or n.attr in (
                    "load", "save", "loadtxt", "savetxt", "fromfile", "tofile", "memmap", "genfromtxt", "lib"):
                return False, f"attribute {ast.unparse(n)} not allowed (only np.<function>)", len(nodes), 0
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id not in _ALLOWED_CALLS and \
                n.func.id != funcs[0].name:
            return False, f"call to {n.func.id} not allowed", len(nodes), 0
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            lits += 1
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return False, "string literals not allowed", len(nodes), 0
    return True, "ok", len(nodes), lits


def compile_operation(src):
    """Compile a checked invented operation into a callable (numpy only)."""
    import numpy as np
    ok, why, _, _ = check_operation(src)
    if not ok:
        raise ValueError(why)
    env = {"np": np, "__builtins__": {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                                      for k in _ALLOWED_CALLS}}
    tree = ast.parse(src)
    exec(compile(tree, "<invented>", "exec"), env)  # noqa: S102 - checked above: one pure function
    return env[tree.body[0].name]


def human(n):
    return f"{n:.3g}" if n >= 1e6 else str(n)


__all__ = ["Domain", "Op", "Atom", "parse", "unparse", "length", "constants", "typecheck", "admissible", "evaluate",
           "count", "check_operation", "compile_operation", "MAX_CONSTANTS", "MAX_OPERATION_NODES", "CONST", "math"]
