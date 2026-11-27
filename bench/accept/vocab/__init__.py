"""The starting vocabulary (bench/accept/VOCABULARY.md). Usage:

    import sys; sys.path.insert(0, "bench/accept")
    import vocab
    dom = vocab.get("chess")
    ok, why, typ = vocab.admissible("(count (and own_P (not defended)))", dom)
    value = vocab.evaluate("(count (and own_P (not defended)))", dom, vocab.featurize(fens))
"""
from .core import (Domain, Op, Atom, parse, unparse, length, constants, typecheck, admissible, evaluate, count,
                   check_operation, compile_operation, MAX_CONSTANTS, MAX_OPERATION_NODES, CONST)
from .domains import DOMAINS, get, featurize, POS_NAMES, FLUID_FIELDS, WEIGHT_ATOMS

VERSION = "v1"
