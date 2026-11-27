"""The executable contract between the box and the acceptance suite (bench/accept/INTERFACE.md).

Interface version bench-accept/v1. Everything the box hands the scorer, and everything the bench hands the box, has
a typed form here. The scorer of record applies exactly these checks (`validate_task`, `validate_blueprint`); the box
may import this module to check its own output before handing it over.

Nothing in this file is sealed. It imports only the standard library, numpy, and the bench's frozen copy of the
starting vocabulary (bench/accept/vendor/vocab, vocabulary v2 at commit f0b3b680).
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, TypedDict, Union

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(HERE, "vendor") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "vendor"))
import vocab as _vocab  # noqa: E402  (the bench's frozen copy: bench/accept/vendor/vocab)

INTERFACE_VERSION = "bench-accept/v1"

# ------------------------------------------------------------------------------------------------ fixed numbers
MAX_CONSTANTS = 5                    # per definition, after expanding every invented primitive (SPEC.md §6)
MAX_OPERATION_NODES = 200            # AST nodes of one invented operation (BENCH_BRIEF.md §6), fixed before S2
OVERVIEW_MAX_CHARS = 4000            # "one page"
PART_TYPES = ("rule", "lookup", "irrelevant", "unresolved")
EVIDENCE_KINDS = ("exhaustive", "proven_bound", "statistical", "adversarial_search")
MAP_CLASSES = ("subspace", "dictionary_feature", "param_component")   # a direction is a subspace with k = 1
ENCODINGS = ("numerical", "categorical", "vector")
CLAIM_KINDS = ("type", "rule_holds", "iia_pass", "substitution_pass", "irrelevant_pass", "lookup_pass")
ORTHONORMAL_TOL = 1e-6


# ================================================================================================ 1. the task folder
class SiteSpec(TypedDict, total=False):
    name: str                        # e.g. "resid_3", "h", "block_6"
    shape: List[int]                 # activation shape per input, excluding the batch axis: [positions, features]
    stream: Optional[str]            # sites sharing one residual stream (same basis) carry the same stream name
    order: int                       # position of the site in the forward pass (0 = first)
    value_offset: int                # position of value index 0 (1 for token tasks with a BOS position, else 0)
    labels: Optional[List[str]]      # names of the feature axis's basis directions, where the bench publishes them


class ParamSpec(TypedDict):
    name: str                        # e.g. "layer_0/mlp/linear_1/w"
    shape: List[int]


class OutputSpec(TypedDict, total=False):
    name: str                        # the output's name ("out")
    kind: Literal["categorical", "numerical", "distribution"]
    values: Optional[List[Any]]      # categorical: the output values, in decoder order
    positions: Literal["per_position", "per_input"]


class BehaviourSpec(TypedDict, total=False):
    description: str                 # in the task's own terms
    outputs: List[OutputSpec]
    metric: Literal["accuracy", "r2", "kl"]
    reference_rule: Optional[str]    # plain-words statement of the task rule, where there is one
    decision: Dict[str, Any]         # how a decision is read from the output: {"kind": "argmax"} | {"kind": "sign"}
                                     # | {"kind": "threshold", "value": x}


class TaskSpec(TypedDict, total=False):
    interface: str                   # INTERFACE_VERSION
    id: str                          # network id within the draw, e.g. "S0-03"
    draw: str                        # draw id
    tier: str                        # S0 .. S7
    domain: str                      # a vocabulary domain: chess | seq | tok | fluid | weights
    network: Dict[str, Any]          # {"format": "tracr_np" | "rnn_relu" | "attn_only" | ..., "path": "network",
                                     #  "n_params": int, "multiply_adds": int}
    behaviour: BehaviourSpec
    inputs: Dict[str, Any]           # {"path": "inputs.json" | "inputs.npz", "n": int}
    train_items: Optional[Dict[str, Any]]   # {"path": "train_items.json", "n": int} where known, else null
    sites: List[SiteSpec]
    params: List[ParamSpec]
    notes: str


def load_task(task_dir: str) -> TaskSpec:
    with open(os.path.join(task_dir, "task.json")) as f:
        return json.load(f)


def load_inputs(task_dir: str, task: Optional[TaskSpec] = None):
    """The task's inputs as the vocabulary domain's context: tok -> {"tokens": [[...]]}; seq -> {"x": [N, n]};
    chess -> {"fens": [...]} (call vocab.featurize for features); fluid -> {field: [N, H, W]}; weights -> {"nets"}."""
    task = task or load_task(task_dir)
    path = os.path.join(task_dir, task["inputs"]["path"])
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    if path.endswith(".npz"):
        return {k: v for k, v in np.load(path, allow_pickle=False).items()}
    raise ValueError(f"unknown inputs file {path}")


def canonical_key(domain: str, ctx: Dict[str, Any], i: int) -> str:
    """The key a lookup table uses for input i: the input itself, in a canonical text form."""
    if domain == "tok":
        return " ".join(str(t) for t in ctx["tokens"][i])
    if domain == "seq":
        return ",".join(repr(float(v)) for v in np.asarray(ctx["x"][i]).ravel())
    if domain == "chess":
        return str(ctx["fens"][i])
    raise ValueError(f"lookup keys are not defined for domain {domain}; use train-item ids")


# ================================================================================================ 2. the network
Edit = Callable[[np.ndarray], np.ndarray]      # activation [N, *site shape] -> activation of the same shape


class RunResult(TypedDict):
    outputs: Any                     # decoded outputs, one per input (a list per position for per-position outputs)
    raw: np.ndarray                  # the output vector(s) before decoding
    sites: Dict[str, np.ndarray]     # site name -> activations [N, *site shape] (only when asked)


class Network(Protocol):
    """What the task folder's loader returns (bench/accept/netlib.py: `load_network(task_dir)`)."""
    task: TaskSpec
    sites: List[SiteSpec]

    def run(self, ctx: Dict[str, Any], edits: Optional[Dict[str, Edit]] = None,
            param_edits: Optional[Dict[str, np.ndarray]] = None, return_sites: bool = False) -> RunResult: ...


# ================================================================================================ 3. the blueprint
class MapSubspace(TypedDict, total=False):
    cls: Literal["subspace"]         # written "class" in JSON
    basis: List[List[float]]         # [d, k], orthonormal columns; or
    labels: List[str]                # names from the site's published labels (each one basis direction)


class MapDictionaryFeature(TypedDict):
    cls: Literal["dictionary_feature"]
    dictionary: str                  # path, relative to the blueprint, of an .npz with W_enc [d, m], b_enc [m],
                                     # W_dec [m, d], b_dec [d]
    feature: int


class MapParamComponent(TypedDict):
    cls: Literal["param_component"]
    param: str                       # a parameter named in the task's "params"
    decomposition: str               # path of an .npz holding "components" [C, *param shape] summing to the param
    component: int


class Location(TypedDict, total=False):
    site: str                        # a site named in the task
    positions: Union[Literal["all", "flat"], List[int]]
    # "all": one value per position (Seq, SOp, Sq values), value index i read at position value_offset + i;
    # [p]: one value per input, read at position p; "flat": one value per input, read from the flattened site
    map: Dict[str, Any]              # one of the three map classes, with "class" set


class Encoding(TypedDict, total=False):
    kind: Literal["numerical", "categorical", "vector"]
    scale: float                     # numerical (k = 1): value = scale * coordinate + offset
    offset: float
    values: List[Any]                # categorical: one basis direction per value, value = values[argmax]
    magnitude: float                 # categorical: coordinate written for the chosen value (default 1)


class Claim(TypedDict):
    claim: str                       # one of CLAIM_KINDS
    value: Any                       # e.g. the type claimed, or True
    p: float                         # stated probability that the claim holds
    evidence: str                    # one of EVIDENCE_KINDS


class Table(TypedDict, total=False):
    entries: List[List[Any]]         # [[key, value], ...]; key = canonical_key(...) or a train-item id
    keys_are: Literal["inputs", "train_items"]
    default: Any                     # value on inputs not in the table (else the scorer resamples it)


class Part(TypedDict, total=False):
    id: str
    type: str                        # one of PART_TYPES
    vtype: str                       # the vocabulary type of its value (SOp, Seq, Num, Sq, Pos, Field, ...)
    location: Location               # required for every part: every variable is located (R3)
    encoding: Encoding
    definition: Optional[str]        # rule: an S-expression of the vocabulary (VOCABULARY.md §1)
    table: Optional[Table]           # lookup
    upstream: List[str]              # ids of the parts its definition uses (every @id, and nothing else)
    choice: Optional[str]            # set when the part computes the network's own choice: the output name
    claims: List[Claim]
    evidence: Dict[str, str]         # claim name -> evidence kind


class Blueprint(TypedDict, total=False):
    interface: str
    task: str                        # the task id it explains
    domain: str
    library: Dict[str, Dict[str, Any]]   # $name -> {"kind": "expression", "definition"} | {"kind": "operation",
                                         # "source", "args", "ret"}, each with "provenance": {"run", "evidence"}
    parts: List[Part]
    output: str                      # the part whose value is the network's output
    overview: Dict[str, Any]         # {"text": <= OVERVIEW_MAX_CHARS, "predictor": an S-expression over @parts}
    systems: List[Dict[str, Any]]    # [{"id", "role", "parts": [ids], "fed_by": [ids], "feeds": [ids]}]
    wiring: List[Dict[str, Any]]     # [{"from", "to", "strength"}] (optional; upstream lists are binding)
    ledger: Dict[str, Any]           # {"shares": {type: share}, "unresolved_by_site": {site: share}}
    rivals: List[Dict[str, Any]]     # [{"path": "rival_1.json", "separating_experiment": text}]
    cost_map: Optional[Dict[str, Any]]   # R15: {"reference", "steps": [{"step", "parts", "flops", "error"}], "skipped"}
    rendering: Optional[str]         # path of the reader rendering's index (index.md), relative to the blueprint
    usage: Optional[Dict[str, float]]    # {"gpu_hours", "model_tokens"} this run used (R13)


def load_blueprint(path: str) -> Union[Blueprint, str]:
    """A blueprint file, or the raw text if it is not one (the format check then rejects it)."""
    with open(path) as f:
        text = f.read()
    try:
        bp = json.loads(text)
    except json.JSONDecodeError:
        return text
    return bp if isinstance(bp, dict) else text


# ================================================================================================ 4. the box
class EstimateQuestion(TypedDict):
    id: str
    quantity: str                    # "accuracy_vs_rule" | "mean_output" | "decision_rate"
    distribution: Dict[str, Any]     # a generator the bench names, with its parameters
    subpopulation: Optional[Dict[str, Any]]


class EstimateAnswer(TypedDict):
    id: str
    estimate: float
    ops: int                         # floating-point operations the estimator itself counted


class Decision(TypedDict):
    id: str
    input: int                       # index into the inputs given with the request
    output: str


class TraceAnswer(TypedDict, total=False):
    id: str
    contributions: Dict[str, float]  # part id -> contribution to the decision
    flip: Dict[str, Any]             # {"kind": "input", "changes": [[element, new value], ...]}
                                     # | {"kind": "part", "part": id, "value": v}
    p: float                         # stated probability that the flip changes the decision


class EditRequest(TypedDict):
    id: str
    capability: str                  # plain words: "switch off look-ahead", "remove the gate"
    kind: Literal["remove", "strengthen", "redirect"]


class EditAction(TypedDict, total=False):
    part: str
    op: Literal["resample", "mean", "zero", "scale", "set"]
    factor: float
    value: Any


class EditPlan(TypedDict, total=False):
    id: str
    parts: List[str]
    actions: List[EditAction]        # applied by the scorer at each part's location in the network
    predicted: Dict[str, Any]        # {"decisions": [one per evaluation input], "p": stated probability per input}


class Detector(TypedDict):
    parts: List[str]                 # read through their locations
    weights: List[float]
    bias: float
    threshold: float                 # fires where weights . values + bias > threshold


class Box(Protocol):
    """The builder's entry points. The start configuration names the class ("module:Class") and its single config;
    both are used unchanged for every network and for certification (R1, R14)."""

    def __init__(self, config: Dict[str, Any]) -> None: ...

    def explain(self, task_dir: str, out_dir: str) -> str:
        """Write out_dir/blueprint.json (plus rendering and rivals); return its path. May run the network."""

    def estimate(self, blueprint_path: str, question: EstimateQuestion) -> EstimateAnswer:
        """R9: answer without running the network. Called in a process without the network's weights."""

    def trace(self, blueprint_path: str, ctx: Dict[str, Any], decisions: List[Decision]) -> List[TraceAnswer]:
        """R7: per decision, each part's contribution and a predicted minimal flip."""

    def edit(self, blueprint_path: str, ctx: Dict[str, Any], requests: List[EditRequest]) -> List[EditPlan]:
        """R11: map each capability edit to parts and predict its effect on ctx's inputs."""

    def fingerprint(self, blueprint_path: str, behaviour: str) -> Detector:
        """R7: a detector of a named behaviour, over the blueprint's parts."""

    def verify(self, task_dir: str, blueprint_path: str) -> Literal["certified", "rejected"]:
        """Gate 1's verifier hook: the builder's checker on one account. Never prints or writes the account."""


class StartConfig(TypedDict):
    entry: str                       # "module:Class" implementing Box
    config: Dict[str, Any]
    budget: Dict[str, Dict[str, float]]   # {"gpu_hours": {"base": a, "per_param": b}, "model_tokens": {...}}


# ================================================================================================ 5. the format check
def _err(errs, msg):
    errs.append(msg)


def check_location(loc: Any, task: TaskSpec, bp_dir: str = ".") -> List[str]:
    """Every location is a named site plus an allowed linear map of a declared dimension (SPEC R3)."""
    errs: List[str] = []
    if not isinstance(loc, dict):
        return ["location is not an object"]
    sites = {s["name"]: s for s in task.get("sites", [])}
    site = sites.get(loc.get("site"))
    if site is None:
        return [f"unknown site {loc.get('site')!r}"]
    pos = loc.get("positions", "all")
    npos = site["shape"][0] if len(site["shape"]) > 1 else 1
    if isinstance(pos, list):
        if len(pos) != 1 or not all(isinstance(p, int) and 0 <= p < npos for p in pos):
            _err(errs, f"positions {pos!r}: one position within 0..{npos - 1}")
    elif pos not in ("all", "flat"):
        _err(errs, f"positions {pos!r}: 'all', 'flat' or [p]")
    m = loc.get("map")
    if not isinstance(m, dict) or m.get("class") not in MAP_CLASSES:
        return errs + [f"map class must be one of {MAP_CLASSES}; anything else is rejected (unrestricted maps make any "
                       f"network implement any algorithm)"]
    d = int(np.prod(site["shape"])) if pos == "flat" else site["shape"][-1]
    if m["class"] == "subspace":
        if "labels" in m:
            labels = site.get("labels") or []
            bad = [l for l in m["labels"] if l not in labels]
            if not m["labels"] or bad:
                _err(errs, f"subspace labels not published for site {site['name']}: {bad[:3]}")
        elif "basis" in m:
            B = np.asarray(m["basis"], float)
            if B.ndim != 2 or B.shape[0] != d or B.shape[1] < 1 or B.shape[1] > d:
                _err(errs, f"basis shape {B.shape}: must be [{d}, k] with 1 <= k <= {d}")
            elif not np.allclose(B.T @ B, np.eye(B.shape[1]), atol=ORTHONORMAL_TOL):
                _err(errs, "basis columns are not orthonormal")
        else:
            _err(errs, "subspace map needs 'basis' or 'labels'")
    elif m["class"] == "dictionary_feature":
        p = os.path.join(bp_dir, str(m.get("dictionary", "")))
        if not os.path.isfile(p):
            _err(errs, f"dictionary file missing: {m.get('dictionary')}")
        else:
            D = np.load(p)
            for k, shp in (("W_enc", (d, None)), ("b_enc", (None,)), ("W_dec", (None, d)), ("b_dec", (d,))):
                if k not in D:
                    _err(errs, f"dictionary lacks {k}")
            if not errs and not (0 <= int(m.get("feature", -1)) < D["W_enc"].shape[1]):
                _err(errs, "dictionary feature index out of range")
    elif m["class"] == "param_component":
        params = {p["name"]: p for p in task.get("params", [])}
        if m.get("param") not in params:
            _err(errs, f"unknown parameter {m.get('param')!r}")
        p = os.path.join(bp_dir, str(m.get("decomposition", "")))
        if not os.path.isfile(p):
            _err(errs, f"decomposition file missing: {m.get('decomposition')}")
    return errs


def location_dim(loc: Dict[str, Any], task: TaskSpec) -> int:
    m = loc["map"]
    if m["class"] == "subspace":
        return len(m["labels"]) if "labels" in m else int(np.asarray(m["basis"]).shape[1])
    return 1


def validate_blueprint(bp: Any, task: TaskSpec, bp_dir: str = ".") -> List[str]:
    """The format check. Returns the list of problems; an empty list means the blueprint is well formed.
    An account that fails here is not scored further, and failing here never counts as a must-fail rejection."""
    errs: List[str] = []
    if not isinstance(bp, dict):
        return ["not a blueprint: an account must be a JSON blueprint of located, typed parts (INTERFACE.md §3)"]
    if bp.get("interface") != INTERFACE_VERSION:
        _err(errs, f"interface must be {INTERFACE_VERSION!r}")
    if bp.get("domain") != task.get("domain"):
        _err(errs, f"domain {bp.get('domain')!r} differs from the task's {task.get('domain')!r}")
        return errs
    dom = _vocab.get(task["domain"])
    library = bp.get("library") or {}
    lib_copy = json.loads(json.dumps(library))
    ok, why = _vocab.check_library(dom, lib_copy)
    if not ok:
        _err(errs, f"library: {why}")
    for name, e in library.items():
        if e.get("kind") == "operation":
            okop, whyop, nodes, _lits = _vocab.check_operation(e.get("source", ""), max_nodes=MAX_OPERATION_NODES)
            if not okop:
                _err(errs, f"${name}: {whyop}")
        if not isinstance(e.get("provenance"), dict):
            _err(errs, f"${name}: provenance missing (which run made it, from which evidence)")
    parts = bp.get("parts")
    if not isinstance(parts, list) or not parts:
        return errs + ["parts: a non-empty list is required"]
    ids = [p.get("id") for p in parts if isinstance(p, dict)]
    if len(ids) != len(parts) or len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        return errs + ["parts: every part is an object with a unique, non-empty id"]
    vtypes = {p["id"]: p.get("vtype") for p in parts}
    for p in parts:
        pid = p["id"]
        t = p.get("type")
        if t not in PART_TYPES:
            _err(errs, f"{pid}: type {t!r} not one of {PART_TYPES}")
        if p.get("vtype") not in dom.types:
            _err(errs, f"{pid}: vtype {p.get('vtype')!r} not a type of domain {dom.name}")
        if "location" not in p:
            _err(errs, f"{pid}: every part is located (R3)")
        else:
            errs += [f"{pid}: {e}" for e in check_location(p["location"], task, bp_dir)]
        enc = p.get("encoding") or {}
        if enc.get("kind") not in ENCODINGS:
            _err(errs, f"{pid}: encoding kind must be one of {ENCODINGS}")
        elif "location" in p and isinstance(p["location"], dict) and isinstance(p["location"].get("map"), dict) \
                and not any(e.startswith(f"{pid}:") for e in errs):
            k = location_dim(p["location"], task)
            if enc["kind"] == "numerical" and k != 1:
                _err(errs, f"{pid}: a numerical encoding needs a location of dimension 1 (got {k})")
            if enc["kind"] == "categorical" and len(enc.get("values", [])) != k:
                _err(errs, f"{pid}: a categorical encoding needs one value per basis direction ({k})")
        ups = p.get("upstream", [])
        if not isinstance(ups, list) or any(u not in vtypes for u in ups):
            _err(errs, f"{pid}: upstream must list existing part ids")
            ups = [u for u in ups if u in vtypes] if isinstance(ups, list) else []
        if t == "rule":
            text = p.get("definition")
            if not isinstance(text, str):
                _err(errs, f"{pid}: a rule needs a definition")
            else:
                okd, whyd, typ = _vocab.admissible(text, dom, {u: vtypes[u] for u in ups}, lib_copy)
                if not okd:
                    _err(errs, f"{pid}: definition not admissible: {whyd}")
                elif typ != p.get("vtype"):
                    _err(errs, f"{pid}: definition has type {typ}, part declares {p.get('vtype')}")
                refs = set(_refs(text))
                if refs != set(ups):
                    _err(errs, f"{pid}: upstream must be exactly the parts its definition uses")
        if t == "lookup":
            tab = p.get("table")
            if not isinstance(tab, dict) or not isinstance(tab.get("entries"), list) or not tab["entries"]:
                _err(errs, f"{pid}: a lookup lists its table (entries)")
            elif tab.get("keys_are") not in ("inputs", "train_items"):
                _err(errs, f"{pid}: table keys_are must be 'inputs' or 'train_items'")
        if p.get("choice") is not None and p.get("choice") not in [o["name"] for o in task["behaviour"]["outputs"]]:
            _err(errs, f"{pid}: choice must name one of the task's outputs")
        for c in p.get("claims", []) or []:
            if c.get("claim") not in CLAIM_KINDS or not (0.0 <= float(c.get("p", -1)) <= 1.0) \
                    or c.get("evidence") not in EVIDENCE_KINDS:
                _err(errs, f"{pid}: claim {c!r} needs a claim kind, a probability in [0, 1] and an evidence kind")
    if bp.get("output") not in vtypes:
        _err(errs, "output must name a part")
    if not errs:
        try:
            order_parts(parts)
        except ValueError as e:
            _err(errs, str(e))
    ov = bp.get("overview")
    if ov is not None:
        if len(str(ov.get("text", ""))) > OVERVIEW_MAX_CHARS:
            _err(errs, f"overview text over {OVERVIEW_MAX_CHARS} characters (one page)")
        pred = ov.get("predictor")
        if pred is not None:
            okp, whyp, _ = _vocab.admissible(pred, dom, {u: vtypes[u] for u in _refs(pred) if u in vtypes}, lib_copy)
            if not okp or any(u not in vtypes for u in _refs(pred)):
                _err(errs, f"overview predictor not admissible: {whyp}")
    return errs


def _refs(text: str) -> List[str]:
    import re
    return re.findall(r"@([A-Za-z0-9_.:\-]+)", text)


def order_parts(parts: List[Part]) -> List[Part]:
    """Parts in dependency order; raises ValueError on a cycle."""
    by_id = {p["id"]: p for p in parts}
    done, out, pending = set(), [], list(parts)
    while pending:
        moved = False
        for p in list(pending):
            if all(u in done for u in p.get("upstream", [])):
                out.append(p); done.add(p["id"]); pending.remove(p); moved = True
        if not moved:
            raise ValueError("the wiring has a cycle: " + ", ".join(p["id"] for p in pending))
    assert len(out) == len(by_id)
    return out


def budget_for(start: StartConfig, n_params: int) -> Dict[str, float]:
    """R13: the budget the start configuration states, as a function of the network's size."""
    return {k: float(v.get("base", 0.0)) + float(v.get("per_param", 0.0)) * n_params
            for k, v in start.get("budget", {}).items()}


if __name__ == "__main__":
    # python3 bench/accept/interface.py <task folder> <blueprint.json>: the format check, as the scorer applies it
    tdir, bpath = sys.argv[1], sys.argv[2]
    problems = validate_blueprint(load_blueprint(bpath), load_task(tdir), os.path.dirname(os.path.abspath(bpath)))
    print("format: ok" if not problems else "format: FAIL\n  " + "\n  ".join(problems))
