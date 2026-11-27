"""The verifier hook (Gate 1; BENCH_BRIEF.md §6): the scorer feeds the builder's checker the sealed true blueprints
and the S5a accounts, and it prints only counts per tier: "certified X of N true blueprints, rejected Y of M
accounts". Nothing about any blueprint or account is printed or written.

Provisional formats until `bench/accept/INTERFACE.md` exists (then `load_task` and `load_blueprint` are rewritten to
it, and these provisional formats are dropped):
- a task folder holds `task.json`: {"adapter": "tracr_lib" | "algzoo", ...adapter arguments, "run_alone_line": x},
  and `inputs.json` (token sequences) or `inputs.npy` (numbers);
- a blueprint is JSON: {"domain", "output", "library", "parts": [{"id", "type", "vtype", "site", "labels" or
  "basis", "encoding", "scale", "definition", "table", "upstream"}]}; an account that is not such a file is read as
  text and offered as it is.
Usage: python3 -m box.hook <tier> <task folder> <true blueprints folder> <accounts folder>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box.blueprint import Blueprint, Part  # noqa: E402


def load_blueprint(path):
    if not path.endswith(".json"):
        return open(path).read()            # offered as it is: the checker decides
    try:
        d = json.load(open(path))
        parts = [Part(p["id"], p["type"], p["vtype"], p["site"], labels=p.get("labels"),
                      basis=None if p.get("basis") is None else np.asarray(p["basis"], float),
                      encoding=p.get("encoding", "categorical"), scale=p.get("scale", 1.0),
                      definition=p.get("definition"), table=p.get("table"), upstream=p.get("upstream", []))
                 for p in d["parts"]]
        return Blueprint(d["domain"], parts, d["output"], library=d.get("library", {}), name=os.path.basename(path))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return open(path).read()


def dump_blueprint(bp, path):
    json.dump({"domain": bp.domain, "output": bp.output, "library": bp.library,
               "parts": [{"id": p.id, "type": p.type, "vtype": p.vtype, "site": p.site, "labels": p.labels,
                          "basis": None if p.basis is None else np.asarray(p.basis).tolist(), "encoding": p.encoding,
                          "scale": p.scale, "definition": p.definition, "table": p.table, "upstream": p.upstream}
                         for p in bp.parts]}, open(path, "w"))


def load_task(folder):
    t = json.load(open(os.path.join(folder, "task.json")))
    if t["adapter"] == "tracr_lib":
        from tracr.compiler import lib
        from tracr.rasp import rasp
        from box.adapters.tracr_net import TracrNet
        from box.verify import Checker
        programs = {"frac_prevs": lambda: lib.make_frac_prevs(rasp.numerical(rasp.Map(lambda x: x == "x", rasp.tokens)).named("is_x")).named("frac"),
                    "hist": lambda: lib.make_hist().named("hist"),
                    "reverse": lambda: lib.make_reverse(rasp.tokens).named("reverse"),
                    "sort_unique": lambda: lib.make_sort_unique(rasp.tokens, rasp.tokens).named("sort"),
                    "pair_balance": lambda: lib.make_pair_balance(rasp.tokens, "(", ")").named("pair_balance")}
        net = TracrNet(programs[t["program"]](), t["vocab"], t["max_seq_len"])
        X = json.load(open(os.path.join(folder, "inputs.json")))
        return Checker(net, X, seed=t.get("seed", 0), n_pairs=t.get("n_pairs", 120), control_reps=t.get("control_reps", 300)), t
    if t["adapter"] == "algzoo":
        from box.adapters.algzoo_rnn import AlgZooRNN
        from box.verify_vec import VecChecker
        X = np.load(os.path.join(folder, "inputs.npy"))
        net = AlgZooRNN(t["target"])
        return VecChecker(net, X, test_idx=t.get("test_idx"), seed=t.get("seed", 0)), t
    raise ValueError(f"unknown adapter {t['adapter']}")


def run(tier, task_folder, true_folder, accounts_folder):
    chk, t = load_task(task_folder)
    line = float(t["run_alone_line"])
    true = [load_blueprint(os.path.join(true_folder, f)) for f in sorted(os.listdir(true_folder))] if true_folder else []
    accounts = [load_blueprint(os.path.join(accounts_folder, f)) for f in sorted(os.listdir(accounts_folder))] if accounts_folder else []
    certified = sum(chk.certify(bp, line)["verdict"] == "certified" for bp in true)
    rejected = sum(chk.certify(a, line)["verdict"] == "rejected" for a in accounts)
    print(f"{tier}: certified {certified} of {len(true)} true blueprints, rejected {rejected} of {len(accounts)} accounts")
    return certified, len(true), rejected, len(accounts)


if __name__ == "__main__":
    a = sys.argv[1:] + [None] * (4 - len(sys.argv[1:]))
    run(*a[:4])
