"""Run a blueprint alone (INTERFACE.md §6): a separate process given a folder with bp.json, task.json (metadata only:
id, domain, behaviour, sites), ctx.json (the inputs) and given.json (resampled values for lookups off-table and for
unresolved and irrelevant parts). It never receives the network's weights. Writes out.json: every part's values."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "vendor"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import core  # noqa: E402


def main(folder):
    bp = json.load(open(os.path.join(folder, "bp.json")))
    task = json.load(open(os.path.join(folder, "task.json")))
    ctx = json.load(open(os.path.join(folder, "ctx.json")))
    given = json.load(open(os.path.join(folder, "given.json")))
    if task["domain"] != "tok":
        ctx = {k: np.asarray(v) for k, v in ctx.items()} if task["domain"] != "chess" else ctx
        given = {k: (np.asarray(v, dtype=object) if v is not None else None) for k, v in given.items()}
        given = {k: (np.asarray(v.tolist(), float) if v is not None and _is_num(v) else v) for k, v in given.items()}
    vals = core.evaluate_blueprint(bp, task, ctx, given=given)
    json.dump({"values": {k: core._jsonable(v) for k, v in vals.items()}}, open(os.path.join(folder, "out.json"), "w"))


def _is_num(v):
    try:
        np.asarray(v.tolist(), float)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main(sys.argv[1])
