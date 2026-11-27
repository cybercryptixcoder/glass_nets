"""An example of the Box protocol (INTERFACE.md §7), for testing the verifier hook's plumbing. It is NOT the builder's
checker, it explains nothing, and it reads nothing sealed: `verify` uses only the task folder and the account.

verify: certified when the account is well formed, every part typed lookup is not at the decision (a decision that
follows a rule cannot be a table), the blueprint alone reproduces the network on the task's own inputs (at least 0.95),
and every interchange-tested part beats its family-wise random-subspace line; else rejected.

Start configuration for the hook: {"entry": "example_box:ExampleBox", "config": {}, "budget": {}} with
bench/accept/scorer on the Python path.
"""
import os

import numpy as np

import core
import score
from core import I, netlib


class ExampleBox:
    def __init__(self, config):
        self.config = config

    def verify(self, task_dir, blueprint_path):
        task = I.load_task(task_dir)
        bp = I.load_blueprint(blueprint_path)
        bdir = os.path.dirname(os.path.abspath(blueprint_path))
        if I.validate_blueprint(bp, task, bdir):
            return "rejected"
        if any(p["type"] == "lookup" and p["location"]["site"] == "out" for p in bp["parts"]):
            return "rejected"
        st, problems = score.statistics(task_dir, blueprint_path, sealed_dir=None)
        if problems or not np.isfinite(st.get("alone_output", np.nan)) or st["alone_output"] < 0.95:
            return "rejected"
        if st.get("iia_parts", 0) and st.get("iia_above_control", 0.0) < 1.0:
            return "rejected"
        return "certified"
