"""Blind recovery of the five compiled programs, their readable blueprints, and the live reader test.
Run: python3 box/run_tracr_reader.py"""
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box import report as REP  # noqa: E402
from box import reader as RD  # noqa: E402
from box.test_checker import programs  # noqa: E402
from box.tracr_box import run_box, probe_inputs  # noqa: E402
from box.verify import Checker  # noqa: E402

out = {}
for name, (prog, voc) in programs().items():
    net, bp, rep, variables, defined = run_box(name, prog, voc, verbose=False)
    md = REP.save(name, bp, rep, "results/build/blueprints")
    fresh = probe_inputs(voc, 5, 8, seed=99, unique=(name == "sort_unique"))
    truth = net.outputs(fresh)
    numerical = net.numerical_output
    truth = [[round(float(x), 3) for x in t] for t in truth] if numerical else [[x for x in t] for t in truth]
    chk = Checker(net, fresh, seed=0, n_pairs=10, control_reps=1)
    bad = RD.scrambled(md, bp)
    md_bad = REP.to_markdown(name, bad, {"verdict": "(not checked)", "checks": []})
    res = {"verdict": rep["verdict"]}
    for arm, text, what in (("blueprint", md, "a blueprint"), ("scrambled blueprint", md_bad, "a blueprint"),
                            ("no blueprint", "(No description is available.)", "nothing but the format")):
        pred, txt, usage = RD.ask(text, what, fresh)
        res[arm] = round(RD.score(pred, truth, numerical), 3)
        open(f"results/build/blueprints/{name}_reader_{arm.replace(' ', '_')}.txt", "w").write(txt)
    out[name] = res
    print(name, res, flush=True)
json.dump(out, open("results/build/reader_tracr_session2.json", "w"), indent=1)
