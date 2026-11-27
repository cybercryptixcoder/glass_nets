"""A stronger control for the reader test (DIARY 06:00): the reader gets a certified blueprint of a different program
(hist's for reverse and sort_unique, reverse's for hist). A reader that follows the blueprint scores low against the
true network; one that guesses the task from the inputs scores as it did with no blueprint."""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box import reader as RD  # noqa: E402
from box.test_checker import programs  # noqa: E402
from box.tracr_box import probe_inputs  # noqa: E402
from box.adapters.tracr_net import TracrNet  # noqa: E402

P = programs()
out = {}
for name, other in (("reverse", "hist"), ("sort_unique", "hist"), ("hist", "reverse")):
    prog, voc = P[name]
    net = TracrNet(prog, voc, 5)
    fresh = probe_inputs(voc, 5, 8, seed=99, unique=(name == "sort_unique"))
    truth = [[x for x in t] for t in net.outputs(fresh)]
    md_other = open(f"results/build/blueprints/{other}.md").read()
    pred, txt, _ = RD.ask(md_other, "a blueprint", fresh)
    out[name] = {"other program's blueprint": other, "accuracy against the true network": round(RD.score(pred, truth, False), 3)}
    open(f"results/build/blueprints/{name}_reader_other_{other}.txt", "w").write(txt)
    print(name, out[name], flush=True)
json.dump(out, open("results/build/reader_swap_session2.json", "w"), indent=1)
