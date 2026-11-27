"""The verifier hook on the public known cases, written as task folders in a temporary directory: it must print only
counts, and the counts must match the known cases (all true certified, all accounts rejected). Run: python3 box/test_hook.py"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box.hook import dump_blueprint  # noqa: E402
from box import test_checker as T  # noqa: E402
from box.adapters.tracr_net import TracrNet  # noqa: E402

import numpy as np  # noqa: E402

root = tempfile.mkdtemp(prefix="hook_")
rng = np.random.default_rng(1)
ok_all = True
for name, (prog, voc) in T.programs().items():
    net = TracrNet(prog, voc, 5)
    X = T.inputs_for(name, voc)
    d = os.path.join(root, name); os.makedirs(os.path.join(d, "true")); os.makedirs(os.path.join(d, "accounts"))
    json.dump({"adapter": "tracr_lib", "program": name, "vocab": voc, "max_seq_len": 5, "run_alone_line": 0.999}, open(os.path.join(d, "task.json"), "w"))
    json.dump(X, open(os.path.join(d, "inputs.json"), "w"))
    bp = T.true_blueprint(name, net, X)
    dump_blueprint(bp, os.path.join(d, "true", "true.json"))
    for i, v in enumerate(T.variants(name, bp, net, rng)):
        if isinstance(v, tuple):
            open(os.path.join(d, "accounts", f"a{i}.py"), "w").write(v[0])
        else:
            dump_blueprint(v, os.path.join(d, "accounts", f"a{i}.json"))
    out = subprocess.run([sys.executable, "-m", "box.hook", "S0", d, os.path.join(d, "true"), os.path.join(d, "accounts")],
                         capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    lines = [l for l in out.stdout.strip().splitlines() if l]
    expect = f"S0: certified 1 of 1 true blueprints, rejected 8 of 8 accounts"
    good = lines == [expect]
    ok_all &= good
    print(("PASS " if good else "FAIL ") + f"{name}: hook printed {lines}" + (f" | stderr: {out.stderr[-300:]}" if not good else ""))
sys.exit(0 if ok_all else 1)
