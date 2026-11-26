"""Per-square primitives for all positions, with a given network's own line. Writes ~/kdata/rosetta/prims_<NET>.npy [N, 64, K] float16."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rosetta.chess_lib import build, NAMES  # noqa: E402
K = os.path.expanduser("~/kdata")
net = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else None
items = json.load(open(os.path.join(K, "chess", "LD2", "meta.json")))["items"][:n]
lines = json.load(open(os.path.join(K, "rosetta", f"{net}_lines.json")))[:n]
t0 = time.time(); P = build([it["fen"] for it in items], lines, procs=int(os.environ.get("PROCS", "2")))
print(P.shape, round(time.time() - t0, 1), "s", flush=True)
if n is None:
    np.save(os.path.join(K, "rosetta", f"prims_{net}.npy"), P.astype(np.float16))
    json.dump(NAMES, open(os.path.join(K, "rosetta", "prim_names.json"), "w"))
