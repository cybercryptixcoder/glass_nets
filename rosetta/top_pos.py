"""The 16 variables LD2's evaluation depends on most (glass2 attribution): can a counted program (how many squares
satisfy a rule, material removed) define their board-wide totals? Line from the untrained-network controls
(feats_controls_s10.json, line_pos_max), fixed earlier. Usage: python3 rosetta/top_pos.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta.run_feats import L, fmap  # noqa: E402
from rosetta.engine import Defs  # noqa: E402
att = np.array(json.load(open(f"{L.RES}/glass2_attr.json"))["attribution"]); order = np.argsort(-att)
line = json.load(open(f"{L.RES}/feats_controls_s{L.SITE}.json"))["line_pos_max"]
D = Defs(L.P, L.names, L.TR, L.TE); out = {"line_pos": line, "variables": []}
for f in order[:16]:
    f = int(f); d = D.define(lambda rr, f=f: fmap(f, rr))
    p = d.get("pos", {})
    r = {"feature": f, "attribution_share": round(float(att[f] / att.sum()), 3), "square_fidelity": d.get("heldout_fidelity"),
         "pos_text": p.get("text"), "pos_fidelity_material_removed": p.get("heldout_fidelity"), "pos_fidelity_raw": p.get("heldout_fidelity_raw"),
         "pos_above_line": bool(abs(p.get("heldout_fidelity") or 0) > line)}
    out["variables"].append(r); print(r, flush=True)
json.dump(out, open(f"{L.RES}/top_pos_s{L.SITE}.json", "w"), indent=1)
