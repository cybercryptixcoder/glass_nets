"""Replication in a second, independently trained network (Leela 744706): train the same sparse dictionary on its last
block, and ask whether each LD2 definition is matched by one of ITS variables (look-ahead primitives from 744706's own
line). Usage: SITE=10 python3 rosetta/replicate.py"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.engine import Defs, spec_map  # noqa: E402
from rosetta.core import fidelity  # noqa: E402
from rosetta.sae import train  # noqa: E402
RES = L.RES; SITE = L.SITE

if __name__ == "__main__":
    t0 = time.time()
    D = Defs(L.P, L.names, L.TR, L.TE)
    rows = np.concatenate([D.rs, D.rt])
    H = L.site_acts(L.get_net("744706"), rows).reshape(len(rows), 128, 64)
    V = H.transpose(0, 2, 1).reshape(-1, 128).astype(np.float32)
    sae = train(V, epochs=8)
    with torch.no_grad():
        v, i = sae.codes(torch.tensor(V))
    Z = np.zeros((len(V), 512), np.float32); np.put_along_axis(Z, i.numpy().astype(np.int64), v.numpy(), axis=1)
    Z = Z.reshape(len(rows), 64, 512); Zs, Zt = Z[:len(D.rs)], Z[len(D.rs):]
    Zt = Zt - Zs.mean(0, keepdims=True)
    P2 = np.load(os.path.join(L.R, "prims_744706.npy"), mmap_mode="r")
    Pt2 = np.asarray(P2[D.rt], dtype=np.float32)
    targets = []
    la = json.load(open(f"{RES}/feats_lookahead_s{SITE}.json"))
    line = json.load(open(f"{RES}/feats_controls_s{SITE}.json"))["line_max"]
    dfn = json.load(open(f"{RES}/feats_define_s{SITE}.json"))
    import glob
    dic = {}
    for fp in glob.glob(f"{RES}/dict_s{SITE}_w*.json"):
        for e in json.load(open(fp)):
            dic[e["feature"]] = e
    # the definitions to replicate: the look-ahead-bound ones, the accepted residual carriers, and the dictionary's
    # strongest definitions (top 12 by held-out fidelity)
    for r in la["features"]:
        e = dic.get(r["feature"])
        if e and e.get("program"):
            targets.append(("look-ahead scan", r["feature"], e["text"], e["program"], e["heldout_fidelity"]))
    for u in dfn["features"]:
        if u["accepted_sq"]:
            targets.append(("residual carrier", u["feature"], u["text"], u["program"], u["heldout_fidelity"]))
    top = sorted([e for e in dic.values() if e.get("program")], key=lambda e: -abs(e["heldout_fidelity"]))[:12]
    for e in top:
        targets.append(("strongest", e["feature"], e["text"], e["program"], e["heldout_fidelity"]))
    Zc = Zt.reshape(-1, 512); Zc = Zc / (Zc.std(0) + 1e-9)
    out = {"line": line, "rows": len(D.rt), "items": []}
    seen = set()
    for kind, f, text, prog, fid in targets:
        if (f, text) in seen:
            continue
        seen.add((f, text))
        pm = spec_map(Pt2, prog).reshape(-1); pz = (pm - pm.mean()) / (pm.std() + 1e-9)
        c = (Zc * pz[:, None]).mean(0)
        j = int(np.nanargmax(np.abs(np.nan_to_num(c))))
        r = {"kind": kind, "ld2_feature": f, "definition": text, "ld2_fidelity": fid, "744706_feature": j,
             "744706_fidelity": round(float(c[j]), 4), "replicates": bool(abs(c[j]) >= line)}
        out["items"].append(r); print(json.dumps(r), flush=True)
    json.dump(out, open(f"{RES}/replicate_744706_s{SITE}.json", "w"), indent=1)
    print("done", round(time.time() - t0), "s")
