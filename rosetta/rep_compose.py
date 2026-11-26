"""Replication of composed definitions in 744706 (its own sparse dictionary, its own line), against the composition
line from controls. Usage: python3 rosetta/rep_compose.py"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.engine import Defs, spec_map  # noqa: E402
from rosetta.sae import train  # noqa: E402
from rosetta.core import literal_specs  # noqa: E402

def prog_from_text(D, text):
    by = {s[3]: s[:3] for s in D.specs}
    return [by[t] for t in text.split(" and ")]

if __name__ == "__main__":
    t0 = time.time()
    cj = json.load(open(f"{L.RES}/compose_sub_K3_s{L.SITE}.json")); line = cj["line"]
    D = Defs(L.P, L.names, L.TR, L.TE)
    rows = np.concatenate([D.rs, D.rt])
    H = L.site_acts(L.get_net("744706"), rows).reshape(len(rows), 128, 64)
    V = H.transpose(0, 2, 1).reshape(-1, 128).astype(np.float32)
    torch.manual_seed(0); sae = train(V, epochs=8)
    with torch.no_grad():
        v, i = sae.codes(torch.tensor(V))
    Z = np.zeros((len(V), 512), np.float32); np.put_along_axis(Z, i.numpy().astype(np.int64), v.numpy(), axis=1)
    Z = Z.reshape(len(rows), 64, 512); Zs, Zt = Z[:len(D.rs)], Z[len(D.rs):]
    Zc = (Zt - Zs.mean(0, keepdims=True)).reshape(-1, 512); Zc = Zc / (Zc.std(0) + 1e-9)
    P2 = np.load(os.path.join(L.R, "prims_744706.npy"), mmap_mode="r"); Pt2 = np.asarray(P2[D.rt], dtype=np.float32)
    out = {"line": line, "items": []}
    for f in cj["features"]:
        m = np.full(Pt2.shape[:2], f["weights"][0], np.float32)
        for t, w in zip(f["programs"], f["weights"][1:]):
            m += w * spec_map(Pt2, prog_from_text(D, t))
        pz = m.reshape(-1); pz = (pz - pz.mean()) / (pz.std() + 1e-9)
        c = (Zc * pz[:, None]).mean(0); j = int(np.nanargmax(np.abs(np.nan_to_num(c))))
        r = {"ld2_feature": f["feature"], "programs": f["programs"], "744706_feature": j, "744706_fidelity": round(float(c[j]), 4), "replicates": bool(abs(c[j]) >= line)}
        out["items"].append(r); print(json.dumps(r), flush=True)
    json.dump(out, open(f"{L.RES}/rep_compose_K3_s{L.SITE}.json", "w"), indent=1)
    print("done", round(time.time() - t0), "s")
