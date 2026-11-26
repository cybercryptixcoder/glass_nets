"""Substitution of the capture concept inside 744706: train its sparse dictionary (as rep_compose.py), find its variable
that best matches LD2's composed variable 174, define that variable as a three-program composition over 744706's own
primitives (its own line), substitute it at block 10 and measure the share of its value effect restored."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SITE", "10")
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.engine import Defs, spec_map, additive_map, restore_share  # noqa: E402
from rosetta.sae import train  # noqa: E402
from rosetta.rep_compose import prog_from_text  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    P2 = np.load(os.path.join(L.R, "prims_744706.npy"), mmap_mode="r")
    D = Defs(P2, L.names, L.TR, L.TE)
    rows = np.concatenate([D.rs, D.rt]); ns = len(D.rs)
    m = L.get_net("744706")
    H = L.site_acts(m, rows).reshape(len(rows), 128, 64)
    V = H.transpose(0, 2, 1).reshape(-1, 128).astype(np.float32)
    sae = train(V, epochs=8)
    with torch.no_grad():
        v, i = sae.codes(torch.tensor(V))
    Z = np.zeros((len(V), 512), np.float32); np.put_along_axis(Z, i.numpy().astype(np.int64), v.numpy(), axis=1)
    Z = Z.reshape(len(rows), 64, 512)
    cj = json.load(open(f"{L.RES}/compose_sub_K3_s{L.SITE}.json")); c174 = [x for x in cj["features"] if x["feature"] == 174][0]
    Pt2 = np.asarray(P2[D.rt], dtype=np.float32)
    m174 = np.full(Pt2.shape[:2], c174["weights"][0], np.float32)
    for t, w in zip(c174["programs"], c174["weights"][1:]):
        m174 += w * spec_map(Pt2, prog_from_text(D, t))
    Zc = (Z[ns:] - Z[:ns].mean(0, keepdims=True)).reshape(-1, 512); Zc = Zc / (Zc.std(0) + 1e-9)
    pz = m174.reshape(-1); pz = (pz - pz.mean()) / (pz.std() + 1e-9)
    f = int(np.nanargmax(np.abs(np.nan_to_num((Zc * pz[:, None]).mean(0)))))
    pos = {r: j for j, r in enumerate(rows)}
    fm = lambda rr: Z[[pos[r] for r in rr], :, f]
    comp = D.define_additive(fm, terms=3)
    mu = Z[:ns, :, f].mean(0)
    te = D.rt[:800]; idx = np.array([pos[r] for r in te])
    am = Z[idx, :, f]; rep = np.clip(additive_map(np.asarray(P2[te], np.float32), comp["terms"], comp["weights"]) + mu[None], 0, None)
    d = sae.dec.weight.detach()[:, f].view(1, 128, 1, 1)
    def ed(a, b=None):
        def e(site, h):
            if site == L.SITE:
                h = h - d * torch.tensor(a, dtype=h.dtype).view(-1, 1, 8, 8)
                if b is not None:
                    h = h + d * torch.tensor(b, dtype=h.dtype).view(-1, 1, 8, 8)
            return h
        return e
    v0 = L.value(m, te); va = L.value(m, te, ed(am)); vs = L.value(m, te, ed(am, rep))
    out = {"744706_feature": f, "programs": comp["texts"], "weights": comp["weights"], "composed_fidelity": comp["curve"][-1]["heldout_fidelity"],
           "restore": round(restore_share(v0, va, vs), 4), "mean_abs_value_change_ablation": round(float(np.mean(np.abs(va - v0))), 5)}
    print(json.dumps(out), round(time.time() - t0), "s")
    json.dump(out, open(f"{L.RES}/sub_744706_capture.json", "w"), indent=1)
