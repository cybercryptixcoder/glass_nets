"""Build the interface's reference: a public S0 model and its true blueprint (INTERFACE.md §8).

The model is Tracr's own library program `make_frac_prevs` (the fraction of tokens so far that are "x"), compiled by
Tracr with its residual basis left as compiled and its labels published, so locations can use the `labels` map. The
true blueprint is the published circuit, written in vocabulary v2. Public: nothing here is sealed.

  python3 bench/accept/reference/make_reference.py      (writes bench/accept/reference/frac_prevs/)
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ACCEPT = os.path.dirname(HERE)
sys.path.insert(0, ACCEPT)
import interface as I  # noqa: E402

from tracr.compiler import compiling, lib  # noqa: E402
from tracr.rasp import rasp  # noqa: E402


def canon(label):
    name, sep, val = label.partition(":")
    return re.sub(r"_\d+$", "", name) + sep + val


def main():
    out = os.path.join(HERE, "frac_prevs")
    os.makedirs(os.path.join(out, "network"), exist_ok=True)
    is_x = rasp.numerical(rasp.Map(lambda t: t == "x", rasp.tokens)).named("is_x")
    prog = lib.make_frac_prevs(is_x).named("frac_prevs")
    V, max_len = ["a", "b", "x"], 6
    model = compiling.compile_rasp_to_model(prog, vocab=set(V), max_seq_len=max_len, compiler_bos="BOS")
    labels = [canon(str(x)) for x in model.residual_labels]
    P = {}
    for k, v in model.params.items():
        for kk, vv in v.items():
            name = {"token_embed": "token_embed", "pos_embed": "pos_embed"}.get(k) or k.replace("transformer/", "") + "/" + kk
            P[name] = np.asarray(vv, np.float64)
    readout = np.zeros((len(labels), 1))
    readout[labels.index("frac_prevs"), 0] = 1.0
    P["readout"] = readout
    np.savez(os.path.join(out, "network", "params.npz"), **P)
    cfg = model.model_config
    meta = {"n_layers": cfg.num_layers, "n_heads": cfg.num_heads, "key_size": cfg.key_size,
            "input_vocab": {str(k): int(v) for k, v in model.input_encoder.encoding_map.items()}, "bos": "BOS",
            "max_seq_len": max_len, "readout_kind": "numerical", "readout_values": None, "d_model": len(labels)}
    json.dump(meta, open(os.path.join(out, "network", "meta.json"), "w"), indent=1)
    sites = [{"name": f"resid_{k}", "shape": [max_len + 1, len(labels)], "order": k, "stream": "resid",
              "value_offset": 1, "labels": labels} for k in range(2 * cfg.num_layers + 1)]
    task = {"interface": I.INTERFACE_VERSION, "id": "REF-frac_prevs", "draw": "reference", "tier": "S0",
            "domain": "tok",
            "network": {"format": "tracr_np", "path": "network",
                        "n_params": int(sum(v.size for k, v in P.items() if k != "readout")),
                        "multiply_adds": int(sum(v.shape[0] * v.shape[1] for k, v in P.items() if k.endswith("/w")))},
            "behaviour": {"description": "Tracr's library program make_frac_prevs on tokens == 'x', compiled by Tracr; "
                                         "the output at each position after BOS.",
                          "outputs": [{"name": "out", "kind": "numerical", "values": None, "positions": "per_position"}],
                          "metric": "r2", "decision": {"kind": "threshold", "value": 0.5}},
            "inputs": {"path": "inputs.json", "n": 200}, "train_items": None, "sites": sites,
            "params": [{"name": k, "shape": list(v.shape)} for k, v in P.items()],
            "notes": "The interface's public reference (INTERFACE.md §8)."}
    json.dump(task, open(os.path.join(out, "task.json"), "w"), indent=1)
    rng = np.random.default_rng(0)
    seqs = [[V[i] for i in rng.integers(0, 3, int(rng.integers(2, max_len + 1)))] for _ in range(200)]
    json.dump({"tokens": seqs}, open(os.path.join(out, "inputs.json"), "w"))
    tok = [l for l in labels if l.startswith("tokens:") and l not in ("tokens:BOS", "tokens:compiler_pad")]
    idx = [l for l in labels if l.startswith("indices:")]

    def part(pid, site, labs, enc, definition, ups):
        return {"id": pid, "type": "rule", "vtype": "SOp",
                "location": {"site": site, "positions": "all", "map": {"class": "subspace", "labels": labs}},
                "encoding": enc, "definition": definition, "upstream": ups,
                "claims": [{"claim": "type", "value": "rule", "p": 0.99, "evidence": "exhaustive"}],
                "evidence": {"type": "exhaustive"}}
    cat = lambda labs: {"kind": "categorical", "values": [_typed(l.split(":", 1)[1]) for l in labs], "null_below": 0.5}  # noqa: E731
    bp = {"interface": I.INTERFACE_VERSION, "task": task["id"], "domain": "tok", "library": {},
          "parts": [part("tokens", "resid_0", tok, cat(tok), "tokens", []),
                    part("indices", "resid_0", idx, cat(idx), "indices", []),
                    part("is_x", "resid_2", ["is_x"], {"kind": "numerical", "scale": 1.0, "offset": 0.0},
                         '(map (eqc "x") @tokens)', ["tokens"]),
                    part("frac_prevs", "resid_3", ["frac_prevs"], {"kind": "numerical", "scale": 1.0, "offset": 0.0},
                         "(aggregate (select @indices @indices LEQ) @is_x)", ["indices", "is_x"])],
          "output": "frac_prevs",
          "overview": {"text": "At each position, the share of the tokens so far (this one included) that are 'x': an "
                               "MLP marks the x tokens, then one attention head averages the marks over the earlier "
                               "positions.", "predictor": "@frac_prevs"},
          "systems": [{"id": "mark", "role": "mark the x tokens", "parts": ["tokens", "is_x"], "fed_by": [],
                       "feeds": ["average"]},
                      {"id": "average", "role": "average the marks over the prefix", "parts": ["indices", "frac_prevs"],
                       "fed_by": ["mark"], "feeds": []}]}
    # the site each s-op is first written at, read from the weights
    written = {}
    for l in range(cfg.num_layers):
        for k_site, key in ((2 * l + 1, f"layer_{l}/attn/linear/w"), (2 * l + 2, f"layer_{l}/mlp/linear_2/w")):
            for c in np.nonzero(np.abs(P[key]).sum(0) > 1e-9)[0]:
                written.setdefault(labels[c].split(":", 1)[0], k_site)
    for p in bp["parts"][2:]:
        p["location"]["site"] = f"resid_{written[p['id']]}"
    json.dump(bp, open(os.path.join(out, "true_blueprint.json"), "w"), indent=1)
    print("written", out, "sites:", {p["id"]: p["location"]["site"] for p in bp["parts"]})


def _typed(x):
    try:
        f = float(x)
        return int(f) if f.is_integer() else f
    except ValueError:
        return x


if __name__ == "__main__":
    main()
