"""The box on compiled transformers (S0's kind), blind: it is given the network (weights, its token interface and its
outputs when run), never the program or the residual labels. It returns a blueprint and certifies it.

Stages (SPEC.md §1, GAP.md C3 to C6):
1. Map: run the network on probe sequences; find which residual directions each sublayer writes; group written
   directions into variables. A group of 0/1 directions that never fire together and always sum to one is a
   categorical variable (one direction per value); any other written direction is a numerical variable. This uses the
   encoding prior of compiled transformers (one-hot categorical values, single-direction numbers).
2. Define: in the order the network writes them, define each variable from the vocabulary, with every earlier
   variable available as a part (@id): a categorical variable needs a definition whose values map one to one onto
   its directions; a numerical one a definition its value is an exact affine function of.
3. Invent (R8): a numerical variable no definition matches exactly is offered to the operation search: one-line
   numpy transforms of the best definitions; an exact one becomes a new operation in the library.
4. Assemble: parts with their locations (axis directions of the residual, an allowed linear map), definitions,
   wiring; the output part is the variable the network's decoded output reads; anything left is unresolved.
5. Certify with the checker (box/verify.py): run alone, interchange interventions against family-wise control lines.
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from box import tok_fast as F  # noqa: E402
from box.blueprint import Blueprint, Part  # noqa: E402
from box.verify import Checker, vocab  # noqa: E402

TRANSFORMS = {  # the operation search's grammar for inventions: (name, numpy body, literals)
    "recip_inc": ("1.0 / (x + 1.0)", 2), "recip_dec": ("1.0 / x - 1.0", 2), "recip": ("1.0 / x", 1), "square": ("x * x", 0),
    "sqrt": ("np.sqrt(np.abs(x))", 0), "log_inc": ("np.log(x + 1.0)", 1), "exp_neg": ("np.exp(-x)", 0),
}


def probe_inputs(vocab_tokens, max_len, n, seed, unique=False):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        L = int(rng.integers(1, max_len + 1))
        if unique:
            L = min(L, len(vocab_tokens))
        s = list(rng.choice(np.array(vocab_tokens, dtype=object), size=L, replace=not unique))
        out.append([int(x) if isinstance(vocab_tokens[0], (int, np.integer)) else str(x) for x in s])
    return out


# ---------------------------------------------------------------------------------------------------- 1. map
def map_variables(net, seqs):
    _, sites, mask = net.run(seqs, return_sites=True)
    valid = mask[:, 1:]
    variables, used = [], set()
    for k, site in enumerate(net.sites):
        R = sites[site][:, 1:, :]
        D = R if k == 0 else R - sites[net.sites[k - 1]][:, 1:, :]
        written = [j for j in range(net.d) if j not in used and np.abs(D[valid][:, j]).max() > 1e-6]
        vals = {j: R[valid][:, j] for j in written}
        written = [j for j in written if np.ptp(vals[j]) > 1e-6]     # constant directions carry nothing
        binary = [j for j in written if np.all(np.isin(np.round(vals[j], 6), [0.0, 1.0]))]
        groups, left = [], list(binary)
        while left:
            g = [left.pop(0)]
            for j in list(left):
                if all(not np.any((vals[j] > 0.5) & (vals[i] > 0.5)) for i in g):
                    g.append(j); left.remove(j)
            groups.append(g)
        cats = [g for g in groups if len(g) > 1 and np.allclose(sum(vals[j] for j in g), 1.0)]
        in_cat = {j for g in cats for j in g}
        for g in cats:
            variables.append({"site": site, "k": k, "dims": g, "kind": "categorical"})
        for j in written:
            if j not in in_cat:
                variables.append({"site": site, "k": k, "dims": [j], "kind": "numerical"})
        used |= set(written)
    for i, v in enumerate(variables):
        v["id"] = f"v{i}"
        R = sites[v["site"]][:, 1:, :]
        v["value"] = np.argmax(R[:, :, v["dims"]], -1).astype(float) if v["kind"] == "categorical" else R[:, :, v["dims"][0]]
        v["value"] = np.where(valid, v["value"], np.nan)
    return variables, valid


# ---------------------------------------------------------------------------------------------------- 2. define
def _bijection(c, g):
    """min of the two purities between candidate values c and group indices g (1.0 = one to one). A candidate that
    is undefined where the variable is defined scores 0."""
    if np.any(np.isnan(c) & ~np.isnan(g)):
        return 0.0
    ok = ~np.isnan(c) & ~np.isnan(g)
    if ok.sum() == 0:
        return 0.0
    c, g = np.round(c[ok], 6), g[ok]
    def purity(a, b):
        tot = 0
        for v in np.unique(a):
            _, cnt = np.unique(b[a == v], return_counts=True)
            tot += cnt.max()
        return tot / len(a)
    return min(purity(c, g), purity(g, c))


def _commit(text, defined, by):
    """Rewrite a definition that uses alternative labelings (@v~j): commit each such upstream variable to that
    labeling, recursively (an alternative may itself use another's alternative)."""
    import re
    for uid, j in set(re.findall(r"@(v\d+)~(\d+)", text)):
        alt = defined[uid]["alternatives"][int(j)]
        defined[uid].update({"definition": _commit(alt["definition"], defined, uid), "value": alt["value"], "committed_by": by,
                             "alternatives": []})
        text = text.replace(f"@{uid}~{j}", f"@{uid}")
    return text


def _valkey(c, valid):
    return hash(np.round(np.nan_to_num(c[valid], nan=-777.0), 6).tobytes())


def _is_int(c, valid):
    x = c[valid & np.isfinite(c)] if c is not None else np.array([])
    return bool(len(x)) and bool(np.allclose(x, np.round(x), atol=1e-6))


def _affine_r2(c, y):
    if np.any(~np.isfinite(c) & np.isfinite(y)):
        return -1.0, 0.0, 0.0
    ok = np.isfinite(c) & np.isfinite(y)
    if ok.sum() < 3 or np.std(c[ok]) == 0 or np.std(y[ok]) == 0:
        return -1.0, 0.0, 0.0
    a, b = np.polyfit(c[ok], y[ok], 1)
    return float(1 - np.mean((a * c[ok] + b - y[ok]) ** 2) / np.var(y[ok])), float(a), float(b)


def candidates(atoms, valid, codec, consts, symbolic, kind="any"):
    """Every one-step RASP expression over the atoms: map, sequence_map, aggregate, aggregate_cat, selector_width.
    atoms: {text: value [N, T]}; symbolic: the atoms whose values are tokens rather than numbers (only compared,
    selected on and carried, never averaged or added). Yields (text, value, symbolic)."""
    names = list(atoms)
    num = [a for a in names if a not in symbolic]
    elementwise, mixing = kind in ("any", "mlp"), kind in ("any", "attn")
    for a in (names if elementwise else []):
        if a in num:
            for f in F.FN1:
                yield f"(map {f} {a})", np.where(valid, F.FN1[f](atoms[a]), np.nan), False
        for f, mk in F.FN1C.items():
            if a in symbolic and f not in ("eqc", "ltc", "gtc"):
                continue
            cs = consts["token"] if (a in symbolic or f in ("eqc", "ltc", "gtc")) else consts["int"]
            for c in cs:
                lit = codec.literal(c) if a in symbolic else repr(int(c)) if float(c).is_integer() else repr(float(c))
                with np.errstate(all="ignore"):
                    yield f"(map ({f} {lit}) {a})", np.where(valid, mk(c)(atoms[a]), np.nan), False
    for a, b in (itertools.product(names, names) if elementwise else []):
        for f in F.FN2:
            if (a in symbolic or b in symbolic) and f not in ("equal", "less", "greater"):
                continue
            with np.errstate(all="ignore"):
                yield f"(sequence_map {f} {a} {b})", np.where(valid, F.FN2[f](atoms[a], atoms[b]), np.nan), False
    for kq in (itertools.product(names, names) if mixing else []):
        for p in F.PREDS:
            sel = F.select(atoms[kq[0]], atoms[kq[1]], p)
            stext = f"(select {kq[0]} {kq[1]} {p})"
            yield f"(selector_width {stext})", F.selector_width(sel, valid), False
            for v in names:
                if v in num:
                    yield f"(aggregate {stext} {v})", F.aggregate(sel, atoms[v], valid), False
                yield f"(aggregate_cat {stext} {v})", F.aggregate_cat(sel, atoms[v], valid), v in symbolic


def define_variables(variables, valid, tok, idx, codec, max_len, out_id=None, outv=None, net=None):
    consts = {"token": sorted(set(np.unique(tok[~np.isnan(tok)]).tolist())), "int": list(range(-max_len, max_len + 1))}
    defined, library = {}, {}
    base = {"tokens": tok, "indices": idx}
    for v in variables:
        reads = net.reads(v["k"]) if (net is not None and v["k"] > 0) else None
        atoms, symbolic = {}, set()
        for u in variables:
            if u["k"] < v["k"] and u["id"] in defined and defined[u["id"]]["status"] == "defined" and \
                    (reads is None or set(u["dims"]) & reads):
                atoms["@" + u["id"]] = defined[u["id"]]["value"]
                if defined[u["id"]].get("symbolic"):
                    symbolic.add("@" + u["id"])
                for j, alt in enumerate(defined[u["id"]].get("alternatives", [])):
                    atoms[f"@{u['id']}~{j}"] = alt["value"]
        best = (-1.0, None, None, None, None)
        if v["k"] == 0:   # the inputs themselves
            pool = [("tokens", tok, not codec.numeric), ("indices", idx, False)]
        else:
            # an MLP acts position by position (map, sequence_map); attention moves values across positions
            # (select-based); which one wrote the variable is known from the sublayer index
            pool = list(candidates(atoms, valid, codec, consts, symbolic, "mlp" if v["k"] % 2 == 0 else "attn"))
        near = []
        sym_of = {}
        for text, c, sym in pool:
            sym_of[text] = sym
            if np.all(np.isnan(c)):
                continue
            if v["id"] == out_id:   # the output part must give the decoded output itself
                ok = ~np.isnan(outv)
                s = float(np.mean(np.isclose(np.nan_to_num(c[ok], nan=-1e9), outv[ok]))); extra = None
                if v["kind"] == "numerical" and s < 0.999:
                    s = min(s, 0.998)
            elif v["kind"] == "categorical":
                s = _bijection(c, v["value"]); extra = None
            else:
                s, a_, b_ = _affine_r2(c, v["value"]); extra = (a_, b_)
            if not sym:
                near.append((s, text, c))
            key = (round(s, 6), _is_int(c, valid), -len(text))   # exact first; then natural (integer) values; then short
            if best[1] is None or key > best[4]:
                best = (s, text, c, extra, key)
        s, text, c = best[0], best[1], best[2]
        rec = {"score": round(float(s), 6), "definition": text, "value": c, "status": "defined" if s >= 0.999 else "open",
               "symbolic": sym_of.get(text, False)}
        if v["kind"] == "categorical" and v["id"] != out_id and rec["status"] == "defined":
            # other exact labelings, kept for later variables to choose from (a categorical variable's labels are
            # fixed only by how later parts compare it with other variables)
            alts, seen_vals = [], {_valkey(c, valid)}
            def natural(z):   # exact first; then non-negative values starting at 0 (counts, positions); then short
                x = z[2][valid & np.isfinite(z[2])]
                return (-round(z[0], 6), not (len(x) and x.min() >= 0), not (len(x) and x.min() == 0), len(z[1]))
            for s_, t_, c_ in sorted(near, key=natural):
                if s_ < 0.999 or not _is_int(c_, valid):
                    continue
                kv = _valkey(c_, valid)
                if kv in seen_vals:
                    continue
                seen_vals.add(kv); alts.append({"definition": t_, "value": c_})
                if len(alts) >= 8:
                    break
            rec["alternatives"] = alts
        if rec["status"] == "open" or (v["kind"] == "categorical" and not _is_int(c, valid)):
            near.sort(key=lambda t: -t[0])
            tries = [(t, cv) for _s, t, cv in near[:30]] + [(a, atoms[a]) for a in atoms if a not in symbolic]
            for t, cv in tries:
                for name, (body, lits) in TRANSFORMS.items():
                    with np.errstate(all="ignore"):
                        tv = eval(body, {"np": np, "x": cv})  # noqa: S307 - fixed grammar of one-liners
                    if v["id"] == out_id:
                        ok = ~np.isnan(outv)
                        s2 = float(np.mean(np.isclose(np.nan_to_num(tv[ok], nan=-1e9), outv[ok])))
                    elif v["kind"] == "categorical":
                        s2 = _bijection(np.round(tv, 6), v["value"])
                    else:
                        s2 = _affine_r2(tv, v["value"])[0]
                    if s2 >= 0.999 and (v["kind"] != "categorical" or _is_int(tv, valid)):
                        lib_name = f"inv_{name}"
                        library[lib_name] = {"kind": "operation", "args": ["SOp"], "ret": "SOp",
                                             "source": f"def {lib_name}(x):\n    return {body}\n",
                                             "provenance": f"operation search, to define {v['id']} (exact affine fit R2 {s2:.4f})"}
                        rec = {"score": round(float(s2), 6), "definition": f"(${lib_name} {t})", "value": tv, "status": "defined",
                               "invented": lib_name}
                        break
                if rec["status"] == "defined":
                    break
        if rec["status"] == "defined" and "~" in (rec["definition"] or ""):
            rec["definition"] = _commit(rec["definition"], defined, v["id"])
        defined[v["id"]] = rec
    return defined, library


# ---------------------------------------------------------------------------------------------------- 4. assemble
def _refs(text):
    return sorted({t[1:] for t in text.replace("(", " ").replace(")", " ").split() if t.startswith("@")})


def assemble(net, variables, defined, library, out_seqs_values, valid):
    """The output part is the variable the decoded outputs read one to one (or exactly, if numerical)."""
    out_part, best = None, -1.0
    for v in variables:
        if defined[v["id"]]["status"] != "defined":
            continue
        s = _bijection(v["value"], out_seqs_values) if v["kind"] == "categorical" else _affine_r2(v["value"], out_seqs_values)[0]
        if s > best + 1e-9 or (abs(s - best) <= 1e-9 and v["k"] > next(u["k"] for u in variables if u["id"] == out_part)):
            best, out_part = s, v["id"]
    parts = []
    for v in variables:
        d = defined[v["id"]]
        B = np.zeros((net.d, len(v["dims"]))); B[v["dims"], np.arange(len(v["dims"]))] = 1.0
        if d["status"] == "defined":
            parts.append(Part(v["id"], "rule", "SOp", v["site"], basis=B, encoding=v["kind"],
                              definition=_vocab_text(d["definition"]), upstream=_refs(d["definition"]),
                              evidence={"score": d["score"], "invented": d.get("invented")}))
        else:
            parts.append(Part(v["id"], "unresolved", "SOp", v["site"], basis=B, encoding=v["kind"],
                              evidence={"best_score": d["score"], "best_try": d["definition"]}))
    return Blueprint("tok", parts, out_part, library=library, name="recovered blind"), best


def _vocab_text(t):
    return t


# ---------------------------------------------------------------------------------------------------- the whole box
def run_box(program_name, program, vocab_tokens, max_len=5, n_probe=150, seed=0, verbose=True):
    from box.adapters.tracr_net import TracrNet
    net = TracrNet(program, vocab_tokens, max_len)
    unique = program_name == "sort_unique"
    probes = probe_inputs(vocab_tokens, max_len, n_probe, seed, unique)
    codec = F.Codec(vocab_tokens)
    tok, idx = codec.encode_batch(probes)
    variables, valid = map_variables(net, probes)
    outs = net.outputs(probes)
    T = valid.shape[1]
    outv = np.full((len(probes), T), np.nan)
    for i, o in enumerate(outs):
        for j, x in enumerate(o):
            try:
                outv[i, j] = codec.code[x] if (not codec.numeric and x in codec.code) else float(x)
            except (TypeError, ValueError):
                outv[i, j] = np.nan
    # which variable the network's decoded output reads: one to one with its categories, or exactly (numerical)
    def out_score(v):
        return _bijection(v["value"], outv) if v["kind"] == "categorical" else _affine_r2(v["value"], outv)[0]
    out_id = max(variables, key=lambda v: (round(out_score(v), 6), v["k"]))["id"]
    defined, library = define_variables(variables, valid, tok, idx, codec, max_len, out_id, outv, net)
    bp, out_match = assemble(net, variables, defined, library, outv, valid)
    bp.output = out_id
    # literal constants in definitions are in the evaluator's codes; for strings the codec wrote them quoted already
    held = probe_inputs(vocab_tokens, max_len, 80, seed + 1, unique)
    chk = Checker(net, held, seed=0, n_pairs=120, control_reps=300)
    report = chk.certify(bp, run_alone_line=1.0 - 1e-9)
    if verbose:
        print(f"== {program_name}: {len(variables)} variables found; output part {bp.output} (match {out_match:.3f})")
        for v in variables:
            d = defined[v["id"]]
            print(f"   {v['id']} at {v['site']} ({v['kind']}, {len(v['dims'])} directions): "
                  f"{d['status']} {d['definition'] if d['status'] == 'defined' else '(best ' + str(d['definition']) + ')'} [{d['score']}]")
        print(f"   verdict: {report['verdict']}; " + "; ".join(f"{c[0]}: {c[2]}" for c in report["checks"] if not c[1] or c[0] in ("run alone",)))
    return net, bp, report, variables, defined


if __name__ == "__main__":
    from box.test_checker import programs
    out = {}
    for name, (prog, voc) in programs().items():
        net, bp, rep, variables, defined = run_box(name, prog, voc)
        out[name] = {"verdict": rep["verdict"], "checks": [(c[0], bool(c[1]), c[2]) for c in rep["checks"]],
                     "parts": [{"id": p.id, "type": p.type, "site": p.site, "dims": len(p.basis[0]) if p.basis is not None else 0,
                                "definition": p.definition, "upstream": p.upstream} for p in bp.parts],
                     "output": bp.output, "library": bp.library}
    os.makedirs("results/build", exist_ok=True)
    json.dump(out, open("results/build/tracr_blind_session2.json", "w"), indent=1, default=str)
