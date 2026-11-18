"""Progressive replacement (experiment R2 in docs/glass/PLAN_SCALING.md; its own entry point, off by default).

The reader starts from an EXACT program of the network (a copy, with a variable at every site) plus its own notes from
an earlier run, and replaces parts of it with rules, one at a time. Every submission gets the intent-v2 certificate
(docs/glass/INTENT.md): fidelity with a 95% interval against the task rule, copied information S against the copy
frontier, the calibrated judge's inside tests, and where the program disagrees with the network. No verdict.

    python3 lab/replacer.py TARGET --notes-from RUN_ID [--steps 40] [--model NAME] [--tag r2]
"""
import argparse, json, math, os, sys, time, uuid
from pathlib import Path
import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(LAB.parent / "scripts" / "glass"))
import constructor as C  # noqa: E402  (loads lab/.env)
import judge, nets, luminal  # noqa: E402
import info_measure as IM  # noqa: E402

C.READER = "v1"                        # plain judge text, no stage table / FIX FIRST
OUT = LAB / "constructions"
FRONTIER_DIR = LAB.parent / "results" / "glass" / "scaling" / "frontier"

_base = C.SYSTEM
REPLACE_SYSTEM = (_base[:_base.index("7. Compression")] + "\n" + _base[_base.index("YOUR TOOLS"):_base.index("WHEN THE ANSWER TOKENS")] + """
YOUR STARTING POINT AND YOUR GOAL (progressive replacement)
You start from an EXACT program of this network: its own weights and arithmetic written out, with a variable registered at every site. It passes the judge's truth tests and explains nothing: it is a copy. You also get your own notes from an earlier study of this network.
Turn the copy into an ACCOUNT a person can read, one replacement at a time:
- Pick one part (a unit or group of units, one step, an attention pattern, the readout) and replace its copied numbers with a rule you have checked with python (a comparison, a running maximum, a count, a pointer, a threshold, a small formula). State every rule in words in DESCRIPTION.
- Submit after each replacement and read what it cost.
- Parts you have not understood yet stay as copied numbers. That is honest; the report charges for them.
- Keep a variable in ALIGN only where your program really computes something that lives at that site. Every variable you keep in ALIGN is tested.

WHAT EACH REPORT GIVES YOU (there is no pass/fail verdict)
- FIDELITY: the share of inputs where your program makes the network's decision, with a 95% interval, next to the same number for the TASK RULE (what knowing the task alone gives). An account of THIS network must stay above the task rule.
- COPIED INFORMATION S = bits your file carries / bits of the cheapest copy of the network's weights that reaches your fidelity. The starting copy is near 1. A readable account is at 0.5 or below. Writing numbers differently (integers, scales, tables, encodings, other bases) does not lower S; only replacing numbers by rules does.
- INSIDE TESTS for every variable you place at a site, and WHERE your program disagrees with the network (by decision, with example inputs).
The aim, in order: (1) fidelity above the task rule; (2) inside claims that pass; (3) S as low as you can get it with rules you can state.

HOW TO WORK
Check a rule with python before you put it in (regress a unit's activation on your candidate quantities; compare the rule's output with the unit's). Small steps: replace one part, submit, read, continue. Keep messages short; think with code. When you cannot replace more without losing fidelity, finish with a short summary of the account and no tool call.""")


def fmt(a):
    return np.array2string(np.asarray(a), separator=", ", precision=6, threshold=10**9, max_line_width=150, floatmode="maxprec")


def copy_start(meta, W):
    """An exact construction of the network with a variable at every site (the starting point)."""
    kind = meta["forward"]
    head = 'import numpy as np\nDESCRIPTION = "Exact copy of the network (starting point). Nothing is explained yet."\n'
    if kind == "rnn_relu":
        T = len([s for s in nets.site_table(kind, W) if s.startswith("h")]) if False else None
        body = (f"W_x = np.array({fmt(W['W_x'])})\nW_h = np.array({fmt(W['W_h'])})\nW_o = np.array({fmt(W['W_o'])})\n\n"
                "def model(x, iv):\n    s = np.zeros((len(x), W_h.shape[0]))\n    for t in range(x.shape[1]):\n"
                "        s = iv(f'h{t + 1}', np.maximum(x[:, t:t + 1] * W_x + s @ W_h, 0))\n    return s @ W_o\n\n")
        n = int(np.load(LAB / "models" / meta["id"] / "inputs.npy").shape[1])
        return head + body + "ALIGN = {" + ", ".join(f"'h{t}': ['h{t}']" for t in range(1, n + 1)) + "}\n"
    if kind == "transformer":
        nl = len([k for k in W if k.endswith(".W_Q")])
        lines = [f"W_E = np.array({fmt(W['W_E'])})", f"W_pos = np.array({fmt(W['W_pos'])})", f"W_U = np.array({fmt(W['W_U'])})"]
        for l in range(nl):
            for m in "QKVO":
                lines.append(f"{m}{l} = np.array({fmt(W[f'L{l}.W_{m}'])})")
        pos = int(np.asarray(W["out_positions"]).ravel()[0]) if "out_positions" in W else -1
        body = "\n".join(lines) + f"""
LAYERS = [{', '.join(f'(Q{l}, K{l}, V{l}, O{l})' for l in range(nl))}]

def model(x, iv):
    B, P = x.shape
    h = iv('embed', (W_E[x] + W_pos[:P]).reshape(B, -1)).reshape(B, P, -1)
    for l, (Q, K, V, O) in enumerate(LAYERS):
        q, k, v = (np.einsum('bpd,hde->bhpe', h, M) for M in (Q, K, V))
        a = np.einsum('bhpe,bhqe->bhpq', q, k) / np.sqrt(Q.shape[-1])
        a = np.exp(a - a.max(-1, keepdims=True)); a = a / a.sum(-1, keepdims=True)
        a = iv(f'pattern{{l}}', a.reshape(B, -1)).reshape(a.shape)
        o = iv(f'attn_out{{l}}', np.einsum('bhpq,bhqe,hed->bpd', a, v, O).reshape(B, -1)).reshape(B, P, -1)
        h = iv(f'resid_mid{{l}}', (h + o).reshape(B, -1)).reshape(B, P, -1)
    return h[:, {pos}] @ W_U

"""
        al = ["'embed': ['embed']"] + [f"'pattern{l}': ['attn{l}.pattern'], 'attn_out{l}': ['attn{l}.out'], 'resid_mid{l}': ['resid{l}.mid']" for l in range(nl)]
        return head + body + "ALIGN = {" + ", ".join(al) + "}\n"
    raise ValueError(kind)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


_FRONT = {}


def frontier(target, meta, W, X):
    if target in _FRONT:
        return _FRONT[target]
    f = FRONTIER_DIR / f"{target}.json"
    if f.exists():
        pts = [tuple(p) for p in json.loads(f.read_text())]
    else:
        FRONTIER_DIR.mkdir(parents=True, exist_ok=True)
        pts = IM.frontier(meta, W, X)
        f.write_text(json.dumps(pts))
    _FRONT[target] = pts
    return pts


def certificate(target, src, rep, variant="original"):
    """The intent-v2 certificate of a construction file (dict) plus the text shown to the reader."""
    meta, W, X = judge.load_target(target, variant, 0)
    net = nets.forward(meta["forward"], W, X)
    dn = judge.decide(net)
    cert = {"target": target, "n": len(X)}
    try:
        model, _ = judge.load_construction(src)
        out = np.asarray(model(X, lambda name, v: v), dtype=float).reshape(len(X), -1)
        agree = judge.same(net, out)
    except Exception as e:  # noqa: BLE001
        cert["error"] = f"{type(e).__name__}: {e}"
        return cert, f"CERTIFICATE: your program failed to run on the inputs: {cert['error']}"
    k = int(agree.sum()); cert["fidelity"] = k / len(X); cert["fidelity_ci"] = wilson(k, len(X))
    L = judge.load_labels(meta)
    if L is not None:
        kt = int((dn[:, 0] == np.asarray(L).reshape(-1)).sum()); cert["task"] = kt / len(X); cert["task_ci"] = wilson(kt, len(X))
    B = IM.bits(IM.strip(src)); pts = frontier(target, meta, W, X) if variant == "original" else IM.frontier(meta, W, X)
    Cb = IM.C_of(pts, cert["fidelity"]); cert.update(bits=B, copy_bits=Cb, S=B / Cb)
    sites = (rep or {}).get("sites", {})
    cert["sites"] = {s: bool(r.get("pass")) for s, r in sites.items()}
    cert["claims"] = len(sites); cert["claims_passed"] = sum(cert["sites"].values())
    cls = {}
    for c in np.unique(dn[:, 0]):
        m = dn[:, 0] == c
        cls[int(c)] = (float(agree[m].mean()), int(m.sum()))
    cert["by_class"] = cls
    beats = "task" in cert and cert["fidelity_ci"][0] > cert["task_ci"][1]
    cert["beats_task"] = bool(beats)
    deep = [x for x, v in cert["sites"].items() if v and x not in ("embed", "h1")]
    cert["deep_claims_passed"] = len(deep)
    cert["works"] = bool(beats and len(deep) >= 1 and cert["claims_passed"] == cert["claims"] and cert["S"] <= 0.5)
    # text for the reader
    lo, hi = cert["fidelity_ci"]
    t = [f"CERTIFICATE (no verdict)",
         f"FIDELITY: your program makes the network's decision on {100 * cert['fidelity']:.1f}% of {len(X)} inputs (95% interval {100 * lo:.1f}-{100 * hi:.1f}%)."]
    if "task" in cert:
        t.append(f"TASK RULE: knowing the task alone gives {100 * cert['task']:.1f}% ({100 * cert['task_ci'][0]:.1f}-{100 * cert['task_ci'][1]:.1f}%). "
                 + ("You are ABOVE it." if beats else "You are NOT above it: so far this explains the task, not this network."))
    t.append(f"COPIED INFORMATION: S = {cert['S']:.2f} (your file {B} bits; the cheapest copy of the weights reaching {100 * cert['fidelity']:.1f}% needs {Cb} bits). "
             "Near 1 = still a copy; a readable account is at 0.5 or below.")
    t.append(f"INSIDE CLAIMS: {cert['claims_passed']} of {cert['claims']} sites pass: " + ", ".join(f"{s} {'PASS' if v else 'fail'}" for s, v in cert["sites"].items()))
    t.append("BY THE NETWORK'S DECISION: " + "; ".join(f"decides {c}: you match {100 * a:.1f}% (n={n})" for c, (a, n) in cls.items()))
    bad = np.where(~agree)[0]
    if len(bad):
        g = np.sort(np.asarray(net).reshape(len(X), -1), 1)
        margin = g[:, -1] - g[:, -2] if g.shape[1] > 1 else np.abs(g[:, 0])
        pick = bad[np.argsort(-margin[bad])][:6]
        dc = judge.decide(out)
        t.append("DISAGREEMENTS (clearest first): " + "; ".join(
            f"x={np.round(X[i], 2).tolist()} network {dn[i].tolist()} yours {dc[i].tolist()}" for i in pick))
    return cert, "\n".join(t)


def judge_text(text):
    cut = text.find("7. COMPRESSION")
    return text[:cut].rstrip() if cut >= 0 else text


def key(cert):
    if "error" in cert:
        return (False, False, False, -9.0)
    return (cert["works"], cert["beats_task"], cert["claims"] >= 1 and cert["claims_passed"] == cert["claims"], -cert["S"])


TOPDOWN = """

STRATEGY: TOP-DOWN, AND TRADE EXACTNESS FOR RULES
Being exact is NOT the goal: the copy is already exact. The target is an account that (a) stays above the task rule's fidelity, (b) has passing inside claims at sites AFTER the first step (not only the embedding or h1), and (c) has S at or below 0.5. Losing a few points of fidelity to replace a whole matrix by a rule is the right trade.
Work from the answer backwards:
1. At the LAST site before the readout, find a FEW quantities (2-6 numbers per input, possibly soft or graded) such that the activation there is close to an affine function of them, and a short stated rule turns them into the decision. Check both with python (regress the site on them; apply your rule). Put them in the file as variables at that site, computed from the input by the simplest formula you can find, and replace everything before them.
2. Submit and read: fidelity vs the task rule, the inside tests at that site, S.
3. Then explain where those quantities come from: which earlier-site quantities they are computed from, and how. Add those as variables at the earlier site.
If a crisp rule loses too much fidelity, use a graded version of it (a soft count, a weighted comparison) whose few coefficients you measure."""


def run(target, notes_from=None, steps=40, model_name=None, tag="r2", effort="high", max_tokens=12000, variant="original", start_src=None, strategy=None):
    if model_name:
        os.environ["HELIX_FLASH_MODEL"] = model_name
    meta = json.loads((LAB / "models" / target / "meta.json").read_text())
    W = {k: v.astype(np.float64) for k, v in np.load(LAB / "models" / target / "weights.npz").items()}
    cfg = {"target": target, "variant": variant, "told": False, "weights": True, "allow_forward": True, "effort": effort,
           "max_steps": steps, "max_tokens": max_tokens, "seed": 0, "model": luminal.model(), "mode": "replace", "tag": tag,
           "notes_from": notes_from, "judge_version": judge.JUDGE_VERSION}
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + tag + "-" + uuid.uuid4().hex[:4]
    rdir = OUT / target / run_id; rdir.mkdir(parents=True, exist_ok=True)
    log = open(rdir / "events.log", "a")
    emit = lambda ev: (log.write(json.dumps(ev)[:3000] + "\n"), log.flush())  # noqa: E731
    notes = "(no earlier notes)"
    if notes_from:
        prev = json.loads((OUT / target / notes_from / "run.json").read_text())
        nn = prev.get("notes") or prev.get("compactions") or []
        notes = nn[-1]["notes"] if nn else (prev.get("final_message") or notes)
    start = start_src or copy_start(meta, W)
    task = C.task_text(meta, W, {**cfg, "reader": "v1"})
    first = (task + "\n\nYOUR NOTES FROM AN EARLIER STUDY OF THIS NETWORK\n" + notes +
             "\n\nTHE STARTING PROGRAM (an exact copy; replace parts of it with rules)\n```python\n" + start + "```")
    system = REPLACE_SYSTEM + (TOPDOWN if strategy == "topdown" else "")
    cfg["strategy"] = strategy
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": first}]
    rec = {"id": run_id, "config": cfg, "started": time.time(), "steps": [], "submissions": [], "cost": 0.0, "status": "running",
           "usage": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}}
    save = lambda: (rdir / "run.json").write_text(json.dumps(rec, indent=1))  # noqa: E731
    # judge the starting copy once (baseline row)
    p0 = rdir / "submission_00.py"; p0.write_text(start)
    text0, rep0 = C.run_judge(target, variant, p0)
    cert0, ctext0 = certificate(target, start, rep0, variant)
    rec["start_certificate"] = cert0
    best, best_n = cert0, 0
    msgs[-1]["content"] += "\n\nCertificate of the starting copy:\n" + ctext0
    kernel = C.Kernel(target, variant, True, True, 0)
    cancelled = lambda: False  # noqa: E731
    try:
        for step in range(steps):
            content, thinking, calls, usage, finish = C.chat(msgs, C.TOOLS, effort, max_tokens, emit, cancelled)
            rec["usage"]["prompt_tokens"] += usage.get("prompt_tokens") or 0
            rec["usage"]["completion_tokens"] += usage.get("completion_tokens") or 0
            rec["usage"]["calls"] += 1; rec["cost"] += luminal.cost(usage)
            srec = {"step": step, "content": content, "thinking": thinking, "context_tokens": usage.get("prompt_tokens"), "tools": []}
            rec["steps"].append(srec)
            if not calls:
                rec["final_message"] = content
                break
            msgs.append({"role": "assistant", "content": content, "reasoning_content": thinking, "tool_calls": calls})
            for c in calls:
                name, code = c["function"]["name"], C.tool_code(c)
                if name == "submit":
                    n = len(rec["submissions"]) + 1
                    path = rdir / f"submission_{n:02d}.py"; path.write_text(code)
                    text, rep = C.run_judge(target, variant, path)
                    cert, ctext = certificate(target, code, rep, variant)
                    out = ctext + "\n\nJUDGE DETAIL (inside tests):\n" + judge_text(text)
                    rec["submissions"].append({"n": n, "step": step, "file": path.name, "certificate": cert})
                    if key(cert) > key(best):
                        best, best_n = cert, n
                    srec["tools"].append({"tool": "submit", "n": n, "certificate": cert})
                    emit({"type": "submit", "n": n, "cert": {k: cert.get(k) for k in ('fidelity', 'S', 'claims', 'claims_passed', 'beats_task', 'works')}})
                else:
                    res = kernel.run(code)
                    srec["tools"].append({"tool": "python", "code": code, **res})
                    out = res["output"] or "(no output: print what you want to see)"
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out + f"\n[{steps - step - 1} steps left]"})
            save()
            if (usage.get("prompt_tokens") or 0) > 95_000:
                c2, _, _, u2, _ = C.chat(msgs + [{"role": "user", "content": C.NOTES_ASK}], None, "low", 6000, emit, cancelled)
                rec["cost"] += luminal.cost(u2)
                bsrc = (rdir / f"submission_{best_n:02d}.py").read_text()
                msgs = [{"role": "system", "content": system}, {"role": "user", "content": task + "\n\nYOUR NOTES\n" + c2 +
                         f"\n\nYOUR BEST PROGRAM SO FAR (submission {best_n})\n```python\n{bsrc}```\nIts certificate:\n" +
                         certificate(target, bsrc, None, variant)[1] + "\n\nContinue replacing parts with rules."}]
                rec.setdefault("compactions", []).append({"at_step": step, "notes": c2})
        rec["status"] = "done"
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"; rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        kernel.close()
    rec["best"] = best_n; rec["best_certificate"] = best; rec["seconds"] = round(time.time() - rec["started"], 1)
    (rdir / "construction.py").write_text((rdir / f"submission_{best_n:02d}.py").read_text())
    save()
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target"); ap.add_argument("--notes-from"); ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--model"); ap.add_argument("--tag", default="r2"); ap.add_argument("--effort", default="high"); ap.add_argument("--strategy")
    a = ap.parse_args()
    r = run(a.target, a.notes_from, a.steps, a.model, a.tag, a.effort, strategy=a.strategy)
    print(json.dumps({"id": r["id"], "status": r["status"], "best": r["best"], "cost": round(r["cost"], 3),
                      "best_certificate": {k: r["best_certificate"].get(k) for k in ("fidelity", "task", "S", "claims", "claims_passed", "deep_claims_passed", "beats_task", "works")}}))
