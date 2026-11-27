"""The blueprint as a reader sees it (SPEC R6, R10): a top level that fits on a page, every part with where it lives in
the network, its definition, what it reads and how it did under intervention, the invented primitives with their
provenance, and the unresolved ledger. Plain markdown, plus a JSON copy."""
import json

import numpy as np

TOK_SEMANTICS = """How to read the definitions (token sequences; every value is computed at each position of the input):
- `tokens` is the token at each position; `indices` is the position (0, 1, 2, ...).
- `(select KEYS QUERIES PRED)` makes, for each query position q, the set of key positions k where PRED(KEYS[k], QUERIES[q])
  holds. PRED is EQ, NEQ, LT (key < query), LEQ, GT, GEQ, TRUE or FALSE.
- `(aggregate SEL VALUES)` at q is the mean of VALUES over the selected positions (0 if none is selected).
- `(aggregate_cat SEL VALUES)` at q is the one value all the selected positions hold.
- `(selector_width SEL)` at q is the number of selected positions.
- `(map F X)` applies F at each position: `id`, `inc` (x+1), `dec` (x-1), `neg`, `(eqc c)` (1 if x == c else 0), and so on.
- `(sequence_map F X Y)` applies F to X and Y at each position: `plus`, `minus` (X - Y), `times`, `equal`, ...
- `@vN` is the value of part vN; `$name` is an invented operation, applied at each position (its code is listed).
"""


def to_markdown(name, bp, report, semantics=TOK_SEMANTICS):
    lines = [f"# Blueprint: {name}", ""]
    lines.append(f"**Verdict: {report['verdict']}.** " + "; ".join(f"{c[0]}: {c[2]}" for c in report["checks"]
                                                                 if c[0] in ("run alone",)))
    if report.get("iia_line") is not None:
        lines.append(f"Family-wise control line for the intervention tests: {report['iia_line']:.3f}.")
    out = bp.part(bp.output)
    lines += ["", "## Top level", "", f"The network's output is part `{out.id}`: `{out.definition}`.", ""]
    lines += ["## Parts, in the order the network computes them", "",
              "| part | where | kind | definition | reads | intervention test |", "|---|---|---|---|---|---|"]
    checks = {c[0]: c for c in report["checks"]}
    for p in bp.order():
        where = f"{p.site}, {p.basis.shape[1] if p.basis is not None else len(p.labels or [])} direction(s)"
        test = checks.get(f"interventions {p.id}") or checks.get(f"input {p.id}")
        t = "" if test is None else (("pass: " if test[1] else "FAIL: ") + test[2])
        d = p.definition if p.type == "rule" else f"*{p.type}*"
        lines.append(f"| `{p.id}` | {where} | {p.encoding} | `{d}` | {', '.join('`' + u + '`' for u in p.upstream) or 'the input'} | {t} |")
    if bp.library:
        lines += ["", "## Invented primitives", "", "| name | code | provenance |", "|---|---|---|"]
        for n, e in bp.library.items():
            code = e.get("source", e.get("definition", "")).strip().replace("\n", " ")
            lines.append(f"| `${n}` | `{code}` | {e.get('provenance', '')} |")
    unresolved = [p for p in bp.parts if p.type == "unresolved"]
    lines += ["", "## Unresolved ledger", ""]
    lines.append("Nothing is unresolved." if not unresolved else
                 "\n".join(f"- `{p.id}` at {p.site}: best try `{p.evidence.get('best_try')}` ({p.evidence.get('best_score')})"
                           for p in unresolved))
    if semantics:
        lines += ["", "## How to read it", "", semantics]
    return "\n".join(lines) + "\n"


def to_json(bp):
    return {"domain": bp.domain, "output": bp.output, "library": bp.library,
            "parts": [{"id": p.id, "type": p.type, "site": p.site, "encoding": p.encoding, "definition": p.definition,
                       "upstream": p.upstream, "directions": None if p.basis is None else
                       [int(i) for i in np.nonzero(np.abs(np.asarray(p.basis)).sum(1) > 0)[0]]} for p in bp.parts]}


def save(name, bp, report, folder):
    import os
    os.makedirs(folder, exist_ok=True)
    md = to_markdown(name, bp, report)
    open(os.path.join(folder, f"{name}.md"), "w").write(md)
    json.dump(to_json(bp), open(os.path.join(folder, f"{name}.json"), "w"), indent=1)
    return md
