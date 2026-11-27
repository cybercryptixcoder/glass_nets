"""The reader test, live (SPEC R10, with the only model this project has: DeepSeek V4.1 Flash through Luminal, which is
weaker than the frontier reader the spec names). A reader given only the blueprint must predict the network's outputs
on fresh inputs. Controls, fixed before the run: the same reader with a scrambled blueprint (the definitions moved
between parts, so every symbol is still there but the account is wrong), and with no blueprint at all (told only that
this is a network on token sequences). Yes if the reader with the blueprint is far above both controls."""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rosetta import llm  # noqa: E402,F401  (loads the key from lab/.env into the environment)
from lab import luminal  # noqa: E402

PROMPT = """You are given {what} of a neural network that reads a sequence of tokens and writes one output at each
position. Use it to predict the network's output for each input below. Work it out carefully, then end your answer
with one line of JSON only: a list with one list of outputs per input, in order.

{blueprint}

Inputs:
{inputs}
"""


def ask(blueprint_text, what, inputs, effort="low"):
    prompt = PROMPT.format(what=what, blueprint=blueprint_text, inputs="\n".join(f"{i + 1}. {json.dumps(s)}" for i, s in enumerate(inputs)))
    r = luminal.complete({"messages": [{"role": "user", "content": prompt}], "max_tokens": 6000}, effort=effort)
    txt = r["choices"][0]["message"]["content"] or ""
    m = None
    for m in re.finditer(r"\[\s*\[.*?\]\s*\]", txt, re.S):
        pass
    try:
        pred = json.loads(m.group(0)) if m else None
    except Exception:  # noqa: BLE001
        pred = None
    return pred, txt, r.get("usage", {})


def score(pred, truth, numerical):
    hits = tot = 0
    for i, t in enumerate(truth):
        p = pred[i] if pred and i < len(pred) and isinstance(pred[i], list) else []
        for j, v in enumerate(t):
            tot += 1
            if j < len(p):
                try:
                    hits += (abs(float(p[j]) - float(v)) < 0.02) if numerical else (str(p[j]) == str(v))
                except (TypeError, ValueError):
                    pass
    return hits / max(tot, 1)


def scrambled(md, bp):
    """The same blueprint with definitions rotated among the parts that are not inputs."""
    import copy
    b = copy.deepcopy(bp)
    rules = [p for p in b.parts if p.type == "rule" and p.definition not in ("tokens", "indices")]
    defs = [p.definition for p in rules]
    for p, d in zip(rules, defs[1:] + defs[:1]):
        p.definition = d
    return b
