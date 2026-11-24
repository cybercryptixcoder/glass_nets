"""The automated scientist: an LLM that proposes hypotheses as executable code. Kepler's frozen tests judge them.

Uses the lab's Luminal client (DeepSeek V4.1 Flash, `lab/luminal.py`); the key is read from the gitignored lab/.env.
Every prompt and answer is saved under results/kepler/<target>/scientist/ so the trail is reviewable.
"""
import json
import os
import re
import sys
import time

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def _load_env():
    path = os.path.join(REPO, "lab", ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def ask(prompt, system=None, effort="high", max_tokens=12000, log_dir=None, tag="q"):
    _load_env()
    sys.path.insert(0, os.path.join(REPO, "lab"))
    import luminal
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    t0 = time.time()
    r = luminal.complete({"model": luminal.model(), "messages": msgs, "max_tokens": max_tokens}, effort=effort)
    msg = r["choices"][0]["message"]
    text = msg.get("content") or ""
    usage = r.get("usage", {})
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"{tag}.json"), "w") as f:
            json.dump({"system": system, "prompt": prompt, "answer": text,
                       "reasoning": msg.get("reasoning_content") or "", "usage": usage,
                       "seconds": round(time.time() - t0, 1), "model": luminal.model(), "effort": effort}, f, indent=1)
    return text, usage


def extract_json(text):
    """The last JSON array or object in a model answer (inside ```json fences or bare)."""
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    for b in reversed(blocks + [text]):
        b = b.strip()
        for opener, closer in (("[", "]"), ("{", "}")):
            i, j = b.find(opener), b.rfind(closer)
            if i >= 0 and j > i:
                try:
                    return json.loads(b[i:j + 1])
                except Exception:
                    continue
    return None
