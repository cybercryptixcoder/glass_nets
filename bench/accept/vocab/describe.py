"""Print the domain tables of VOCABULARY.md from the code itself, so the document cannot drift from it.
Run: python3 bench/accept/vocab/describe.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vocab  # noqa: E402

for name in vocab.DOMAINS:
    d = vocab.get(name)
    c = vocab.count(d)
    print(f"### Domain `{name}`\n")
    print("Types: " + "; ".join(f"`{t}` ({m})" for t, m in d.types.items()) + "; `C` (a constant).")
    print(f"Root types (what a definition may be): {', '.join('`' + r + '`' for r in d.roots)}. **L = {d.L}.**\n")
    by_type = {}
    for a in d.atoms.values():
        by_type.setdefault(a.type, []).append(a.name)
    print("Base features:")
    for t, names in by_type.items():
        print(f"- `{t}`: " + ", ".join(f"`{n}`" for n in names))
    print("\nOperations (name: argument types -> result type):")
    for n, ovs in d.ops.items():
        sig = "; ".join(f"({' '.join(o.args)}) -> {o.ret}" for o in ovs)
        doc = next((o.doc for o in ovs if o.doc), "")
        print(f"- `{n}`: {sig}" + (f". {doc}" if doc else ""))
    print("\nExpression skeletons by length (constants as fitted slots; part references and invented primitives "
          "not counted):\n")
    print("| length | " + " | ".join(str(i + 1) for i in range(d.L)) + " | total |")
    print("|---" * (d.L + 2) + "|")
    print("| count | " + " | ".join(f"{x:,}" for x in c["by_length"]) + f" | {c['total']:,} |\n")
