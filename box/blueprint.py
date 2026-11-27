"""The blueprint data model (SPEC.md §5), as far as the checker needs it. INTERFACE.md, when the bench-writer has
written it, fixes the file format; `from_interface` will map it onto these classes.

A part is located in the network under an allowed linear map (SPEC R3): a subspace of a named activation site, given
either by residual-basis labels (`labels`) or by an explicit orthonormal basis (`basis`, shape [d, k]). Its value is
encoded in that subspace either `numerical` (one direction carrying the number) or `categorical` (one direction per
value, labelled `<name>:<value>`).
"""
from dataclasses import dataclass, field

import numpy as np

TYPES = ("rule", "lookup", "irrelevant", "unresolved")


@dataclass
class Part:
    id: str
    type: str                      # rule | lookup | irrelevant | unresolved
    vtype: str                     # the vocabulary type of its value (e.g. "SOp")
    site: str                      # named intervention site
    labels: list = None            # residual-basis labels spanning its subspace, or
    basis: np.ndarray = None       # an explicit orthonormal basis [d, k]
    encoding: str = "categorical"  # how its value sits in the subspace
    definition: str = None         # rules: an expression of the starting vocabulary (or the box's library)
    table: dict = None             # lookups: key -> value
    upstream: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


@dataclass
class Blueprint:
    domain: str                    # a vocabulary domain
    parts: list
    output: str                    # the part whose value is the network's output
    library: dict = field(default_factory=dict)   # invented primitives ($name)
    name: str = ""

    def part(self, pid):
        return next(p for p in self.parts if p.id == pid)

    def order(self):
        """Parts in dependency order; raises on a cycle or an unknown upstream part."""
        ids = {p.id for p in self.parts}
        done, out = set(), []
        pending = list(self.parts)
        while pending:
            progressed = False
            for p in list(pending):
                missing = [u for u in p.upstream if u not in ids]
                if missing:
                    raise ValueError(f"part {p.id}: unknown upstream {missing}")
                if all(u in done for u in p.upstream):
                    out.append(p); done.add(p.id); pending.remove(p); progressed = True
            if not progressed:
                raise ValueError("the wiring has a cycle: " + ", ".join(p.id for p in pending))
        return out
