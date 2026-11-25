"""PL1: initial orbital elements and physics features for the SPOCK 2.0 datasets (ledger row PL1).

    python3 -I scripts/planets/features.py            # writes $PDATA/{resonant,random}_ic.npz

Data: Zenodo 14327471 (Thadhani et al. 2025), unpacked in $PDATA/spock2/dataset/{resonant,random}/.
Only CSV labels and REBOUND simulation archives are read (no pickles).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import rebound

PDATA = os.environ.get("PDATA", os.path.expanduser("~/pdata"))


def elements(sim):
    """Masses, semimajor axes, eccentricities, inclinations, pericentre longitudes, mean longitudes, periods of planets 1-3."""
    ps = sim.particles
    M = ps[0].m
    out = []
    for p in ps[1:4]:
        o = p.orbit(primary=ps[0])
        out.append([p.m / M, o.a, o.e, o.inc, o.pomega, o.l, o.P, o.Omega])
    return np.array(out)        # 3 x 8


def load(name):
    d = os.path.join(PDATA, "spock2", "dataset", name)
    lab = pd.read_csv(os.path.join(d, "sim_labels.csv"), index_col=0)
    sa = rebound.Simulationarchive(os.path.join(d, "clean_initial_conditions.bin"))
    n = len(sa)
    assert n == len(lab), (n, len(lab))
    E = np.zeros((n, 3, 8))
    t0 = time.time()
    for i in range(n):
        E[i] = elements(sa[i])
        if i % 20000 == 0:
            print(name, i, f"{time.time() - t0:.0f}s", flush=True)
    return E, lab


def main():
    for name in ("resonant", "random"):
        E, lab = load(name)
        P1 = E[:, 0, 6]
        t = lab["instability_time"].values / P1
        ts = lab["shadow_instability_time"].values / P1
        np.savez(os.path.join(PDATA, f"{name}_ic.npz"), E=E, t=t, t_shadow=ts, stable=lab["Stable"].values.astype(bool),
                 runstring=lab["runstring"].values.astype(str))
        print(name, len(t), "P1 range", P1.min(), P1.max(), "log t range", np.log10(t).min(), np.log10(t).max(), flush=True)


if __name__ == "__main__":
    main()
