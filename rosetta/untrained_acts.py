"""Untrained control for the SAE route: LD2 with its weights permuted (seed 21), site activations for the engine's
search and scoring rows. Writes acts_LD2_s<SITE>_untrained.npy [n, 128, 64] and the row list."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rosetta import run_ld2 as L  # noqa: E402
from rosetta.engine import Defs  # noqa: E402
D = Defs(L.P, L.names, L.TR, L.TE)
rows = np.concatenate([D.rs, D.rt]); H = L.site_acts(L.get_net(untrained_seed=21), rows)
np.save(os.path.join(L.R, f"acts_LD2_s{L.SITE}_untrained.npy"), H.reshape(len(rows), 128, 64).astype(np.float16))
np.save(os.path.join(L.R, f"rows_LD2_s{L.SITE}_untrained.npy"), rows)
print("saved", H.shape)
