"""PL1: the neural network baseline (SPOCK DeepRegressor, Cranmer et al. 2021) on a fixed subset of the random dataset.
Subset: 250 systems drawn with seed 0 among random-dataset systems that survive the first 1e4 orbits (log10 t >= 4).
Writes $PDATA/nn_random_subset.csv incrementally (index, median, lower, upper)."""
import os, sys, time, warnings
import numpy as np, rebound
warnings.filterwarnings("ignore")
from spock import DeepRegressor
PDATA = os.path.expanduser("~/pdata")
z = np.load(os.path.join(PDATA, "random_ic.npz"))
lt = np.log10(z["t"])
cand = np.where(lt >= 4)[0]
idx = np.sort(np.random.default_rng(0).choice(cand, int(os.environ.get("N", "250")), replace=False))
sa = rebound.Simulationarchive(os.path.join(PDATA, "spock2", "dataset", "random", "clean_initial_conditions.bin"))
m = DeepRegressor()
out = os.path.join(PDATA, "nn_random_subset.csv")
done = set()
if os.path.exists(out):
    done = {int(l.split(",")[0]) for l in open(out) if l[0].isdigit()}
f = open(out, "a")
t0 = time.time()
for k, i in enumerate(idx):
    if int(i) in done:
        continue
    r = m.predict_instability_time(sa[int(i)], samples=40)
    med, lo, hi = [float(np.asarray(x).ravel()[0]) for x in r]
    f.write(f"{i},{med},{lo},{hi}\n"); f.flush()
    if k % 25 == 0:
        print(k, f"{time.time() - t0:.0f}s", flush=True)
