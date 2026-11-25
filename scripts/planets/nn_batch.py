"""Network baseline (SPOCK DeepRegressor = Cranmer et al. 2021 BNN) on a random
subset of the held-out 'random' dataset, batched. Appends
index,log10 median,log10 lower,log10 upper to ~/pdata/nn_random_batch.csv
(resumable). Settings: samples=300, max_model_samples=30 (documented in RESULTS)."""
import os, numpy as np, rebound, warnings
warnings.filterwarnings("ignore")
from spock import DeepRegressor

P = os.path.expanduser(os.environ.get("PDATA", "~/pdata"))
NAME = os.environ.get("SET", "random")
z = np.load(f"{P}/{NAME}_ic.npz"); lt = np.log10(z["t"]); cand = np.where(lt >= 4)[0]
N = int(os.environ.get("N", "3000"))
idx = np.sort(np.random.default_rng(1).choice(cand, min(N, len(cand)), replace=False))
out = f"{P}/nn_{NAME}_batch.csv"
done = set()
if os.path.exists(out):
    done = {int(l.split(",")[0]) for l in open(out) if l[:1].isdigit()}
sa = rebound.Simulationarchive(f"{P}/spock2/dataset/{NAME}/clean_initial_conditions.bin")
m = DeepRegressor()
todo = [int(i) for i in idx if int(i) not in done]
B = 50
for k in range(0, len(todo), B):
    chunk = todo[k:k + B]
    sims = [sa[i] for i in chunk]
    med, lo, hi = m.predict_instability_time(sims, samples=300, max_model_samples=30, Ncpus=1, seed=0)
    with open(out, "a") as f:
        for i, a, b, c in zip(chunk, med, lo, hi):
            f.write(f"{i},{np.log10(a):.4f},{np.log10(b):.4f},{np.log10(c):.4f}\n")
    print(k + len(chunk), "of", len(todo), flush=True)
