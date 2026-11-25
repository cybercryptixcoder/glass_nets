"""Is the network baseline faithful? Re-score 40 already-scored random systems with
SPOCK's default sampling (samples=1000, max_model_samples=100) and compare medians."""
import os, numpy as np, rebound, warnings, json
warnings.filterwarnings("ignore")
from spock import DeepRegressor
P = os.path.expanduser("~/pdata")
rows = np.loadtxt(f"{P}/nn_random_batch.csv", delimiter=",", ndmin=2)
pick = rows[np.random.default_rng(4).choice(len(rows), 40, replace=False)]
sa = rebound.Simulationarchive(f"{P}/spock2/dataset/random/clean_initial_conditions.bin")
med, lo, hi = DeepRegressor().predict_instability_time([sa[int(i)] for i in pick[:, 0]], samples=1000, max_model_samples=100, Ncpus=1, seed=1)
a, b = np.clip(pick[:, 1], 4, 9), np.clip(np.log10(med), 4, 9)
out = {"n": 40, "mean_abs_diff_clipped": round(float(np.mean(np.abs(a - b))), 4), "max_abs_diff_clipped": round(float(np.max(np.abs(a - b))), 4), "corr": round(float(np.corrcoef(a, b)[0, 1]), 4)}
print(out); json.dump(out, open("results/planets/nn_settings_check.json", "w"))
