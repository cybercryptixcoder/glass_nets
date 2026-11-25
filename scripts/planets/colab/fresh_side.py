"""Colab side jobs for the fresh set: short features or network predictions for items lo..hi."""
import os, sys, numpy as np, rebound, warnings
warnings.filterwarnings("ignore")
mode, w, W, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
sa = rebound.Simulationarchive("/content/pdata/fresh.bin")
items = [i for i in range(lo, hi) if i % W == w]
if mode == "feat":
    from spock import FeatureClassifier
    KEYS = ["EMcrossnear","EMfracstdnear","EPstdnear","MMRstrengthnear","EMcrossfar","EMfracstdfar","EPstdfar","MMRstrengthfar","MEGNO","MEGNOstd","Tsec"]
    m = FeatureClassifier(); out = open(f"/content/featfresh_w{w}_{lo}.csv", "a")
    for k in range(0, len(items), 10):
        ch = items[k:k+10]
        res = m.generate_features([sa[i] for i in ch], n_jobs=1)
        if isinstance(res, tuple): res = [res]
        for i, (lst, ok) in zip(ch, res):
            d = lst[0]; out.write(f"fresh,{i}," + ",".join(f"{float(d[q]):.6g}" for q in KEYS) + f",{int(bool(ok))}\n")
        out.flush()
else:
    from spock import DeepRegressor
    m = DeepRegressor(); out = open(f"/content/nnfresh_w{w}_{lo}.csv", "a")
    for k in range(0, len(items), 25):
        ch = items[k:k+25]
        med, l, u = m.predict_instability_time([sa[i] for i in ch], samples=300, max_model_samples=30, Ncpus=1, seed=0)
        for i, a, b, c in zip(ch, med, l, u): out.write(f"{i},{np.log10(a):.4f},{np.log10(b):.4f},{np.log10(c):.4f}\n")
        out.flush()
