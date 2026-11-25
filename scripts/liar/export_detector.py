"""Write the within-model K1 detector (score_k.fit on the model's own train split) to JSON for mech.py.
Usage: python3 -I export_detector.py DATA_DIR TAG OUT_JSON [NEG]  (NEG = truth for K1, failed_lie for LT2 T1)"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_k
d, tag, out = sys.argv[1:4]
neg = sys.argv[4] if len(sys.argv) > 4 else "truth"
D = score_k.load(d, tag)
sc, clf, s_mean = score_k.fit(D, neg=neg)
json.dump(dict(tag=tag, neg=neg, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=clf.coef_[0].tolist(),
               intercept=float(clf.intercept_[0])), open(out, "w"))
print("wrote", out)
