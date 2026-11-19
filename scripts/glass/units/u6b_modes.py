"""U6b: number of ReLU on/off patterns (modes) per step needed to cover 90% of inputs."""
import sys, json
from pathlib import Path
from collections import Counter
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "lab"))
import judge, nets
out = {}
for target in sys.argv[1:] or ["zoo_2nd_argmax_4_3", "zoo_2nd_argmax_8_5", "zoo_2nd_argmax_8_8", "zoo_2nd_argmax_16_10", "zoo_argmedian_8_5", "zoo_argmedian_16_5"]:
    for var in ("original", "shuffled"):
        meta, W, X = judge.load_target(target, var, 0); X = X[:8000]; T = X.shape[1]
        acts = {}; nets.forward("rnn_relu", W, X, acts=acts); row = []
        for t in range(1, T + 1):
            P = np.asarray(acts[f"h{t}"]).reshape(len(X), -1) > 0
            c = Counter(map(lambda r: r.tobytes(), P)); cum = 0; k = 0
            for _, m in c.most_common():
                cum += m; k += 1
                if cum >= 0.9 * len(X): break
            row.append(k)
        out[f"{target}|{var}"] = row
        print(f"{target:22s} {var:9s} patterns covering 90% per step: {row}", flush=True)
(ROOT / "results/glass/units/u6b.json").write_text(json.dumps(out))
