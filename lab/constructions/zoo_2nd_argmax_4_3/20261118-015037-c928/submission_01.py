import numpy as np
DESCRIPTION = ("Partial construction. Step 1 exactly: h1 is a signed-magnitude code of x1. "
               "Hidden units come in two branches; when x1>0 units 1,2 are active with "
               "h1=(3.9653*x1, 4.6995*x1,0,0); when x1<0 units 3,4 are active with "
               "h1=(0,0,-4.7163*x1,-3.9652*x1). So h1 = A*relu(x1) + B*relu(-x1). "
               "Later stages not yet explained; placeholder output.")
def model(x, iv):
    p = iv("x1_pos", np.maximum(x[:, 0], 0.0))
    n = iv("x1_neg", np.maximum(-x[:, 0], 0.0))
    out = np.stack([x[:, 1] - x[:, 0], x[:, 2], -x[:, 1]], axis=1)
    return out
ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"]}
