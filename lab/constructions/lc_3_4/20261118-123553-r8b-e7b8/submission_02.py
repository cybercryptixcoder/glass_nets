import numpy as np
DESCRIPTION = "probe"
def model(x, iv):
    v = iv("v", x[:, 0] * 1.0)
    return np.zeros((x.shape[0], 4))
ALIGN = {"v": ["embed"]}
