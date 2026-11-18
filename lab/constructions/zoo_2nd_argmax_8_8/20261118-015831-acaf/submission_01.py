
import numpy as np

DESCRIPTION = """Stage 1 (h1) explained exactly: the inputs are independent standard normals; the
first hidden state is h1 = relu(x1 * W_x), so it depends only on the sign and size of x1. Splitting into
positive and negative part of x1 gives two variables; every h1 unit is exactly a linear combination of
them (unit u = w_u * pos - w_u * neg). Later stages not yet explained: the construction currently makes
a placeholder prediction for the output (treated as unrelated to the state), so only the h1 stage should
verify."""

def model(x, iv):
    pos = iv("x1_pos", np.maximum(x[:, 0], 0.0))
    neg = iv("x1_neg", np.maximum(-x[:, 0], 0.0))
    # h1 = pos * w+ + neg * w-
    h1 = pos[:, None] * WX_POS[None, :] + neg[:, None] * WX_NEG[None, :]
    out = np.zeros((x.shape[0], 8))
    return out

WX = np.array([-0.207, 0.03, -1.203, -0.151, -0.467, 0.003, 0.619, 0.004])
WX_POS = np.maximum(WX, 0.0)
WX_NEG = np.minimum(WX, 0.0)

ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"]}
