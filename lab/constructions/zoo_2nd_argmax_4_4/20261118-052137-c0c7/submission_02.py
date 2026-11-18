
import numpy as np

DESCRIPTION = """Step 1 of the recurrence is just x_1: h1 = x_1*W_x gated by sign, i.e. two signed copies of x_1
(h1 = [0.9975*max(x1,0), 2.2207*max(-x1,0), 2.0354*max(-x1,0), 0.3939*max(x1,0)]), so a single scalar per sign.
The remaining steps are still the raw recurrence (diagnostic re-description) and will be compressed next."""

WX = np.array([0.9975, -2.2207, -2.0354, 0.3939])
WH = np.array([[-0.2137,  2.8348, -0.1426,  0.9157],
               [ 2.4333,  1.3203, -0.4893,  2.5231],
               [-2.1697, -1.4067,  0.5307, -2.7988],
               [-0.0335,  0.4774,  2.4935, -0.7525]])
WO = np.array([[ 1.6376,  0.6297, -2.5617,  0.5466],
               [ 1.0779, -0.5340, -1.0086,  0.6374],
               [-1.2079,  1.2053,  0.8859, -0.9309],
               [-2.5583, -0.5449,  3.3309, -0.8870]])

def model(x, iv):
    x = np.asarray(x, dtype=float)[:, :4]
    xp = iv("x1pos", np.maximum(x[:, 0], 0.0))
    xm = iv("x1neg", np.maximum(-x[:, 0], 0.0))
    h1 = iv("h1", np.stack([0.9975*xp, 2.2207*xm, 2.0354*xm, 0.3939*xp], 1))
    h = h1
    for t in range(1, 4):
        pre = x[:, t:t+1] * WX[None, :] + h @ WH
        h = iv("h%d" % (t+1), np.maximum(pre, 0.0))
    return h @ WO

ALIGN = {"x1pos": ["h1"], "x1neg": ["h1"], "h1": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"]}
