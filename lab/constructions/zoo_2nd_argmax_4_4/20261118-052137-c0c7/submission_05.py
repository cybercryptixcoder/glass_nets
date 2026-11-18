
import numpy as np

DESCRIPTION = """Stage 1 (VERIFIED): h1 = x_1 split by sign; units 0,3 carry max(x1,0), units 1,2 carry max(-x1,0).
Probing a structural hypothesis for the rest: rows 1 and 2 of the recurrent matrix are nearly exact negatives
(their sum has norm 0.39 against row norms 3.8), so I use one row and its negative. Stages 2-4 are otherwise the raw recurrence."""

WX = np.array([0.9975, -2.2207, -2.0354, 0.3939])
R0 = np.array([-0.2137,  2.8348, -0.1426,  0.9157])
R1 = np.array([ 2.3765,  1.3635, -0.5099,  2.6610])
R3 = np.array([-0.0335,  0.4774,  2.4935, -0.7525])
WH = np.stack([R0, R1, -R1, R3])
WO = np.array([[ 1.6376,  0.6297, -2.5617,  0.5466],
               [ 1.0779, -0.5340, -1.0086,  0.6374],
               [-1.2079,  1.2053,  0.8859, -0.9309],
               [-2.5583, -0.5449,  3.3309, -0.8870]])

def model(x, iv):
    x = np.asarray(x, dtype=float)[:, :4]
    xp = iv("x1pos", np.maximum(x[:, 0], 0.0))
    xm = iv("x1neg", np.maximum(-x[:, 0], 0.0))
    h = np.stack([0.9975*xp, 2.2207*xm, 2.0354*xm, 0.3939*xp], 1)
    for t in range(1, 4):
        pre = x[:, t:t+1] * WX[None, :] + h @ WH
        h = iv("h%d" % (t+1), np.maximum(pre, 0.0))
    return h @ WO

ALIGN = {"x1pos": ["h1"], "x1neg": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"]}
