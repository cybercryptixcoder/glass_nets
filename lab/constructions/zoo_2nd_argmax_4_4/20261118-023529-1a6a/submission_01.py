
import numpy as np
DESCRIPTION = ("First pass. The network is homogeneous of degree 1: relu(x_t*Wx + h_prev@Wh) with no bias, so out(c*x)=c*out(x) for c>0 and only the direction of x matters. "
               "h1 = relu(x1*Wx) with Wx = [0.998,-2.221,-2.035,0.394] (units 0,3 positive, 1,2 negative; units 1 and 2 proportional, units 0 and 3 proportional). "
               "Each later step adds the input injected along the same direction Wx and mixes the running state through the 4x4 recurrent matrix Wh, then rectifies. "
               "The 4 outputs are a linear read (Wo) of the final state. This version writes the recurrence with measured constants; later versions will compress it.")
WX = np.array([0.9975, -2.2207, -2.0354, 0.3939])
WH = np.array([[-0.2137, 2.8348, -0.1426, 0.9157],
               [2.4333, 1.3203, -0.4893, 2.5231],
               [-2.1697, -1.4067, 0.5307, -2.7988],
               [-0.0335, 0.4774, 2.4935, -0.7525]])
WO = np.array([[1.6376, 0.6297, -2.5617, 0.5466],
               [1.0779, -0.534, -1.0086, 0.6374],
               [-1.2079, 1.2053, 0.8859, -0.9309],
               [-2.5583, -0.5449, 3.3309, -0.887]])

def model(x, iv):
    h = iv("h1", np.maximum(x[:, 0:1] * WX, 0))
    h = iv("h2", np.maximum(x[:, 1:2] * WX + h @ WH, 0))
    h = iv("h3", np.maximum(x[:, 2:3] * WX + h @ WH, 0))
    h = iv("h4", np.maximum(x[:, 3:4] * WX + h @ WH, 0))
    return h @ WO

ALIGN = {"h1": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"]}
