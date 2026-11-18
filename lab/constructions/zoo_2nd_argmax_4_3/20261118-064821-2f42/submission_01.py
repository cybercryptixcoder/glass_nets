
import numpy as np

DESCRIPTION = ("Step 1: the first hidden state is just relu(x1*W_x), i.e. two numbers p1=max(x1,0), "
"n1=max(-x1,0) scaled by the four input weights (units 0,1 fire when x1>0, units 2,3 when x1<0). "
"Later steps kept exact (temporary re-implementation) so the stage table shows a baseline: "
"h2=relu(x2*W_x+h1@W_h), h3=relu(x3*W_x+h2@W_h), out=h3@W_o.")

WX = np.array([3.9652629, 4.6994882, -4.7163196, -3.9652016])
WH = np.array([[1.3387, 1.9277, -2.3877, -1.8931],
               [-2.1996, -2.4339, 2.8057, 2.6429],
               [2.6352, 2.7868, -2.4339, -2.1912],
               [-1.8948, -2.3826, 1.9331, 1.3409]])
WO = np.array([[-1.6718, -18.7477, 11.4305],
               [1.5993, 15.9993, -15.7403],
               [1.6175, 15.9592, -15.6973],
               [-1.6575, -18.7243, 11.4769]])

def model(x, iv):
    p1 = iv("p1", np.maximum(x[:, 0], 0.0))
    n1 = iv("n1", np.maximum(-x[:, 0], 0.0))
    h1 = np.stack([np.abs(WX[0]) * p1, np.abs(WX[1]) * p1,
                   np.abs(WX[2]) * n1, np.abs(WX[3]) * n1], 1)
    h2 = iv("h2u", np.maximum(x[:, 1:2] * WX + h1 @ WH, 0.0))
    h3 = iv("h3u", np.maximum(x[:, 2:3] * WX + h2 @ WH, 0.0))
    return h3 @ WO

ALIGN = {"p1": ["h1"], "n1": ["h1"], "h2u": ["h2"], "h3u": ["h3"]}
