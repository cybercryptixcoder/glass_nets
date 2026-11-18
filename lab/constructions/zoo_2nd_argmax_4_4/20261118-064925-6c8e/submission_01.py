
import numpy as np
DESCRIPTION = "Literal recurrent construction: h_t = relu(x_t*Wx + h_{t-1}@Wh), out = h4@Wo. Baseline submission, no compression yet."
Wx = np.array([0.997,-2.221,-2.035,0.394])
Wh = np.array([[-0.214,2.835,-0.143,0.916],
               [2.433,1.320,-0.489,2.523],
               [-2.170,-1.407,0.531,-2.799],
               [-0.033,0.477,2.494,-0.753]])
Wo = np.array([[1.638,0.630,-2.562,0.547],
               [1.078,-0.534,-1.009,0.637],
               [-1.208,1.205,0.886,-0.931],
               [-2.558,-0.545,3.331,-0.887]])
def model(x, iv):
    h = np.zeros((x.shape[0],4))
    for t in range(4):
        h = np.maximum(x[:,t:t+1]*Wx + h@Wh, 0.0)
        h = iv("h%d"%(t+1), h)
    return h@Wo
ALIGN = {"h1":["h1"],"h2":["h2"],"h3":["h3"],"h4":["h4"]}
