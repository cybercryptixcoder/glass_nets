
import numpy as np

DESCRIPTION = ("Stage 1 (h1) is exact: the first input enters only through its sign, "
               "h1 = [relu(x1), relu(-x1)] @ M with M the four input weights, i.e. "
               "unit0 = 4.716*relu(-x1), unit1 = 3.965*relu(x1), unit2 = 4.700*relu(x1), "
               "unit3 = 3.965*relu(-x1). Later stages are still crude (a linear readout of x).")

M1 = np.array([[0.0, 3.9653, 4.6995, 0.0],
               [-4.7163, 0.0, 0.0, -3.9652]])

CO = np.array([[-248.188, -84.951, 158.248],
               [3.345, 22.506, -21.814],
               [-245.919, -79.809, 148.575],
               [-17.237, -139.164, -13.348]])

def model(x, iv):
    x1 = x[:, 0]
    p = iv("x1pos", np.maximum(x1, 0.0))
    n = iv("x1neg", np.maximum(-x1, 0.0))
    h1 = iv("h1", np.stack([p, n], 1) @ M1)
    A = np.stack([np.ones(len(x)), x[:, 0], x[:, 1], x[:, 2]], 1)
    return A @ CO

ALIGN = {"x1pos": ["h1"], "x1neg": ["h1"]}
