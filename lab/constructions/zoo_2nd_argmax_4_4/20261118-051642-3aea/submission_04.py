
import numpy as np
DESCRIPTION = ("Same recurrence in the gauge where each hidden unit u is rescaled by 1/|Wx_u|: then the input vector "
  "becomes Wx=(+1,-1,-1,+1) so step 1 is h1 = relu(x1*(+1,-1,-1,+1)) = (u,m,m,u), u=relu(x1), m=relu(-x1) -- no constants. "
  "The recurrent matrix in that gauge is Wh' = diag(1/|Wx|) Wh diag(1/|Wx|)^-1, and the readout becomes "
  "Wo'[u,:] = Wo[u,:]*|Wx_u|. Only class-score differences matter, so class 3 is the zero reference, and one overall "
  "scale is fixed by D0[0]=1: the readout needs 11 numbers.")
Wh = np.array([[-0.2137, 1.2733, -0.0699, 2.3190],
               [5.4173, 1.3203, -0.5339, 14.2247],
               [-4.4272, -1.2893, 0.5307, -14.4622],
               [-0.0132, 0.0847, 0.4826, -0.7525]])
D0 = np.array([1.0, 0.89885, -0.51820, -0.60500])
D1 = np.array([0.07609, -2.39047, 3.99545, 0.12384])
D2 = np.array([-2.84920, -3.35906, 3.39800, 1.52672])
S = np.array([1.0, -1.0, -1.0, 1.0])

def model(x, iv):
    B = x.shape[0]
    u = np.maximum(x[:, 0], 0.0); m = np.maximum(-x[:, 0], 0.0)
    h = iv("h1", np.stack([u, m, m, u], axis=1))
    for t in range(1, 4):
        h = iv("h%d" % (t+1), np.maximum(x[:, t:t+1]*S + h @ Wh, 0))
    z = np.zeros(B)
    return np.stack([h @ D0, h @ D1, h @ D2, z], axis=1)
ALIGN = {"h1": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"]}
