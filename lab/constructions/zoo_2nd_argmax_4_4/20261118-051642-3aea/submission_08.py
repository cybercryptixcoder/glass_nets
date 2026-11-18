
import numpy as np
DESCRIPTION = ("Recurrence in the unit gauge where each hidden unit u is rescaled by 1/|Wx_u|, making the input vector "
  "Wx = (+1,-1,-1,+1). Step 1 is 2 numbers only: h1 = relu(x1*(+1,-1,-1,+1)) = (u,m,m,u) with u=relu(x1), m=relu(-x1) "
  "-- no constants at all, and the site h1 carries just the two variables (u,m). Later steps add x_t*(+1,-1,-1,+1) to the "
  "previous state mapped by the rescaled recurrent matrix Wh' and threshold. Only class-score differences matter, so class 3 "
  "is the zero reference; the readout is written with its two smallest-magnitude entries rounded to the nearby simple values "
  "(class1/hidden2 = 4, class0/hidden2 = -1/2), which changes no decision but removes two numbers. "
  "Wh' = diag(1/|Wx|) Wh diag(|Wx|), readout' = diag(|Wx|) Wo.")
Wh = np.array([[-0.2137, 1.2733, -0.0699, 2.3190],
               [5.4173, 1.3203, -0.5339, 14.2247],
               [-4.4272, -1.2893, 0.5307, -14.4622],
               [-0.0132, 0.0847, 0.4826, -0.7525]])
D0 = np.array([1.025, 0.906, -0.5, -0.62])
D1 = np.array([0.094, -2.382, 4.0, 0.114])
D2 = np.array([-2.858, -3.367, 3.406, 1.531])
S = np.array([1.0, -1.0, -1.0, 1.0])

def model(x, iv):
    B = x.shape[0]
    u = np.maximum(x[:, 0], 0.0); m = np.maximum(-x[:, 0], 0.0)
    h1v = iv("h1", np.stack([u, m], axis=1))
    h = np.stack([h1v[:, 0], h1v[:, 1], h1v[:, 1], h1v[:, 0]], axis=1)
    for t in range(1, 4):
        h = iv("h%d" % (t+1), np.maximum(x[:, t:t+1]*S + h @ Wh, 0))
    z = np.zeros(B)
    return np.stack([h @ D0, h @ D1, h @ D2, z], axis=1)
ALIGN = {"h1": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"]}
