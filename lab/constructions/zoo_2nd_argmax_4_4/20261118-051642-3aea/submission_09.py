
import numpy as np
DESCRIPTION = ("Recurrence in the unit gauge where each hidden unit u is rescaled by 1/|Wx_u|, making the input vector "
  "Wx = (+1,-1,-1,+1). Step 1 is 2 numbers only: h1 = relu(x1*(+1,-1,-1,+1)) = (u,m,m,u) with u=relu(x1), m=relu(-x1) "
  "-- no constants at all, and the site h1 carries just the two variables (u,m). Later steps add x_t*(+1,-1,-1,+1) to the "
  "previous state mapped by the rescaled recurrent matrix Wh' and threshold. The decision is a 4-way argmax, so only "
  "class-score differences matter: class 3 is the zero reference, and the readout is then 12 numbers of which the four that "
  "sit on the simple grid (1, -1/2, 4, 7/2) are written exactly, leaving 8. "
  "Wh' = diag(1/|Wx|) Wh diag(|Wx|), readout' = diag(|Wx|) Wo, differences against class 3.")
Wh = np.array([[-0.2137, 1.2733, -0.0699, 2.3190],
               [5.4173, 1.3203, -0.5339, 14.2247],
               [-4.4272, -1.2893, 0.5307, -14.4622],
               [-0.0132, 0.0847, 0.4826, -0.7525]])
D0 = np.array([1.0, 0.88994, -0.5, -0.60534])
D1 = np.array([0.0692, -2.39484, 4.0, 0.12786])
D2 = np.array([-2.92976, -3.45803, 3.5, 1.56954])
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
