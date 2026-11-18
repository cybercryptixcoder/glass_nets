
import numpy as np
DESCRIPTION = ("The recurrence in a normalised gauge. Rescaling each hidden unit u by 1/|Wx_u| (which leaves every "
  "ReLU decision unchanged) makes the input vector Wx = (+1,-1,-1,+1), so step 1 is h1 = relu(x1*(+1,-1,-1,+1)) = "
  "(u,m,m,u) with u=relu(x1), m=relu(-x1): two numbers, no constants. Later steps add x_t*(+1,-1,-1,+1) to the "
  "state mapped by the rescaled recurrent matrix and threshold. The readout only needs class-score differences "
  "(subtracting class 3's column leaves the argmax unchanged) and one overall scale, i.e. 11 numbers for 4 classes.")
Wh = np.array([[-0.21424, 2.84190, -0.14296, 0.91800],
               [1.09573, 0.59454, -0.22033, 1.13617],
               [-1.06607, -0.69117, 0.26076, -1.37517],
               [-0.08505, 1.21199, 6.33029, -1.91038]])
D0 = np.array([1.0, 0.898872, -0.518071, -0.604924])
D1 = np.array([0.076168, -2.390316, 3.995333, 0.123823])
D2 = np.array([-2.849033, -3.358780, 3.397956, 1.526662])
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
