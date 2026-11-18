import numpy as np

DESCRIPTION = """
Draft 3, mechanism probe. No biases, so h1 = relu(x1*W_x) is exactly the sign-magnitude code of
input 1: relu(x1) in units {0,2,4,6,7} and relu(-x1) in units {1,3,5}.
At every later step the state is driven by the current input (large input gain) plus the previous
state, so each h_t is tested here against the ten clipped-input quantities relu(+/-x_i): positive and
negative parts of every input. The output is still a placeholder until the readout stage is resolved.
"""

Wx = np.array([0.643, -6.421, 0.423, -6.574, 0.045, -4.913, -0.054, 7.002])
Wp = np.maximum(Wx, 0.0)
Wn = np.maximum(-Wx, 0.0)


def model(x, iv):
    pos = iv("pos1", np.maximum(x[:, 0], 0.0))
    neg = iv("neg1", np.maximum(-x[:, 0], 0.0))
    h1 = pos[:, None] * Wp[None, :] + neg[:, None] * Wn[None, :]
    feats = [pos, neg]
    for t in range(1, 5):
        feats.append(np.maximum(x[:, t], 0.0))
        feats.append(np.maximum(-x[:, t], 0.0))
    for t in range(2, 6):
        for j in range(t):
            iv("p%d" % (j + 1), feats[2 * j]) if t == 2 else None
        if t > 2:
            for j in range(t):
                iv("p%d_%d" % (j + 1, t), feats[2 * j])
                iv("n%d_%d" % (j + 1, t), feats[2 * j + 1])
    out = np.zeros((x.shape[0], 5))
    out[:, 0] = h1.sum(1)
    return out


ALIGN = {"pos1": ["h1"], "neg1": ["h1"],
         "p1_3": ["h2", "h3", "h4", "h5"], "n1_3": ["h2", "h3", "h4", "h5"]}
