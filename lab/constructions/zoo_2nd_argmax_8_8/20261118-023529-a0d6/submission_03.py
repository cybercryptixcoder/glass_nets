
import numpy as np

DESCRIPTION = ("Test of a hypothesis: the network has no biases, so out is a homogeneous piecewise-linear function of the input. "
               "Measuring out on single-coordinate inputs (x = +/- e_j) shows a strong diagonal: a negative x_c costs class c about 5-14 units, "
               "and a nonzero x_j costs every LATER class about 2 units. So class c wins when x_c is positive and no earlier number was large. "
               "Model: out_c = -a_c*relu(-x_c) - 2*sum_{j<c} |x_j| - small own-positive cost. Stage 1 (h1 = the two seeds relu(+/-x1*Wx)) is the only site named.")

a = np.array([10.8, 13.9, 12.6, 11.8, 9.5, 6.8, 5.7, 5.3])
p = np.array([1.2, 0.6, 0.5, 0.3, 0.3, 0.1, 0.2, 0.1])

def model(x, iv):
    xp = iv("x1_pos", np.maximum(x[:, 0], 0.0))
    xn = iv("x1_neg", np.minimum(x[:, 0], 0.0))
    xs = np.concatenate([(xp + xn)[:, None], x[:, 1:]], axis=1)
    neg = np.maximum(-xs, 0.0)
    pos = np.maximum(xs, 0.0)
    # prefix cost: every number hurts the classes after it
    pre = np.zeros_like(xs)
    s = np.zeros(xs.shape[0])
    for j in range(8):
        pre[:, j] = s
        s = s + np.abs(xs[:, j])
    return -a * neg - p * pos - 2.0 * pre

ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"]}
