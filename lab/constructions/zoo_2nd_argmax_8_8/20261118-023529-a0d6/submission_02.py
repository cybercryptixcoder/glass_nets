
import numpy as np

DESCRIPTION = ("First real (non-copy) attempt. Each input number enters the state only through the single scalar x_t (W_x is 1x8), "
               "so at step 1 the state is exactly x1*relu(Wx) + x1*relu(-Wx), i.e. two seeds: the positive part and the negative part of x1. "
               "Because the state stays positive after a couple of steps the recurrence is essentially linear and amplifying, so each class score is "
               "an affine gain on its own input number minus a dilution term from the numbers that come after it. "
               "Only stage 1 (h1) is explained here; the readout is a fitted rule, not the weight matrix.")

A = np.array([4.70, 7.19, 6.53, 6.73, 5.86, 4.62, 3.84, 4.00])
D = np.array([-4.90, -6.80, -7.60, -7.60, -7.30, -5.90, -5.90, -5.80])
B = 0.75

def model(x, iv):
    p = iv("x1_pos", np.maximum(x[:, 0], 0.0))
    n = iv("x1_neg", np.minimum(x[:, 0], 0.0))
    xs = np.concatenate([(p + n)[:, None], x[:, 1:]], axis=1)
    # dilution: score of class c loses B for every later input number
    tail = np.zeros_like(xs)
    s = np.zeros(xs.shape[0])
    for j in range(7, -1, -1):
        tail[:, j] = s
        s = s + xs[:, j]
    return D + A * xs - B * tail

ALIGN = {"x1_pos": ["h1"], "x1_neg": ["h1"]}
