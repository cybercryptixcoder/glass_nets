
import numpy as np

DESCRIPTION = ("One-layer ReLU RNN implementing 'which of the three inputs is the median'. "
 "Step 1: the state is x1*W_x clipped at 0, so it carries x1+ and x1- (two lobes of W_x). "
 "Step 2 and 3: add the next input through W_x and the previous state through W_h, then ReLU. "
 "The network is (nearly exactly) symmetric under x -> -x with the hidden vector reversed: "
 "W_h[left-reversed] = W_h and W_o has rows (a,b,b,a), so only two rows of W_h and two rows "
 "of W_o are independent. Readout: out = h3 @ W_o.")

Wx = np.array([3.965263, 4.699488, -4.71632, -3.965202])
r0 = np.array([1.338687, 1.927741, -2.387666, -1.893123])
r1 = np.array([-2.199573, -2.433872, 2.805652, 2.642897])
Wh = np.array([r0, r1, r1[::-1], r0[::-1]])          # reversal symmetry
o0 = np.array([-1.664693, -18.736006, 11.453696])     # mean of W_o rows 0 and 3
o1 = np.array([1.608388, 15.979230, -15.718832])      # mean of W_o rows 1 and 2
Wo = np.array([o0, o1, o1, o0])
Wp = np.maximum(Wx, 0); Wn = np.maximum(-Wx, 0)

def model(x, iv):
    pos = iv("pos", np.maximum(x[:, 0], 0))    # x1+
    neg = iv("neg", np.maximum(-x[:, 0], 0))   # x1-
    h1 = pos[:, None] * Wp + neg[:, None] * Wn
    h2 = iv("h2", np.maximum(x[:, 1:2] * Wx + h1 @ Wh, 0))
    h3 = iv("h3", np.maximum(x[:, 2:3] * Wx + h2 @ Wh, 0))
    return h3 @ Wo

ALIGN = {"pos": ["h1"], "neg": ["h1"], "h2": ["h2"], "h3": ["h3"]}
