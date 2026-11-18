
DESCRIPTION = (
 "Step 1: h1 = relu(x1*W_x) is exactly a linear function of the two numbers pos=max(x1,0) and neg=max(-x1,0) "
 "(each unit is 0 or a fixed multiple of x1 with a fixed sign). "
 "Step 10: the last hidden state is almost rank one - h10 ~ s10*u with u a fixed positive direction (>=97% of its "
 "variance on one axis, growing from 84% at step 1 to 99% at step 10), and u is essentially the mean direction "
 "(cos 0.99997). s10 is a rectified leaky accumulator of the inputs: negative inputs feed the positive state "
 "units (W_x is negative on units 6,10,13,14), and each step multiplies the accumulated magnitude by about 1.7, "
 "clamped at zero by the ReLU. The readout of the (positive) state gives a fixed logit pattern in which class 5 "
 "dominates, so the network always answers class 5."
)
K = [-0.63, -0.64, 0.00, -0.47, 0.20, 1.00, 0.33, 0.02, -0.36, -0.05]

def model(x, iv):
    x1 = x[:, 0]
    pos1 = iv("pos1", np.maximum(x1, 0.0))
    neg1 = iv("neg1", np.maximum(-x1, 0.0))
    s = iv("s1", 10.0 * neg1)          # state magnitude after step 1
    for t in range(1, 10):
        s = np.maximum(1.7 * s + 6.0 * np.maximum(-x[:, t], 0.0), 0.0)
        if t == 8:
            s = iv("s10", s)
    out = s[:, None] * np.array(K)[None, :]
    return out

ALIGN = {"pos1": ["h1"], "neg1": ["h1"], "s1": ["h1"], "s10": ["h10"]}
