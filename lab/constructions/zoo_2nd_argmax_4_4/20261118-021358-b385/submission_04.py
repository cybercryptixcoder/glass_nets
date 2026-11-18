
import numpy as np
DESCRIPTION = """A 4-step rectified linear filter over the input stream, written in per-unit units (each hidden unit
divided by its own input gain, so the input always enters with weight +1 or -1).

Step 1: the state depends only on x1 and is a pure SIGN SPLIT of it. Units 0 and 2 receive -x1 (they are
-relu(-x1)) and units 1 and 3 receive +x1 (they are relu(x1)); i.e. the two negative-sign units hold the
negative part of x1 and the two positive-sign units its positive part. So two variables, rp = relu(x1) and
rn = relu(-x1), span the whole first state exactly.

Steps 2-4: every unit repeats the same operation - add the new input with its fixed sign (+1 or -1), add a
fixed recurrent mixture of the previous state, and rectify. Unit 1 and unit 2 are on for essentially every
input (their rectifier is nearly never active), so they act as linear accumulators of the stream; units 0 and
3 are genuinely rectified (unit 3 is off about two thirds of the time), i.e. they hold one-sided evidence.
The influence of the inputs decays with lag: x1 matters most, then x2, x3, and x4 hardly at all.

Output: a fixed linear readout of the final state; class 0 is never the winner and the decision is largely a
comparison between the two channels (class 1 wins when the positive-part/negative-part balance of the stream
favours units 1,3, class 2 when it favours units 0,2)."""

sign_in = np.array([-1.0, 1.0, -1.0, 1.0])          # input enters unit j as sign_in[j] * x_t
gain    = np.array([2.0354, 0.9975, 2.2207, 0.3939]) # h = gain * state (used only to read step 1)
# recurrent mixture: state_t = relu(sign_in[:,None]*x_t + state_{t-1} @ A)
a0 = [-0.1426, 1.0828, 0.8393, -0.1730]
a1 = [-0.1047, 2.4333, 1.1333, 3.3436]
a2 = [ 3.0929, 5.5513, -0.7525, -15.7789]
a3 = [-0.0947, 0.1885, -0.2495, -2.1697]
A = np.array([a0, a1, a2, a3])
# readout: out = state @ B
b0 = [-5.2140, -1.8946, 6.7795, -1.0868]
b1 = [-2.5519, 1.0752, 0.6281, 1.6335]
b2 = [-2.6823, 1.9673, -1.2101, -2.2398]
b3 = [ 0.2511, 0.4748, 0.2153, -0.3494]
B = np.array([b0, b1, b2, b3])

def model(x, iv):
    rp = iv("rp", np.maximum(x[:, 0], 0.0))     # positive part of x1
    rn = iv("rn", np.maximum(-x[:, 0], 0.0))    # negative part of x1
    z = iv("s1", np.stack([-rn, rp, -rn, rp], axis=1))   # step-1 state = relu(sign_in * x1)
    for t in range(1, 4):
        z = iv("s%d" % (t + 1), np.maximum(sign_in * x[:, [t]] + z @ A, 0.0))
    return z @ B

ALIGN = {"rp": ["h1"], "rn": ["h1"], "s1": ["h1"], "s2": ["h2"], "s3": ["h3"], "s4": ["h4"]}
