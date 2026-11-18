
import numpy as np
DESCRIPTION = """The network is a four-step rectified linear filter over the input stream, with no biases.

Step 1 is a pure SIGN SPLIT of x1: units 0 and 2 are fixed negative multiples of relu(-x1) and units 1 and 3
fixed positive multiples of relu(x1). So h1 is two-dimensional, spanned exactly by the two variables
rp = relu(x1) (positive part of x1) and rn = relu(-x1) (negative part).

Steps 2-4 repeat one operation per unit: add the new input with that unit's fixed sign (+1 for units 1,3;
-1 for units 0,2), add a fixed recurrent mixture of the previous state, and rectify. I write each unit
divided by its own input gain, so the input always enters with weight exactly +1 or -1. Unit 1 and unit 2 are
on for essentially every input (their rectifier almost never binds), so they act as linear ACCUMULATORS of the
stream; units 0 and 3 are genuinely rectified (unit 3 is off about two thirds of the time), so they hold
one-sided evidence about the negative side of the stream. The state is amplified roughly four-fold per step,
which is why the influence of the inputs falls off with lag - x1 matters most, then x2, x3, and x4 hardly at
all (zeroing x4 changes only ~3% of the decisions).

Output: a fixed linear readout of the final state. Class 0 is never the winner; the decision is essentially a
comparison of the two channels (the positive-part units 1,3 versus the negative-part units 0,2), i.e. of how
much positive versus negative evidence the filtered stream accumulated."""

# state_t = relu(sign_in * x_t + state_{t-1} @ A), each unit normalised by its input gain
sign_in = np.array([-1.0, 1.0, -1.0, 1.0])
a0 = [-0.14264020323753357, 1.0828034791487526, 0.83932104448680522, -0.17300708482800303]
a1 = [-0.10474758928136924, 2.4333465099334717, 1.1333474623052897, 3.3435863606755403]
a2 = [3.0928773362482209, 5.5512848442949041, -0.75254112482070923, -15.778918441240823]
a3 = [-0.09469674637877197, 0.18850035472917726, -0.2495214017670849, -2.1696624755859375]
A = np.array([a0, a1, a2, a3])
# out = state @ B
b0 = [-5.2139570112984757, -1.8946360140630816, 6.7795145038248279, -1.0868093300964006]
b1 = [-2.5518606798538457, 1.0751775723152335, 0.62814031912390433, 1.633452054257134]
b2 = [-2.6823235611075233, 1.9672527358359133, -1.2100820231893579, -2.2397929486140526]
b3 = [0.25106954379051238, 0.47475166091593124, 0.2153190870871402, -0.34939166956338852]
B = np.array([b0, b1, b2, b3])

def model(x, iv):
    rp = iv("rp", np.maximum(x[:, 0], 0.0))      # positive part of x1
    rn = iv("rn", np.maximum(-x[:, 0], 0.0))     # negative part of x1
    z = np.stack([-rn, rp, -rn, rp], axis=1)     # step-1 state = relu(sign_in * x1)
    for t in range(1, 4):
        z = iv("s%d" % (t + 1), np.maximum(sign_in * x[:, [t]] + z @ A, 0.0))
    return z @ B

ALIGN = {"rp": ["h1"], "rn": ["h1"], "s2": ["h2"], "s3": ["h3"], "s4": ["h4"]}
