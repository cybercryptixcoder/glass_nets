
import numpy as np

DESCRIPTION = (
 "A two-step, two-channel leaky accumulator. Each step splits the input into a positive "
 "channel P = relu(x) and a negative channel N = relu(-x) (the input weights of the two hidden "
 "units have opposite signs, so after step 1 exactly one unit is alive and holds the signed value). "
 "Propagation: each channel decays by a fixed fraction of itself (-0.382 for P, -0.752 for N) and is "
 "re-driven by the opposite channel of the previous step (gain 2.085 into P, 1.142 into N), and the "
 "input of the current step enters with unit gain: P2 = relu(x2 - 0.382 P1 + 2.085 N1), "
 "N2 = relu(-x2 + 1.142 P1 - 0.752 N1). The readout compares the two final channels with the N channel "
 "weighted much more heavily (ratio 4.353): class 1 iff 4.353*N2 > P2, i.e. the network scores "
 "class 1 when the step-2 negative channel carries more evidence than the step-2 positive channel."
)

K00 = -0.38212355971336365
K10 =  2.0851662362478636
K01 =  1.1415294349709517
K11 = -0.7515226602554321
RATIO = 4.353341987440024

def model(x, iv):
    x1 = x[:, 0]
    x2 = x[:, 1]
    P1 = iv("P1", np.maximum(x1, 0.0))
    N1 = iv("N1", np.maximum(-x1, 0.0))
    P2 = iv("P2", np.maximum(x2 + K00 * P1 + K10 * N1, 0.0))
    N2 = iv("N2", np.maximum(-x2 + K01 * P1 + K11 * N1, 0.0))
    d = RATIO * N2 - P2
    return np.stack([-0.5 * d, 0.5 * d], axis=1)

ALIGN = {"P1": ["h1"], "N1": ["h1"], "P2": ["h2"], "N2": ["h2"]}
