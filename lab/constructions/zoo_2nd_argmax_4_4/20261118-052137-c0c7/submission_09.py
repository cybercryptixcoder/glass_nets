
import numpy as np

DESCRIPTION = """Step 1: W_x = s*(1,-1,-1,1) with s=|W_x|>0, so h1 = s*(relu(x1), relu(-x1), relu(-x1), relu(x1)): the first step only splits x1 by
sign and mirrors it (units 1,2 identical, units 0,3 identical) - a 2-dimensional state.
Gauge: measuring the state in units of s (h' = h/s) removes s from the input weights (they become the integers (1,-1,-1,1)); the recurrence
becomes h'_t = relu(x_t*(1,-1,-1,1) + h'_{t-1} @ Wh2) with Wh2[i,j] = s_i*Wh[i,j]/s_j, homogeneous of degree 1 in the inputs.
Readout: Out = h'_4 @ diag(s)Wo; only class-score differences decide, so store Wod[:,k] = s_i*(Wo[i,k]-Wo[i,0]) (12 numbers).
28 numbers total (16 recurrent + 12 readout) for 27 degrees of freedom; steps 2-4 are the measured generic recurrent transition."""

WH2 = np.array([[-0.213737,  1.273340, -0.069905,  2.318979],
                [ 5.417269,  1.320339, -0.533874, 14.224694],
                [-4.427177, -1.289346,  0.530658,-14.462238],
                [-0.013221,  0.084671,  0.482565, -0.752541]])
WOD = np.array([[-1.005312, -4.188698, -1.088185],
                [-3.579383, -4.633421, -0.978169],
                [ 4.911669,  4.261590,  0.563860],
                [ 0.793057,  2.319706,  0.658305]])

def model(x, iv):
    x = np.asarray(x, dtype=float)[:, :4]
    xp = iv("x1pos", np.maximum(x[:, 0], 0))
    xm = iv("x1neg", np.maximum(-x[:, 0], 0))
    h = np.stack([xp, xm, xm, xp], 1)
    for t in range(1, 4):
        pre = x[:, t:t+1] * np.array([1, -1, -1, 1]) + h @ WH2
        h = iv("h%d" % (t+1), np.maximum(pre, 0))
    return np.concatenate([np.zeros((h.shape[0], 1)), h @ WOD], 1)

ALIGN = {"x1pos": ["h1"], "x1neg": ["h1"], "h2": ["h2"], "h3": ["h3"], "h4": ["h4"]}
