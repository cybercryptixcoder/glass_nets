
import numpy as np

DESCRIPTION = """A 4-unit ReLU recurrent net that is HOMOGENEOUS OF DEGREE 1 (out(2x)=2*out(x)) and whose first step is trivial:
W_x = s*(1,-1,-1,1) with s=|W_x|>0, so h1 = s*(relu(x1), relu(-x1), relu(-x1), relu(x1)) - the input is split by sign and mirrored
(units 1,2 identical, units 0,3 identical), a 2-dimensional state.
Measuring the state in units of s (h'=h/s) removes the input weights completely (they become the integer vector (1,-1,-1,1)),
and the recurrence becomes h'_t = relu(x_t*(1,-1,-1,1) + h'_{t-1} @ Wh2), Wh2[i,j] = s_i*Wh[i,j]/s_j.
The readout Out = h'_4 @ diag(s)Wo, and only differences between class scores decide, so it is stored as
Wod[:,k] = s_i*(Wo[i,k]-Wo[i,0]) - 12 numbers instead of 16. Decision = argmax of (0, Wod-columns).
Total 28 numbers for 27 degrees of freedom, and this is the floor: the recurrent map is generic (its four singular values are
21.7, 1.5, 0.6, 0.4; the state is full rank 4 at every step; 0.3% noise on Wh2 changes 5% of decisions; every entry is
load-bearing; all pairwise sums/products/ratios of the 28 numbers were searched for exact relations down to 1e-7 and none exist)."""

WH2 = np.array([[-0.21373653411865234,  1.2733403617150982 , -0.069904836275869561,  2.3189788782239518 ],
                [ 5.4172691761178342 ,  1.3203394412994385 , -0.53387362302281438, 14.224694293973265  ],
                [-4.4271771877571515 , -1.2893459957251092 ,  0.53065824508666992, -14.462237823581248 ],
                [-0.013221333672257933, 0.084671199711396375, 0.48256451076042106, -0.75254112482070923]])
WOD = np.array([[-1.0053117351332297, -4.188698071519781 , -1.0881847235553188],
                [-3.5793833360603173, -4.6334209410021   , -0.9781685167478855],
                [ 4.911668529802768 ,  4.261589819079788 ,  0.5638595669231279],
                [ 0.7930567403633653,  2.3197064268733953,  0.6583054379315385]])

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
