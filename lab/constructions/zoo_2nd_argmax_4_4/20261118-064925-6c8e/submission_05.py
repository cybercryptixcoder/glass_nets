
import numpy as np
DESCRIPTION = ("Exact mechanism in a sign-normalised hidden basis. With p=relu(x1), n=relu(-x1) the first state is exactly "
 "h1 = p*(1,0,0,1) + n*(0,1,1,0): the four input gains are +1,-1,-1,+1 (unit scales chosen so), so the four hidden units are "
 "nonneg leaky accumulators h_t = relu(+-x_t + h_{t-1}@Wh) (no biases, all 16 recurrent couplings needed). The decision is "
 "argmax of the four class scores; only their differences from class 1 matter, so the model returns [d0.h4, 0, d2.h4, d3.h4]. "
 "Unit scales, the common score shift and the common scale of the score differences are fixed to gauge, which leaves the 27 "
 "numbers below (16 recurrent + 11 score-difference) = the network's 27 decision-relevant degrees of freedom.")
Wx = np.array([1.0,-1.0,-1.0,1.0])
P = np.array([1.0,0.0,0.0,1.0])
N = np.array([0.0,1.0,1.0,0.0])
Wh = np.array([[-0.213736534119, 1.273340361715,-0.069904836276, 2.318978878224],
               [ 5.417269176118, 1.320339441299,-0.533873623023,14.224694293973],
               [-4.427177187757,-1.289345995725, 0.530658245087,-14.462237823581],
               [-0.013221333672, 0.084671199711, 0.482564510760,-0.752541124821]])
D = np.array([[ 0.953772171334,-3.020182886703,-0.078624318557],
              [ 3.395878210871,-1.0              , 2.467857699874],
              [-4.659860812152,-0.616750965691,-4.124908772225],
              [-0.752398905547, 1.448382561829,-0.127842974292]])
def model(x, iv):
    p = iv("h1_p", np.maximum(x[:,0],0.0))
    n = iv("h1_n", np.maximum(-x[:,0],0.0))
    h = p[:,None]*P + n[:,None]*N
    for t in range(1,4):
        h = np.maximum(x[:,t:t+1]*Wx + h@Wh, 0.0)
        h = iv("h%d"%(t+1), h)
    s0 = h@D[:,0]; s2 = h@D[:,1]; s3 = h@D[:,2]
    return np.column_stack([s0, np.zeros_like(s0), s2, s3])
ALIGN = {"h1_p":["h1"], "h1_n":["h1"], "h2":["h2"], "h3":["h3"], "h4":["h4"]}
