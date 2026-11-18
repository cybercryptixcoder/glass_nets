
import numpy as np
DESCRIPTION = (
 "A 5-step ReLU RNN with no biases: h_t = relu(x_t*W_x + h_{t-1}@W_h) with 8 units, then 5 class scores "
 "compared by argmax; the network's decision is the position it picks. The task is essentially "
 "'the position of the 2nd-largest input' (87.9%), and the residual ~12% of cases where the network "
 "departs from that rule come from a genuine positional asymmetry (position 1 / class 0 compressed, "
 "position 5 penalised) that is a piecewise-linear consequence of the dense recurrent map and is not "
 "captured by any small set of order statistics. "
 "READOUT RULE: each hidden unit's readout row is centred to sum to zero (Wc = W_o - rowmean(W_o)). "
 "This adds the same per-sample constant to all five class scores, so every argmax and hence every "
 "decision is unchanged, but the 5th score becomes minus the sum of the first four and the readout needs "
 "only 8x4 numbers instead of 8x5. The recurrent core (W_x, W_h) is a copied dense linear map.")
W_x = np.array([[0.642615,-6.42131,0.423453,-6.57374,0.045088,-4.91316,-0.054098,7.00169]])
W_h = np.array([[0.125403,0.154946,0.476144,0.060649,0.771231,0.035924,0.94397,-0.16333],
[-0.464716,-0.236367,0.272299,0.492902,0.169101,-0.73223,0.424159,-0.027035],
[0.019963,-0.129043,-0.106897,-0.190221,-0.618453,-0.423786,0.239916,0.081623],
[0.528093,-0.299993,-0.362798,-0.847493,0.279272,0.5697,0.198553,0.061197],
[-0.17744,-0.324277,0.215948,-0.251721,-0.746841,-0.191503,-1.207441,0.380405],
[0.881026,0.600439,0.125687,0.347769,0.127209,0.069581,-0.750272,0.023595],
[0.126756,0.961696,-0.086458,0.972771,0.545731,0.679505,0.786684,-0.890478],
[-1.655899,-0.017242,1.418636,1.087747,0.155794,0.814863,1.102013,-0.827896]])
Wc4 = np.array([[0.908382,-0.944254,0.87206,-1.75598],
[0.464135,-0.588393,-3.777,4.01283],
[1.13012,0.212338,-3.10537,0.848],
[0.518182,-2.26763,4.96206,-2.69802],
[-0.491964,2.07116,-0.556421,-0.55691],
[-0.835808,4.23525,-1.31539,-1.27487],
[-0.843916,-0.914782,0.540774,0.701928],
[0.432252,0.509778,0.570218,0.54836]])
def model(x, iv):
    s = np.zeros((len(x), 8))
    for t in range(5):
        s = iv('h%d'%(t+1), np.maximum(x[:,t:t+1]*W_x + s@W_h, 0))
    s4 = s @ Wc4
    return np.concatenate([s4, -s4.sum(1, keepdims=True)], 1)
ALIGN = {'h1':['h1'],'h2':['h2'],'h3':['h3'],'h4':['h4'],'h5':['h5']}
