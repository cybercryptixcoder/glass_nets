
import numpy as np
DESCRIPTION = (
 "A 5-step ReLU RNN with no biases: h_t = relu(x_t*W_x + h_{t-1}@W_h), 8 units, then 5 class scores "
 "compared by argmax, so the decision is the position the network picks. The task it implements is "
 "'the position of the 2nd-largest input' (87.9%); the extra fidelity comes from a positional asymmetry "
 "that is a genuine piecewise-linear consequence of the recurrent map, not capturable by order statistics. "
 "READOUT RULE: every hidden unit's readout row is centred to sum to zero (Wc = W_o - rowmean(W_o)), which "
 "shifts all five scores of a sample by the same amount and so leaves every argmax unchanged; hence the 5th "
 "score equals minus the sum of the first four and the readout is stored as 8x4 numbers, not 8x5. "
 "The recurrent core (W_x, W_h) is a copied dense linear map; every weight is written to 4 significant "
 "figures, which keeps 99.85% of decisions.")
W_x = np.array([[0.6426,-6.421,0.4235,-6.574,0.04509,-4.913,-0.0541,7.002]])
W_h = np.array([[0.1254,0.1549,0.4761,0.06065,0.7712,0.03592,0.944,-0.1633],
[-0.4647,-0.2364,0.2723,0.4929,0.1691,-0.7322,0.4242,-0.02704],
[0.01996,-0.129,-0.1069,-0.1902,-0.6185,-0.4238,0.2399,0.08162],
[0.5281,-0.3,-0.3628,-0.8475,0.2793,0.5697,0.1986,0.0612],
[-0.1774,-0.3243,0.2159,-0.2517,-0.7468,-0.1915,-1.207,0.3804],
[0.881,0.6004,0.1257,0.3478,0.1272,0.06958,-0.7503,0.0236],
[0.1268,0.9617,-0.08646,0.9728,0.5457,0.6795,0.7867,-0.8905],
[-1.656,-0.01724,1.419,1.088,0.1558,0.8149,1.102,-0.8279]])
Wc4 = np.array([[0.9084,-0.9443,0.8721,-1.756],
[0.4641,-0.5884,-3.777,4.013],
[1.13,0.2123,-3.105,0.848],
[0.5182,-2.268,4.962,-2.698],
[-0.492,2.071,-0.5564,-0.5569],
[-0.8358,4.235,-1.315,-1.275],
[-0.8439,-0.9148,0.5408,0.7019],
[0.4323,0.5098,0.5702,0.5484]])
def model(x, iv):
    s = np.zeros((len(x), 8))
    for t in range(5):
        s = iv('h%d'%(t+1), np.maximum(x[:,t:t+1]*W_x + s@W_h, 0))
    s4 = s @ Wc4
    return np.concatenate([s4, -s4.sum(1, keepdims=True)], 1)
ALIGN = {'h1':['h1'],'h2':['h2'],'h3':['h3'],'h4':['h4'],'h5':['h5']}
