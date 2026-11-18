
import numpy as np
DESCRIPTION = ("Recurrent construction, sign-normalised. Each hidden unit i is a nonneg accumulator: "
 "h_t[i] = relu(+-x_t + sum_j Wh[j,i]*h_{t-1}[j]) with the input weights exactly +1,-1,-1,+1 "
 "(unit scales chosen so |Wx|=1); the four scores are h_4 @ Wo. "
 "h1 depends only on the sign and size of x1 (two rays).")
Wx = np.array([1.0,-1.0,-1.0,1.0])
Wh = np.array([[-0.21373653, 1.27334036, -0.06990484, 2.31897888],
               [ 5.41726918, 1.32033944, -0.53387362,14.22469429],
               [-4.42717719,-1.28934600, 0.53065825,-14.46223782],
               [-0.01322133, 0.08467120, 0.48256451, -0.75254112]])
Wo = np.array([[ 1.63345205, 0.62814032,-2.55524602, 0.54526733],
               [ 2.39362799,-1.18575534,-2.23979295, 1.41545948],
               [-2.45849558, 2.45317295, 1.80309424,-1.89463601],
               [-1.00769711,-0.21464037, 1.31200932,-0.34939167]])
def model(x, iv):
    h = np.zeros((x.shape[0],4))
    for t in range(4):
        h = np.maximum(x[:,t:t+1]*Wx + h@Wh, 0.0)
        h = iv("h%d"%(t+1), h)
    return h@Wo
ALIGN = {"h1":["h1"],"h2":["h2"],"h3":["h3"],"h4":["h4"]}
