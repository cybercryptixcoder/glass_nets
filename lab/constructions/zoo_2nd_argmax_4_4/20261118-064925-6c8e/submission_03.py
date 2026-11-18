
import numpy as np
DESCRIPTION = ("Recurrent construction in a sign-normalised hidden basis: h_t[i] = relu(+-x_t + sum_j Wh[j,i] h_{t-1}[j]) "
 "with input gains exactly +1,-1,-1,+1; class scores are h4 projected on the three score DIFFERENCES from class 1 "
 "(class 1's own score set to 0, which does not change the argmax).")
Wx = np.array([1.0,-1.0,-1.0,1.0])
Wh = np.array([[-0.21373653, 1.27334036, -0.06990484, 2.31897888],
               [ 5.41726918, 1.32033944, -0.53387362,14.22469429],
               [-4.42717719,-1.28934600, 0.53065825,-14.46223782],
               [-0.01322133, 0.08467120, 0.48256451, -0.75254112]])
# columns: score(class0)-score(class1), score(class2)-score(class1), score(class3)-score(class1)
D = np.array([[-2.9461,  9.3289,  0.2429],
              [-4.7116,  1.3875, -3.4240],
              [ 7.0540,  0.9336,  6.2442],
              [ 5.8853,-11.3294,  1.0000]])
def model(x, iv):
    h = np.zeros((x.shape[0],4))
    for t in range(4):
        h = np.maximum(x[:,t:t+1]*Wx + h@Wh, 0.0)
        h = iv("h%d"%(t+1), h)
    s0 = h@D[:,0]; s2 = h@D[:,1]; s3 = h@D[:,2]
    return np.column_stack([s0, np.zeros_like(s0), s2, s3])
ALIGN = {"h1":["h1"],"h2":["h2"],"h3":["h3"],"h4":["h4"]}
