
import numpy as np
DESCRIPTION = ("Recurrent construction in a sign-normalised hidden basis. Write p=relu(x1), n=relu(-x1): the first state is exactly "
 "h1 = p*(1,0,0,1)+n*(0,1,1,0) (only two rays, since the input gains are +1,-1,-1,+1). Then h_t = relu(+-x_t + h_{t-1}@Wh) "
 "for t=2,3,4. The decision is the argmax of the four class scores; only their differences from class 1 are needed, so the "
 "model returns [d0.h4, 0, d2.h4, d3.h4] with the three difference vectors (which are scale-free).")
Wx = np.array([1.0,-1.0,-1.0,1.0])
Wh = np.array([[-0.21373653, 1.27334036, -0.06990484, 2.31897888],
               [ 5.41726918, 1.32033944, -0.53387362,14.22469429],
               [-4.42717719,-1.28934600, 0.53065825,-14.46223782],
               [-0.01322133, 0.08467120, 0.48256451, -0.75254112]])
P = np.array([1.0,0.0,0.0,1.0])      # h1 contribution of relu(x1)
N = np.array([0.0,1.0,1.0,0.0])      # h1 contribution of relu(-x1)
# columns: score(class0)-score(class1), score(class2)-score(class1), score(class3)-score(class1)
D = np.array([[ 0.386477, -1.223815, -0.031859],
              [ 1.376043, -0.405210,  1.0     ],
              [-1.888198, -0.249936, -1.671358],
              [-0.304880,  0.586810, -0.051803]])
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
