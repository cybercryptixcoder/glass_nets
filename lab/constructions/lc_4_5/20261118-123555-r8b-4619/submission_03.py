import numpy as np
DESCRIPTION = "constant"
def model(x, iv):
    v = iv('v', (x[:,0]*0+1).astype(float))
    return np.zeros((x.shape[0],5))
ALIGN = {'v':['embed']}
