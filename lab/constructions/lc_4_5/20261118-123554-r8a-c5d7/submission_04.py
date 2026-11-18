import numpy as np
D="Zero."
def model(x,iv):
 v=iv('e',x[:,0].astype(float))
 return np.zeros((len(x),5))
ALIGN={'e':['embed']}
