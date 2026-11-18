
import numpy as np
DESCRIPTION="constant class 0; one dummy variable"
def model(x,iv):
    v=iv("v",np.zeros(x.shape[0]))
    return np.stack([v+1,0*v,0*v,0*v,0*v],1)
ALIGN={"v":["attn0.pattern"]}
