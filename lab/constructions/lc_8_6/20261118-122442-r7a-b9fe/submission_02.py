
import numpy as np
DESCRIPTION=("Read x as a map f on 6 points, f(p)=x[p]. Layer 0 moves into each position the points that map onto it, "
 "so a position's residual carries how many preimages its value has; layer 1 iterates that relation and the readout "
 "scores the cycle structure of the map. Measured on all 6^6 inputs: a 2-cycle anywhere -> class 1 (98%); otherwise a "
 "fixed point with no cycle of length >=3 -> class 0 (95%); otherwise (a fixed point together with a long cycle, or no "
 "fixed point at all) -> class 2 (66% / 91%). Class 3 is a near-tie with class 2 (logit margin ~0.1) and is not reproduced.")
def model(x,iv):
    h=np.arange(6.)
    A=np.stack([iv("tok%d"%p,(x[:,p][:,None]==h).astype(float)) for p in range(6)],1)
    d=[np.einsum('bii->bi',A)]; Ak=A
    for k in range(5):
        Ak=Ak@A; d.append(np.einsum('bii->bi',Ak))
    d1,d2,d3,d4,d5,d6=[a.sum(1) for a in d]
    fp=iv("fp",d1)
    t2=iv("t2",(d2-d1)/2)
    long=iv("long",(((d3-d2)/3+(d4-d2)/4+(d5-d1)/5+(d6-d3-(d2-d1))/6)>0).astype(float))
    h1=(t2>0).astype(float); c0=(fp>0).astype(float)*(1-long)
    s0=2*c0*(1-h1); s1=3*h1; s2=(1-h1)*(1-c0); z=0*s2
    return np.stack([s0,s1,s2,z,z-1,z-2],1)
ALIGN={"tok%d"%p:["embed"] for p in range(6)}
