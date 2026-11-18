
import numpy as np
DESCRIPTION=("Read x as a map f on 6 points, f(p)=x[p]. Layer 0 lets every position gather the points that map "
 "to it, so each position's residual carries whether that position is a fixed point and how many preimages "
 "its value has; layer 1 sums that census and the readout scores three structural classes from it. Census "
 "cells (shares measured on all 6^6 inputs): a 2-cycle -> class 1 (97%); else at least two fixed points "
 "(95%) or a fixed point whose value has two or more preimages (84-99%) -> class 0; a lone fixed point "
 "(77%) or no fixed point (91%) -> class 2. Class 3 is a near-tie with class 2 (margins ~0.1 logit) and is "
 "not reproduced; positions 1,4 and the value 5 carry the strongest weights.")
def model(x,iv):
    h=np.arange(6.)
    tok=[iv("tok%d"%p,(x[:,p][:,None]==h).astype(float)) for p in range(6)]
    T=np.stack([t@h for t in tok],1).astype(int)
    fix=(T==h).astype(float)
    cnt=np.stack([(T==j).sum(1) for j in range(6)],1)
    pre=np.take_along_axis(cnt,T,1)
    nxt=np.take_along_axis(T,T,1)
    cyc=((nxt==h)&(T!=h)).astype(float)
    fp=iv("fp",fix.sum(1))
    t2=iv("t2",cyc.sum(1)/2)
    nP=iv("nP",(fix*pre).max(1))
    h1=t2>0; h0=fp>0; lo=(fp==1)&(nP<1.5)&(~h1)
    s0=(~h1)*(4.0*h0*(~lo)+2.0*fp)
    s1=6.0*h1+2.0*t2
    s2=(~h1)*(3.0*(~h0)+5.0*lo)
    z=np.zeros(len(x))
    return np.stack([s0,s1,s2,z,z-1,z-2],1)
ALIGN={"tok%d"%p:["embed"] for p in range(6)}
