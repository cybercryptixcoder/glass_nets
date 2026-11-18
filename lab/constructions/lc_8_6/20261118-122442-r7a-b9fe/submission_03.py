
import numpy as np
DESCRIPTION=("Read x as a map f on the six points 0..5, f(p)=x[p]. The decision is a census of the map's cycle structure. "
 "Layer 0 moves the value found at position p into position p (attention reads the map at its own position), so a "
 "position's residual states how often its value is hit; layer 1 at the last position combines those counts, and the "
 "readout scores the cycle structure. Measured on all 6^6 inputs: a 2-cycle anywhere -> class 1 (98% of those maps); "
 "else a fixed point and no cycle of length >=3 -> class 0 (95%); else (a fixed point together with a long cycle 66%, "
 "or no fixed point at all 91%) -> class 2. Class 3 (1.2%) is a near three-way tie between classes 0, 2 and 3 decided "
 "by logit margins of order 0.1 and is not reproduced.")
def model(x,iv):
    h=np.arange(6.0)
    T=[]
    for p in range(6):
        t=iv("tok%d"%p,(x[:,p][:,None]==h).astype(float))
        T.append(t.argmax(1))
    T=np.stack(T,1)                                   # token ids [B,6]
    ar=np.arange(6)
    fix=(T==ar)                                       # p with f(p)=p
    cnt=np.stack([(T==j).sum(1) for j in range(6)],1) # how many p have value j
    pre=np.take_along_axis(cnt,T,1)                   # preimage count of f(p)
    fp=iv("fp",fix.sum(1).astype(float))              # number of fixed points
    nP=iv("nP",(fix*pre).max(1).astype(float)/3.0)    # largest preimage count at a fixed point
    A=np.zeros((len(T),6,6)); A[np.arange(len(T))[:,None],ar[None,:],T]=1.0
    Ak=A; d1=np.einsum('bii->b',Ak); d=[d1]
    for k in range(5):
        Ak=Ak@A; d.append(np.einsum('bii->b',Ak))
    d1,d2,d3,d4,d5,d6=d
    t2=iv("t2",((d2-d1)/2.0).round())                 # number of 2-cycles
    long=iv("long",(((d3-d2)/3+(d4-d2)/4+(d5-d1)/5+(d6-d3-(d2-d1))/6)>0.5).astype(float))
    h1=(t2>0.5).astype(float)
    c0=((fp>0.5)&(long<0.5)).astype(float)
    s0=2*c0*(1-h1); s1=3*h1; s2=(1-h1)*(1-c0); z=0*s2
    return np.stack([s0,s1,s2,z,z-1,z-2],1)
ALIGN={"tok%d"%p:["embed"] for p in range(6)}
