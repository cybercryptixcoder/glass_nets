
import numpy as np
DESCRIPTION=("x is a map f on {0..5}, f(p)=x[p]; the decision is a census of its cycle structure. A 2-cycle anywhere -> "
 "class 1 (98% of such maps); else a fixed point with no cycle of length>=3 -> class 0 (95%); else (fixed point plus a "
 "long cycle, 66%, or no fixed point, 91%) -> class 2. Layer 0 reads the map at each position, layer 1 and the readout "
 "score the cycles. Class 3 (1.2%) is a near tie between classes 0, 2, 3 and is not reproduced.")
def model(x,iv):
    h=np.arange(6.)
    T=np.stack([iv("tok%d"%p,(x[:,p][:,None]==h)*1.).argmax(1) for p in range(6)],1)
    fix=T==h
    cnt=np.stack([(T==j).sum(1) for j in range(6)],1)
    pre=np.take_along_axis(cnt,T,1)
    fp=iv("fp",fix.sum(1)*1.)
    A=np.zeros((len(T),6,6)); A[np.arange(len(T))[:,None],h.astype(int)[None,:],T]=1.
    d=[np.einsum('bii->b',A)]; Ak=A
    for _ in range(5):
        Ak=Ak@A; d.append(np.einsum('bii->b',Ak))
    d1,d2,d3,d4,d5,d6=d
    t2=iv("t2",((d2-d1)/2).round())
    L=iv("long",(((d3-d2)/3+(d4-d2)/4+(d5-d1)/5+(d6-d3-(d2-d1))/6)>0.5)*1.)
    h1=(t2>0.5)*1.; c0=((fp>0.5)&(L<0.5))*1.
    s0=2*c0*(1-h1); s1=3*h1; s2=(1-h1)*(1-c0); z=0*s2
    return np.stack([s0,s1,s2,z,z-1,z-2],1)
ALIGN={"tok%d"%p:["embed"] for p in range(6)}
