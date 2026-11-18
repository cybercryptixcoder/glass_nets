
import numpy as np
DESCRIPTION=("x is a map f on {0..5}, f(p)=x[p]. Decision = cycle census: any 2-cycle -> class 1 (98%); else a fixed point "
 "with no cycle of length>=3 -> class 0 (95%); else class 2 (fixed point plus a long cycle, 66%, or no fixed point, 91%), "
 "except in the fixed-point-plus-long-cycle case, where the value v=x_5 at the last position decides: if f(f(v))=f(v) and "
 "v!=5, or v=5 and the map has >=2 fixed points, it is class 0. Layer 0 at the last position looks up position x_5, and the "
 "readout scores the map's iterates from that value. Class 3 (1.2%) is a near tie, not reproduced.")
def model(x,iv):
    h=np.arange(6.); I=np.arange(len(x))
    T=np.stack([iv("tok%d"%p,(x[:,p][:,None]==h)*1.).argmax(1) for p in range(6)],1)
    fp=iv("fp",(T==h).sum(1)*1.)
    A=np.zeros((len(T),6,6)); A[I[:,None],h.astype(int)[None,:],T]=1.
    d=[]; Ak=A
    for _ in range(6):
        d.append(np.einsum('bii->b',Ak)); Ak=Ak@A
    d1,d2,d3,d4,d5,d6=d
    t2=iv("t2",((d2-d1)/2).round())
    L=iv("long",(((d3-d2)/3+(d4-d2)/4+(d5-d1)/5+(d6-d3-(d2-d1))/6)>0.5)*1.)
    v=iv("v5",T[:,5]*1.).astype(int); fv=T[I,v]
    h1=(t2>0.5)*1.
    C=((fp>0.5)&(L<0.5))|((fp>0.5)&(L>0.5)&((T[I,fv]==fv)&(v!=5)|(v==5)&(fp>1.5)))
    c0=C*1.; s2=(1-h1)*(1-c0); z=0*s2
    return np.stack([2*c0*(1-h1),3*h1,s2,z,z-1,z-2],1)
ALIGN={"tok%d"%p:["embed"] for p in range(6)}
