
import numpy as np
DESCRIPTION=("x is a map f on {0..5}, f(p)=x[p]; the decision is a census of its cycle structure. Any 2-cycle -> class 1 "
 "(98%); else a fixed point with no cycle of length>=3 -> class 0 (95%); else class 2 (fixed point plus a long cycle, 66%, "
 "or no fixed point, 91%), except in the fixed-point-plus-long-cycle case, where the value at the last position decides: "
 "if it leads to a fixed point (f(f(v))=f(v)) and is not 5, or it is 5 and there are >=2 fixed points, the map is class 0. "
 "Layer 0 at the last position looks up position x_5, and the readout scores how the map iterates from that value. "
 "Class 3 (1.2%) is a near tie, not reproduced.")
def model(x,iv):
    h=np.arange(6.)
    T=np.stack([iv("tok%d"%p,(x[:,p][:,None]==h)*1.).argmax(1) for p in range(6)],1)
    fp=iv("fp",(T==h).sum(1)*1.)
    A=np.zeros((len(T),6,6)); A[np.arange(len(T))[:,None],h.astype(int)[None,:],T]=1.
    d=[]; Ak=A
    for k in range(6):
        d.append(np.einsum('bii->b',Ak)); Ak=Ak@A
    d1,d2,d3,d4,d5,d6=d
    t2=iv("t2",((d2-d1)/2).round())
    L=iv("long",(((d3-d2)/3+(d4-d2)/4+(d5-d1)/5+(d6-d3-(d2-d1))/6)>0.5)*1.)
    v=iv("v5",T[:,5]*1.); fv=np.take_along_axis(T,v.astype(int)[:,None],1)[:,0]
    ff=np.take_along_axis(T,fv[:,None],1)[:,0]
    h1=(t2>0.5)*1.
    C=((fp>0.5)&(L<0.5))|((fp>0.5)&(L>0.5)&((ff==fv)&(v!=5)|(v==5)&(fp>1.5)))
    c0=C*1.
    s0=2*c0*(1-h1); s1=3*h1; s2=(1-h1)*(1-c0); z=0*s2
    return np.stack([s0,s1,s2,z,z-1,z-2],1)
ALIGN={"tok%d"%p:["embed"] for p in range(6)}
