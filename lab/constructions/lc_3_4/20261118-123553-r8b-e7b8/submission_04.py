import numpy as np
DESCRIPTION = ("The output is a linear readout of the residual stream at position 3. "
 "h0 = E[token]+P[position] with P[3]=0. Each layer is one head: softmax over keys of the "
 "bilinear score h_i M h_j^T, adding pattern@h@VO to the residual. Layer 0 mixes the four "
 "positions; layer 1, at the readout position, mixes the same residual0 values. "
 "Only the two 3x3 score forms and the two 3x3 value-output forms are needed (rounded to 3 digits).")
E=np.array([[.205,.453,.153],[2.83,2.24,-.0289],[2.11,2.36,.323],[3.1,1.59,1.26]])
P=np.array([[-.931,-2.12,-.55],[-1.87,-1.55,3.1],[-5.97,-.123,1.31],[0,0,0]])
A=np.array([[.187,-.394,.0983],[-.448,-1.26,1.45],[-2.83,-1.65,2.65]])
B=np.array([[.0304,-.774,.13],[1.06,-.102,.246],[.746,1.11,-.107]])
C=np.array([[.258,2.94,.266],[-.195,-2.24,-.212],[-.278,1.27,3.81]])
D=np.array([[-1.88,1.57,-1.81],[.0451,.442,-.405],[-.169,-9.85,5.85]])
U=np.array([[37.4,35.6,36.9],[6.3,3.63,4.53],[1.06,-4.67,-2.59]])
def model(x,iv):
    n=len(x)
    h=iv('e',(E[x]+P).reshape(n,-1)).reshape(n,4,3)
    for i,(M,V) in enumerate(((A,B),(C,D))):
        s=h@M@h.transpose(0,2,1);s=np.exp(s-s.max(-1,keepdims=True));s/=s.sum(-1,keepdims=True)
        h=h+iv('a%d'%i,(s@h@V).reshape(n,-1)).reshape(n,4,3)
        h=iv('r%d'%i,h.reshape(n,-1)).reshape(n,4,3)
    o=h[:,3]@U
    return np.stack([o[:,0],o[:,0]*0,o[:,1],o[:,2]],1)
ALIGN={'e':['embed'],'a0':['attn0.out'],'r0':['resid0.mid'],'a1':['attn1.out'],'r1':['resid1.mid']}
