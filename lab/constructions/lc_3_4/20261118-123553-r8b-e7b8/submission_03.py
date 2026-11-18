import numpy as np
DESCRIPTION = ("Logits are a linear readout of the residual stream at position 3. "
 "h0 = E[token] + P[position] (P[3]=0). Each layer is one attention head: "
 "pattern = softmax over keys of the bilinear score h_i M h_j^T, and the head adds pattern@h@VO. "
 "Layer 0 mixes all positions; layer 1's readout row (query 3) mixes the same residual0 values. "
 "Matrices are the two score forms and the two value-output forms; constants are rounded to 3 digits.")
E=[[0.205,0.453,0.153],[2.83,2.24,-0.0289],[2.11,2.36,0.323],[3.1,1.59,1.26]]
P=[[-0.931,-2.12,-0.55],[-1.87,-1.55,3.1],[-5.97,-0.123,1.31],[0,0,0]]
A=[[0.187,-0.394,0.0983],[-0.448,-1.26,1.45],[-2.83,-1.65,2.65]]
B=[[0.0304,-0.774,0.13],[1.06,-0.102,0.246],[0.746,1.11,-0.107]]
C=[[0.258,2.94,0.266],[-0.195,-2.24,-0.212],[-0.278,1.27,3.81]]
D=[[-1.88,1.57,-1.81],[0.0451,0.442,-0.405],[-0.169,-9.85,5.85]]
U=[[37.4,35.6,36.9],[6.3,3.63,4.53],[1.06,-4.67,-2.59]]
E=np.array(E);P=np.array(P);A=np.array(A);B=np.array(B);C=np.array(C);D=np.array(D);U=np.array(U)
def model(x, iv):
    n=len(x)
    h=iv('emb',(E[x]+P).reshape(n,-1)).reshape(n,4,3)
    s=h@A@h.transpose(0,2,1); s=np.exp(s-s.max(-1,keepdims=True)); s=s/s.sum(-1,keepdims=True)
    a0=iv('a0',(s@h@B).reshape(n,-1)).reshape(n,4,3)
    h=iv('r0',(h+a0).reshape(n,-1)).reshape(n,4,3)
    s=h@C@h.transpose(0,2,1); s=np.exp(s-s.max(-1,keepdims=True)); s=s/s.sum(-1,keepdims=True)
    a1=iv('a1',(s@h@D).reshape(n,-1)).reshape(n,4,3)
    h=iv('r1',(h+a1).reshape(n,-1)).reshape(n,4,3)
    o=h[:,3]@U
    return np.stack([o[:,0],0*o[:,0],o[:,1],o[:,2]],1)
ALIGN={'emb':['embed'],'a0':['attn0.out'],'r0':['resid0.mid'],'a1':['attn1.out'],'r1':['resid1.mid']}
