import numpy as np
DESCRIPTION = "probe: same architecture, all constants at 2 significant digits."
def sgn(v):
    pass
E=[[0.21,0.45,0.15],[2.8,2.2,-0.029],[2.1,2.4,0.32],[3.1,1.6,1.3]]
P=[[-0.93,-2.1,-0.55],[-1.9,-1.5,3.1],[-6.0,-0.12,1.3],[0,0,0]]
A=[[0.19,-0.39,0.098],[-0.45,-1.3,1.4],[-2.8,-1.7,2.6]]
B=[[0.030,-0.77,0.13],[1.1,-0.10,0.25],[0.75,1.1,-0.11]]
C=[[0.26,2.9,0.27],[-0.20,-2.2,-0.21],[-0.28,1.3,3.8]]
D=[[-1.9,1.6,-1.8],[0.045,0.44,-0.40],[-0.17,-9.8,5.8]]
U=[[38.,36.,37.],[6.3,3.6,4.5],[1.1,-4.7,-2.6]]
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
