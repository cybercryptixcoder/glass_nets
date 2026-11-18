import numpy as np
DESCRIPTION=("Logits = resid(pos 3) @ U. h = E[token] + P[position] (P[3]=0); each layer is one head: "
 "s = softmax_j(h_i M h_j^T), h = h + s @ h @ V. E,P,M0,V0,M1,V1,U are 3-digit constants, U column 1 = 0 (gauge).")
T=np.array([.205,.453,.153,2.83,2.24,-.0289,2.11,2.36,.323,3.1,1.59,1.26,
 -.931,-2.12,-.55,-1.87,-1.55,3.1,-5.97,-.123,1.31,
 .187,-.394,.0983,-.448,-1.26,1.45,-2.83,-1.65,2.65,
 .0304,-.774,.13,1.06,-.102,.246,.746,1.11,-.107,
 .258,2.94,.266,-.195,-2.24,-.212,-.278,1.27,3.81,
 -1.88,1.57,-1.81,.0451,.442,-.405,-.169,-9.85,5.85,
 37.4,35.6,36.9,6.3,3.63,4.53,1.06,-4.67,-2.59]).reshape(22,3)
E=T[:4];P=np.vstack([T[4:7],np.zeros(3)])
M0=T[7:10];V0=T[10:13];M1=T[13:16];V1=T[16:19]
U=np.zeros((3,4));U[:,[0,2,3]]=T[19:22]
def model(x,iv):
    n=len(x)
    h=iv('emb',(E[x]+P).reshape(n,-1)).reshape(n,4,3)
    for k,(M,V) in enumerate(((M0,V0),(M1,V1))):
        s=h@M@h.transpose(0,2,1);s=np.exp(s-s.max(-1,keepdims=True));s=s/s.sum(-1,keepdims=True)
        h=h+iv('a%d'%k,(s@h@V).reshape(n,-1)).reshape(n,4,3)
        h=iv('r%d'%k,h.reshape(n,-1)).reshape(n,4,3)
    o=h[:,3]@U
    return np.stack([o[:,0],o[:,0]*0,o[:,1],o[:,2]],1)
ALIGN={'emb':['embed'],'a0':['attn0.out'],'r0':['resid0.mid'],'a1':['attn1.out'],'r1':['resid1.mid']}
