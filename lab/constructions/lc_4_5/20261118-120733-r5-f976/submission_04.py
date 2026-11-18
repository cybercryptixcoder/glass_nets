
import numpy as np
DESCRIPTION=("Residual = token + position embedding (5x4). Two softmax attention layers, no mask or bias: "
"score=(x A_l x^T)/2, A_l=W_Ql W_Kl^T, softmax over keys, then x+=attn(x) V_l, V_l=W_Vl W_Ol. Layer 0 is nearly one-hot, "
"layer 1 soft. Only position 4 is read: out=(x4+a1.(x V1)) W_U. Matrices are the network's own integers/1000; an attention row "
"keeps 4 weights, the 5th is 1 minus their sum.")
t=np.array("REPLACE".split(),float)/1000
E=t[:20].reshape(5,4);P=t[20:40].reshape(5,4);M0=t[40:56].reshape(4,4);V0=t[56:72].reshape(4,4)
M1=t[72:88].reshape(4,4);V1=t[88:104].reshape(4,4);U=t[104:].reshape(4,5)
def sm(s):
    s=np.exp(s-s.max(-1,keepdims=True));return s/s.sum(-1,keepdims=True)
def rw(a):return np.concatenate([a,1-a.sum(-1,keepdims=True)],-1)
def model(x,iv):
    b=len(x)
    e=iv('e',(E[x]+P).ravel()).reshape(b,5,4)
    A=iv('A',sm((e@M0)@e.transpose(0,2,1)/2)[...,:4].ravel()).reshape(b,5,4)
    o=iv('o',(rw(A)@e@V0).ravel()).reshape(b,5,4)
    Z=iv('Z',(e+o).ravel()).reshape(b,5,4)
    a=iv('a',sm((Z@M1)@Z.transpose(0,2,1)/2)[:,4,:4])
    q=iv('q',np.einsum('bk,bkd->bd',rw(a),Z)@V1)
    return iv('r',Z[:,4]+q)@U
ALIGN={'e':['embed'],'A':['attn0.pattern'],'o':['attn0.out'],'Z':['resid0.mid'],'a':['attn1.pattern'],'q':['attn1.out'],'r':['resid1.mid']}
